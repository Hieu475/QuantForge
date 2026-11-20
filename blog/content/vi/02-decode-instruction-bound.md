---
title: Điều gì làm kernel decode nhanh: một bảng ablation, và một đính chính về con số đẹp quá mức
date: 2026-11-20
order: 2
summary: Vì sao kernel decode bị giới hạn bởi số lệnh, mỗi pass đáng giá bao nhiêu (đo lạnh, hôm nay), các cạm bẫy khi benchmark trên GPU laptop, và đính chính tuyên bố nhanh hơn Marlin 1.30 lần trước đây không tái hiện được.
---

> **Đính chính (2026-11-20).** Phiên bản trước của bài này, và của README dự án, khẳng định kernel decode sinh ra cần **38 micro giây** cho một layer 4096 x 4096, "nhanh hơn Marlin 1.30 lần". **Con số đó sai**: nó không tái hiện được. Chạy lại đúng harness benchmark đó trên chính commit đã ghi nó cho 50.2 micro giây, và harness hôm nay cho 50.0 micro giây so với 49.6 của Marlin, tỷ lệ **0.99**. Con số là một sai lệch đo đạc của giai đoạn phát triển mà tôi không dựng lại chính xác được. Mọi thứ dưới đây đã được đo lại hôm nay. Trang bạn đang đọc thay câu chuyện cũ ("từ 53 xuống 38 micro giây") bằng phần còn đứng vững: phương pháp chẩn đoán và một bảng ablation.

Một GEMM decode batch 1 đọc mỗi weight INT4 đúng một lần, nên lẽ ra nó chạy ở tốc độ của hệ thống bộ nhớ. Với layer 4096 x 4096 đó là 8.4 MB weight cộng 0.25 MB scale và 0.25 MB zero point: 8.65 MB, khoảng 45 micro giây ở 192 GB/s mà thông số card cho phép.

## Bước 0: đo đúng thứ cần đo, và kiểm tra lại

Các cạm bẫy này áp dụng cho mọi công việc kernel trên GPU laptop, và lời đính chính ở trên cho thấy vì sao chúng quan trọng.

1. **Đo bằng đồng hồ tường che mất kernel.** Launch từ Python tốn 45 đến 270 micro giây, nhiều hơn kernel. Mọi số là thời gian device từ trace CUPTI (profiler của PyTorch).
2. **L2 còn ấm làm kernel trông nhanh hơn thực tế.** RTX 4050 có 24 MB L2, nên weight INT4 8 MB nằm lại trong cache giữa các vòng benchmark và benchmark đo chính cache. Trước mỗi lần gọi tôi flush L2.
3. **Cách flush cũng quan trọng.** Ghi vào buffer 256 MB để lại các dòng dirty trong L2, việc write-back làm kernel *kế tiếp* chậm 20 đến 50 phần trăm; đọc buffer thì loại cache sạch. Sau đó phải chạm lại activation, nếu không một harness tình cờ copy chúng (để pad) sẽ đưa cho kernel một A còn nóng.
4. **Xung nhịp trôi.** GPU dùng chung với desktop và xung nhịp dao động tới 30 phần trăm giữa các phiên. So các kernel *xen kẽ trong cùng một lần chạy*, không bao giờ so với con số của hôm qua. (Lời đính chính là một ví dụ.)
5. **Kiểm tra rằng các byte đến từ DRAM.** `ncu --cache-control all --metrics dram__bytes_read.sum` trên kernel decode báo **9.2 MB** đọc từ DRAM cho 8.65 MB weight và scale; ở 50 micro giây đó là khoảng **184 GB/s**, về cơ bản là giới hạn DRAM của card. Thay flush 256 MB bằng flush 1 GB không làm đổi thời gian của kernel nào (50.3 micro giây cho cả hai). Script là `benchmarks/l2_residency_check.py`.

Vậy ở 4096 x 4096 kernel sinh ra và Marlin **cùng ở giới hạn DRAM**, và hòa chính là kết quả mong đợi.

## Bước 1: đếm lệnh, không đếm byte

Trong lúc phát triển kernel, `ncu` cho bảng lệnh theo từng opcode (`sass__inst_executed_per_opcode`): khoảng **7.7 lệnh thread cho mỗi weight**, gồm một lệnh `I2F` (đổi số nguyên sang số thực) cho mỗi weight. Một kernel phải stream 16.7 triệu weight trong 45 micro giây trên 20 SM có khoảng 10 slot lệnh cho mỗi weight, nên với 7.7 nó gần hết slot issue trước khi hết băng thông: kernel decode *bị giới hạn bởi số lệnh trước khi bị giới hạn bởi độ trễ*. (Số lệnh không phụ thuộc trạng thái cache, nên quan sát này không bị ảnh hưởng bởi lời đính chính.)

## Bước 2: các load scale bị load tám lần, và pass ngừng khớp

Dequantisation của một work item 16 byte (8 cột) cần tám scale và tám zero point. Một pass (`vectorize-global-loads`) gộp tám load vô hướng thành một load 16 byte. Hai lỗi tác động lẫn nhau:

* biểu thức chỉ số ánh xạ một hàng K sang nhóm của nó (`k / 128`) không được gấp thành hằng với các K tile lớn hơn, vì phân tích cận dùng đối số block-size của launch thay vì kích thước tile: **tám load scale và zero point trùng lặp mỗi vòng lặp**;
* sau khi sửa, các load của tám work item trong một thread trở nên *giống hệt nhau*, common-subexpression elimination gộp chúng, và bộ vector hóa chỉ thay **load đầu tiên**. Pass dequantisation đóng gói (bước sau) khớp mẫu "tám lane liên tiếp trích từ cùng một vector load", và với bảy trong tám item nó không còn thấy hình dạng đó; chúng rơi về đường vô hướng với một `I2F` cho mỗi weight.

Thay *tất cả* các bản sao đã khôi phục dạng đóng gói. Bài học là về compiler: một tính chất hiệu năng phụ thuộc vào chuỗi các pass khớp mẫu là mong manh, nên dự án có một test FileCheck mà việc duy nhất là khẳng định PTX không chứa `cvt.rn.f16`. (Tôi không còn nêu mức tăng micro giây cho hai bản sửa này, vì các số đo gốc chính là thứ không còn đứng vững.)

## Bước 3: dequantisation bằng hai lệnh cho hai weight

Dạng đóng gói là mẹo quen thuộc mà Marlin dùng. Một nibble `q` (0 đến 15) đặt vào các bit mantissa thấp của hằng half-precision `1152 = 0x6480` cho ra **đúng `1152 + q`**, không cần lệnh chuyển đổi. Với hai weight cùng lúc, trong một thanh ghi 32 bit có hai lane half-precision:

```text
packed  = (nibble_lo) | (nibble_hi << 16) | 0x64806480        // một tổ hợp kiểu LOP3
zero'   = (byte zero ^ 0x80) | 0x6400 cho mỗi lane              // 1152 + zero_point, dựng theo cùng cách
diff    = packed - zero'          // sub.f16x2: chính xác, một số nguyên trong [-15, 15]
weight  = diff * scale            // mul.f16x2: phép làm tròn duy nhất
```

Cả hai toán hạng mang cùng độ lệch, phép trừ chính xác (số nguyên dưới 2048), và phép làm tròn duy nhất xảy ra ở phép nhân, nên kết quả **giống hệt từng bit** với `(q - zero) * scale` tính theo sách giáo khoa bằng fp16.

## Bước 4: bảng ablation, đo hôm nay (L2 lạnh, 1 x 4096 x 4096)

Cách tốt nhất để nói mỗi pass đáng giá bao nhiêu là tắt nó đi. Cùng kernel, cùng shape, L2 đã flush, thời gian device:

| cấu hình | thời gian | tương đối |
|---|--:|--:|
| mọi pass (kernel decode đã autotune) | **50 us** | 1.00x |
| `vectorize-dequant=false` (dequantisation vô hướng, một `I2F` mỗi weight) | 58 us | chậm 1.16x |
| `prefetch-distance=0` (không look-ahead cho load weight) | 61 us | chậm 1.22x |
| `vectorize-loads=false` (load weight / scale / zero-point vô hướng) | **156 us** | **chậm 3.1x** |
| Marlin (tham chiếu) | 49.6 us | |
| cuBLAS fp16 trên weight đã dequantise (gấp 4 lần byte) | 181 us | |

Khi weight đã nằm sẵn trong L2 (không flush) kernel đầy đủ mất 49 micro giây: ở kích thước này nó bị giới hạn bởi độ trễ (ít warp, mỗi warp có chuỗi phụ thuộc dài), không phải bởi băng thông, nên thời gian lạnh và ấm gần như bằng nhau.

Thứ hạng mang nhiều thông tin. **Load vector hóa là cái lớn nhất**: không có load 16 byte kernel phát lệnh load nhiều gấp ba và chạy chậm gấp ba. Dequantisation đóng gói và look-ahead một tile mỗi cái đáng 16 đến 22 phần trăm. Các pass hoàn toàn *không giúp* trong chế độ này: stage shared-memory thứ hai cho decode, đẩy các store dequantisation xuống, look-ahead hai tile trở lên (các load đã đủ sớm) và split-K.

## Bước 5: wave quantisation trên các shape rộng

Các shape rộng cư xử khác. Trong lúc phát triển, một lần quét kích thước lưới theo băng thông đạt được cho thấy khoảng 183 GB/s khi lưới vừa các block đang cư trú của GPU (80 block 128 cột với K tile 128 trên card này), và khoảng 151 GB/s ngay khi có làn thứ hai không đầy: layer 4096 x 11008 có 86 block cột cho 80 slot. K tile nhỏ hơn (64 thay vì 128, giảm shared memory mỗi block) cùng look-ahead một tile lấy lại một phần mất mát. Đây là số đo thời phát triển; kết quả lạnh trên các shape rộng nằm ở bài tổng quan: 0.93 của Marlin cho 1 x 4096 x 11008 sau phần bổ sung bên dưới, và 0.97 cho 1 x 11008 x 4096.

### Bổ sung: K tile đi theo số wave (1 x 4096 x 11008: từ 150 xuống 135 micro giây)

Shape rộng là nơi duy nhất kernel decode thua rõ (0.85 của Marlin, và chậm hơn gemlite 8 phần trăm). Một lần quét độ rộng đầu ra N ở K = 4096 với hai K tile quan trọng cho thấy lý do (L2 lạnh, micro giây, `block-tile-n=128`, look-ahead một tile):

| N | số block cột | K tile 32 | K tile 128 |
|--:|--:|--:|--:|
| 9216 | 72 | 114 | 115 |
| 10240 | 80 | **124** | 137 |
| 10752 | 84 | **132** | 148 |
| 11008 | 86 | **136** | 150 |
| 11520 | 90 | 161 | 165 |
| 12288 | 96 | **153** | 174 |
| 13312 | 104 | 207 | **166** |
| 14336 | 112 | 211 | **176** |
| 16384 | 128 | 228 | **219** |

Với K tile sâu 128 card giữ 80 block cùng lúc (4 mỗi SM): 81 đến 96 block cột chỉ vượt một wave một chút và các block lẻ chạy một mình. K tile sâu 32 cần ít shared memory hơn mỗi block, nên mọi block đều cư trú và layer chạy trong một wave, nhanh hơn 8 đến 12 phần trăm. Từ khoảng 104 block, wave thứ hai tồn tại dù chọn tile nào và hiệu quả tốt hơn của tile sâu lại thắng. Schedule mặc định giờ dùng tile sâu 32 cho N từ 10240 đến 12288 (MLP rộng 11008 của Llama-7B là trường hợp hiển nhiên), autotuner có thêm ứng viên đó, và kết quả là **1 x 4096 x 11008: từ 150 xuống 135.7 micro giây (0.93 của Marlin, chậm hơn gemlite 5 phần trăm), 16 x 4096 x 11008: từ 149 xuống 135.5 (0.94)**. Đây là quy tắc khớp với số SM và dung lượng shared memory của GPU này: trên card khác cửa sổ sẽ dịch chuyển, và đó là việc của autotuner.

## Rút ra gì

1. Ở giới hạn DRAM, hòa với một kernel viết tay là kết quả đúng, và tuyên bố hơn nó 30 phần trăm trên một kernel bị giới hạn bộ nhớ cần được kiểm tra bằng `dram__bytes_read` trước khi tin.
2. Đếm lệnh trước: tính toán băng thông nói kernel lẽ ra ở 45 micro giây, chỉ histogram opcode giải thích vì sao các phiên bản đầu không như vậy.
3. Hãy ablation, đừng kể chuyện: bảng ở trên nhiều thông tin hơn mọi câu chuyện về các bước.
4. Giữ một test hồi quy cho mọi tính chất hiệu năng phụ thuộc vào tương tác giữa các pass.
