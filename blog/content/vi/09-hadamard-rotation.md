---
title: Phép quay Hadamard làm W4A8 chính xác hơn W4A16, và một layer của Llama-3.2-3B làm nó hỏng
date: 2026-11-23
order: 9
summary: Quay đầu vào của mọi layer tuyến tính (và weight của nó) bằng ma trận Hadamard theo khối xóa gần hết thiệt hại do activation INT8 và còn tự trả lại bằng độ chính xác. Nó làm gì trên bảy model, layer nơi nó thất bại, quyết định theo từng layer sửa được, và những ý tưởng không hiệu quả.
---

Lượng tử hóa activation INT8 theo token là cái giá của [đường W4A8](08-w4a8-beats-marlin.html) nhanh hơn, và người trả giá là các kênh ngoại lai: một kênh lớn quyết định scale của token và phần còn lại của hàng mất độ phân giải. Cách chữa quen thuộc từ QuaRot và SpinQuant là nhân đầu vào với một ma trận trực chuẩn `B` và weight với cùng ma trận đó, `(xB)(WB)^T = xW^T`, để layer không đổi về mặt toán học nhưng cả hai vế đều phẳng hơn khi được lượng tử hóa.

## Phiên bản đủ rẻ

Quay đầy đủ một layer rộng 4096 là một phép nhân 4096 x 4096. Một **ma trận Walsh-Hadamard chéo khối**, mỗi khối 128 x 128 cho mỗi nhóm 128 đầu vào, là một butterfly: 7 tầng cộng và trừ cho mỗi khối, không có phép nhân. `WB` được tính một lần khi lượng tử hóa weight (lúc nạp, hoặc trong GPTQ với Hessian được quay thành `B^T H B`). Ở phía activation, một warp biến đổi một khối bằng shuffle, và trên 64 hàng phép biến đổi nằm *ngay trong kernel lượng tử hóa INT8*: hàng đã quay chờ trong shared memory để lấy scale, nên không tốn thêm một lượt đọc activation (kernel gộp giống từng bit với việc quay rồi mới lượng tử hóa, và có test). Chi phí đo được: 2 đến 4 % của một bước decode ở 1 đến 64 request, 1 đến 2 % ở trên đó.

## Nó mang lại gì

Perplexity Wikitext-2 qua vLLM (kernel thật sự chạy; weight round-to-nearest online, activation INT8):

{{chart:rot_ppl}}

| model | fp16 | W4A8, không quay | W4A8, quay (mặc định) |
|---|--:|--:|--:|
| Qwen2.5-1.5B | 11,932 | 13,949 | **13,376** |
| Qwen2.5-0.5B | 18,10 | 23,148 | 21,395 |
| OLMo-2-0425-1B | 17,188 | 20,843 | 18,987 |
| Llama-3.2-1B | 18,353 | 22,930 | **21,539** |
| Phi-3.5-mini | không vừa bộ nhớ | 9,472 | **8,718** |
| Qwen2.5-3B | không vừa bộ nhớ | 12,767 | **11,830** |
| Llama-3.2-3B | không vừa bộ nhớ | 16,640 | 16,658 |

Với weight GPTQ (Qwen2.5-1.5B, hiệu chỉnh trên wikitext-2 train nên số tuyệt đối hơi đẹp): W4A16 12,605, W4A8 12,864, **W4A8 quay 12,331**. Thiệt hại do activation INT8, +0,42 và +0,63 perplexity trên hai cỡ Qwen trong mô phỏng lượng tử hóa giả, giảm còn +0,01 và +0,04 trên nền weight 4 bit đã quay; và một checkpoint W4A8 có quay chính xác hơn cả checkpoint W4A16 không quay. Giảm nửa nhóm của các scale nguyên xuống 64 (`quantforge_w4a8_g64`) cho thêm 0,21 online và 0,08 với GPTQ, đổi lại 2 đến 3 % tốc độ.

## Layer làm nó hỏng

Quay *mọi thứ* làm Llama-3.2-3B tệ đi: 16,640 thành **17,752**. Chỉ quay các layer có K bằng 8192 cho 17,71, chỉ quay các layer còn lại cho 16,74, nên thủ phạm là các `down_proj`; một bảng sai số theo từng layer trên activation thật (forward trên CPU, kernel thật trên GPU) chỉ ra đúng một layer:

| `down_proj` layer 1 của Llama-3.2-3B | sai số đầu ra tương đối |
|---|--:|
| W4A8 thường | 0,054 |
| đã quay | 0,080 |

Đầu vào của nó có một token với **chuẩn gấp 530 lần trung vị**, chính là attention sink. Phép quay dàn kênh khổng lồ duy nhất đó ra cả khối 128, và token duy nhất mà model phụ thuộc vào giờ bị lượng tử hóa với sai số trải đều cả hàng. Không có gì trong weight đánh dấu một layer như vậy (chuẩn cột bình thường), nên không có cách nào biết nếu thiếu activation. Hai cách sửa:

* **GPTQ có dữ liệu hiệu chỉnh**, nên nó quyết định theo từng layer: với mỗi nhóm layer tuyến tính chung đầu vào, nó so sai số đầu ra của lượng tử hóa có quay và không quay trên vài token của mỗi cửa sổ, gồm cả token có chuẩn lớn nhất, rồi ghi các layer giữ nguyên vào `config.json` (`unrotated_layers`; với Llama-3.2-3B là `layers.1.mlp.down_proj` và `layers.23.self_attn.o_proj`). GPTQ W4A8 của bản 3B: 16,537 không quay, 16,377 khi quyết định theo layer.
* **Lượng tử hóa online không có dữ liệu**, nên nó theo một quy tắc: `down_proj` của bốn layer decoder đầu tiên không được quay. Bảng trên là bằng chứng cho quy tắc đó, bảy model, và nó là quy tắc chứ không phải chứng minh: với họ model tôi chưa chạy, `quantforge_w4a8_plain` là mốc để so sánh.

## Những thứ không hiệu quả

* **Chọn phép quay riêng cho từng layer** trong số các ứng viên rẻ (đơn vị, đổi dấu ngẫu nhiên, hoán vị kênh) theo sai số đầu ra trên token hiệu chỉnh: 13,612 so với 13,606 của Hadamard cố định. Phần còn thiếu so với fp16 là weight 4 bit, không phải cơ sở quay.
* **Activation FP8 (E4M3) thay INT8** trên lõi tensor Ada: không xây. Với phép quay, activation INT8 chỉ tốn +0,01 đến +0,04 perplexity, tốc độ FP8 và INT8 trên chip này như nhau, và dialect NVVM của LLVM 18 không có mma FP8.
* **Weight 3 bit** với phép quay và tìm clip: GPTQ 17,2 thường, 14,6 có quay, 13,6 có quay và clip, so với 13,1 / 12,5 / 12,4 ở 4 bit cùng dữ liệu hiệu chỉnh. Model 3 bit ngang checkpoint GPTQ 4 bit công khai của Marlin, nhưng so với 4 bit cùng hiệu chỉnh thì tốn thêm 0,5 đến 1,2 perplexity để bớt 25 % byte, và kernel không có giải nén 3 bit (hai mặt phẳng bit đọc bằng `tensor.extract` sẽ chậm hơn kernel decode 4 bit). 2 bit không dùng được.
* **Tìm clip cho scale theo kênh** của định dạng thật: -0,08, +0,08 và -0,26 trên ba model, nên để tắt.
* **KV cache fp8** (của chính vLLM): nó nhân đôi dung lượng KV (90k lên 171k token trên card 6 GB) và làm hỏng Qwen2.5 với scale mặc định (12,33 lên 24,96; Marlin 13,64 lên 28,73), đó là chuyện attention của vLLM.

## Hai lỗi đáng một đoạn

Cache biên dịch của vLLM được khóa theo *tên* phương thức lượng tử hóa, không theo biến môi trường. Công tắc quay đầu tiên của tôi là một biến môi trường; lần chạy thứ hai dùng lại đồ thị đã biên dịch của lần đầu, và perplexity ra 10^7. Cách sửa là một thiết lập làm đổi đồ thị được trace phải là một tên lượng tử hóa khác (`quantforge_w4a8_plain`, `quantforge_w4a8_g64`). Và hàm giả (fake) của một custom op phải trả về stride của kết quả thật: `empty_like` của đầu vào không liên tục thì không, và kết quả sai một cách im lặng dưới `torch.compile`.
