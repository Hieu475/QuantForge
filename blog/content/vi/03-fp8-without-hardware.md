---
title: Dequantise weight FP8 (E4M3) không cần phần cứng hỗ trợ: một mẹo bit gói gọn trong bốn phép số nguyên
date: 2026-11-20
order: 3
summary: Thêm weight FP8 vào compiler dù backend NVPTX của LLVM 18 không có kiểu FP8 như thế nào, vì sao phép chuyển đổi là chính xác, cách vector hóa, và nó tốn gì.
---

FP8 là định dạng weight 8 bit phổ biến còn lại. **E4M3** có 1 bit dấu, 4 bit mũ (bias 7) và 3 bit mantissa, độ lớn tối đa 448, không có vô cực, và chỉ một mã hóa NaN (`0x7F` và `0xFF`). Nó giữ độ chính xác tương đối trên các độ lớn khác nhau, phù hợp với weight hơn lưới INT8 đều khi các giá trị trải rộng.

Compiler không thể dùng con đường hiển nhiên. Toolchain ở đây là LLVM / MLIR 18: backend NVPTX không có kiểu số thực E4M3 và không lộ lệnh `cvt.f16x2.e4m3x2` của PTX mới hơn. Thay vì chờ LLVM mới hơn, tôi tìm một phép chuyển đổi chỉ cần lệnh số nguyên.

## Quan sát

Lấy byte FP8 `s eeee mmm` và xét mẫu 16 bit

```text
f16 bits = (s << 15) | ((byte & 0x7F) << 7)
         = s 0 0 e e e e m m m 0 0 0 0 0 0 0      (bit 15 .. bit 0)
```

Đọc như một số half-precision IEEE, 4 bit mũ của FP8 rơi vào bốn bit thấp của trường mũ 5 bit của f16, 3 bit mantissa rơi vào đầu trường mantissa 10 bit. Bias mũ của f16 là 15 và của E4M3 là 7, nên số half-precision là

```text
2^(e - 15) * (1 + m/8)  =  2^-8 * 2^(e - 7) * (1 + m/8)
   (số half-precision)     (2^-8 lần giá trị FP8)
```

tức là **đúng 2^-8 lần giá trị FP8**. Điều này cũng đúng với subnormal: với `e = 0` giá trị FP8 là `m/8 * 2^-6` và mẫu f16 là một subnormal có giá trị `(m * 128 / 1024) * 2^-14 = m/8 * 2^-14`, lại là 2^-8 lần giá trị FP8. Vậy

```text
value(byte) = bitcast_f16( ((byte & 0x7F) << 7) | ((byte & 0x80) << 8) ) * 256
```

và phép nhân với 256 là chính xác (giá trị tối đa là 448, xa giới hạn của f16). Scale của nhóm được nhân sau đó, và đó là phép làm tròn duy nhất, nên kết quả giống hệt việc dequantise ở độ chính xác cao hơn rồi làm tròn một lần.

Chi phí: một AND, một shift, một AND, một shift, một OR, một bit-cast, và hai phép nhân, một trong hai là nhân với hằng. NaN (`0x7F`) được đọc là 480 thay vì NaN; weight đã lượng tử hóa không bao giờ chứa nó.

## Đã kiểm chứng trên cả 256 byte

Một unit test dựng mẹo này bằng PyTorch cho mọi giá trị byte và so với phép chuyển đổi `torch.float8_e4m3fn`: cả 254 byte không-NaN giống hệt nhau, và `0x7F` được đọc là 480.0 như đã ghi. Test nằm cạnh các test layout của repository.

## Làm cho nhanh: f16x2 đóng gói

Bản cài đặt đầu tiên là cây vô hướng ở trên (khoảng 9 lệnh số nguyên cho mỗi weight). Kernel bị giới hạn bởi bộ nhớ ở decode, nên ở 1 x 4096 x 4096 nó đã ngang INT8 (97 so với 100 micro giây). Nhưng trong cả một model, các layer nhỏ bị giới hạn bởi số lệnh, và bản vô hướng decode 155 token/s so với 165 của INT8 trên Qwen2.5-0.5B.

Pass `vectorize-dequant` có sẵn, nhận ra tám cây dequantisation cấp cho một vector store và phát ra số học đóng gói, được mở rộng để nhận ra cây FP8. Với hai byte `b0, b1` nó dựng một từ 32 bit `x = b0 | (b1 << 16)` (mỗi byte trong nửa 16 bit riêng), rồi làm AND / shift / OR trên **cả hai nửa cùng lúc**:

```text
mag  = (x << 7) & 0x3F803F80
sgn  = (x << 8) & 0x80008000
half2 = mag | sgn                       // hai giá trị f16 bằng 2^-8 lần các giá trị FP8
w     = (half2 * 256.h2) * scale.h2     // hai mul.f16x2
```

Dạng đóng gói **giống hệt từng bit** với dạng vô hướng (được kiểm bằng các test đầu-cuối và các case sanitizer) và đưa decode cả model về 165 token/s. Một test FileCheck khẳng định kernel decode không chứa lệnh shift 16 bit nào (dạng vô hướng có 128 lệnh), nên một thay đổi sau này làm hỏng việc khớp mẫu sẽ bị bắt ngay.

## FP8 tốn gì và được gì

| | INT8 | FP8 (E4M3) | ghi chú |
|---|--:|--:|---|
| kernel decode 1 x 4096 x 4096 | 101 us | 99 us | weight 8 bit: gấp đôi byte của INT4 |
| kernel decode 1 x 4096 x 11008 | 272 us | 271 us | |
| prefill 512 x 4096 x 4096 | 564 us | 584 us | chậm hơn 4 phần trăm |
| perplexity Qwen2.5-0.5B (fp16: 19.93) | 19.91 | 20.10 | round-to-nearest, group 128 |
| decode Qwen2.5-0.5B (CUDA graph) | 165 tok/s | 165 tok/s | sau khi vector hóa |

Trên model này INT8 và FP8 tương đương về tốc độ và gần nhau về chất lượng, cả hai nhanh hơn fp16 khoảng 28 phần trăm (165 so với 129 token/s) trên một model cỡ này. Chúng giảm một nửa byte weight của fp16 thay vì giảm còn một phần tư như INT4, nên INT4 vẫn là định dạng cho tốc độ. FP8 là định dạng cho các checkpoint đã phát hành sẵn ở đó, hoặc khi mất chất lượng của INT4 là không chấp nhận được.

## Những gì nó không làm

Chỉ *weight* là FP8. Activation vẫn là fp16 và phép nhân-cộng chạy trên đường Tensor Core fp16. Dùng Tensor Core FP8 của Ada (`mma` với toán hạng `e4m3`) để nhân đôi tốc độ số học là một dự án khác: nó cần activation FP8, scale activation theo tensor và cách phát các lệnh đó từ LLVM 18. Chưa làm.

Các số FP8 cũng đến từ lượng tử hóa round-to-nearest checkpoint fp16; chưa có checkpoint FP8 nào ngoài đời được chạy qua nó.
