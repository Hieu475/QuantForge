---
title: Tensor core FP8 trên GPU laptop: nhanh gấp đôi trên giấy, không nhanh hơn trong một model
date: 2026-11-20
order: 7
summary: Tensor core FP8 của Ada cho một GEMM trần được bao nhiêu, một layer FP8 W8A8 dựng trên cuBLASLt, chất lượng của nó, và vì sao prefill cả model không nhanh hơn fp16.
---

[Bài về FP8](03-fp8-without-hardware.html) lưu *weight* ở FP8 và vẫn nhân bằng fp16. Dòng GeForce RTX 40 có tensor core nhân trực tiếp các toán hạng FP8, nên bước tự nhiên tiếp theo là **W8A8**: lượng tử hóa cả activation sang FP8 và chạy GEMM trên các core đó. Bài này là nghiên cứu khả thi và kết quả (âm) của nó.

## Phần cứng làm được gì

`torch._scaled_mm` gọi GEMM FP8 của cuBLASLt. Thông lượng GEMM trần trên card này, ổn định, so với GEMM cuBLAS fp16:

| M x N x K | fp16 | FP8 | tăng tốc |
|---|--:|--:|--:|
| 512 x 4096 x 4096 | 37.0 TFLOP/s | 77.4 | 2.09x |
| 2048 x 4096 x 4096 | 38.0 | 72.6 | 1.91x |
| 4096 x 4096 x 4096 | 37.8 | 72.7 | 1.92x |

Đính chính (2026-11-21): phiên bản đầu của bảng này có mốc fp16 là 28 TFLOP/s, đo trước khi GPU laptop đạt xung ổn định, làm tỷ lệ bị thổi phồng (2.64x ở 512 hàng). Đo lại khi GPU đã được làm nóng và hai kernel chạy xen kẽ qua ba vòng: fp16 37 đến 38 TFLOP/s, FP8 72 đến 77, tức **1.9 đến 2.1 lần**. Dòng 2048 x 11008 x 4096 chưa được đo lại nên bị bỏ. Các kết luận bên dưới không đổi.

Vậy các core FP8 nhanh gấp khoảng hai lần: có thật, và nhiều hơn những gì kernel W4A16 đạt được ở cỡ prefill (khoảng 0.85 của Marlin, tức khoảng tốc độ fp16).

## Layer

`FP8W8A8Linear` (`quantize_model(model, "fp8_w8a8")`) lưu weight FP8 E4M3 với scale theo kênh đầu ra và lượng tử hóa activation tức thời với scale theo token. Hai chi tiết mỗi cái tốn của tôi một giờ:

* chế độ scale theo hàng của `_scaled_mm` không có trên sm_89 (nó âm thầm trả số sai trong test của tôi), nên phép nhân chạy với scale tensor đơn vị và hai vector scale được áp sau;
* **các tích FP8 thô tràn fp16**: |x| và |w| tới 448 và K = 4096 cho tổng gần 10^8, nên GEMM phải cho ra f32 và scale được áp lên đó; phiên bản đầu của tôi yêu cầu kết quả f16 và test in ra `inf`.

Không do compiler sinh: backend NVPTX của LLVM 18 không phát được lệnh `mma` FP8, nên layer này là tích hợp cuBLASLt, không phải kernel QuantForge.

## Chất lượng ổn

Perplexity Wikitext-2 trên toàn tập test: Qwen2.5-0.5B fp16 18.79, FP8 W8A8 19.10 (scale theo tensor: 19.04), INT4 round-to-nearest 22.91. Llama-3.2-1B fp16 17.91, FP8 W8A8 18.13. FP8 giữ độ chính xác tương đối qua các độ lớn khác nhau, nên một scale cho cả tensor cũng chạy được; cái giá là khoảng 0.2 đến 0.3 điểm perplexity.

## Tốc độ thì không

Prefill Qwen2.5-0.5B (một lượt forward, tốt nhất trong 10 lần):

| biến thể | 1 x 2048 token | 8 x 512 token | perplexity |
|---|--:|--:|--:|
| fp16 | 108 ms | 214 ms | 18.79 |
| INT4 (QuantForge) | 127 ms | 255 ms | 22.91 |
| FP8 W8A8, scale theo token / theo kênh | 117 ms | 236 ms | 19.10 |
| FP8 W8A8, một scale cho mỗi tensor | 103 ms | 326 ms | 19.04 |

Không có mức tăng. Theo từng layer bức tranh thất thường: với đầu vào 2048 token, layer FP8 mất từ 0.35 đến 1.5 lần thời gian của layer fp16 tùy shape. Vì sao:

1. **GEMM chỉ là một phần của layer.** GEMM FP8 trần của layer 2048 x 896 x 4864 mất 0.24 ms so với 0.43 ms ở fp16, nhưng lượng tử hóa activation (một phép rút gọn và một lượt theo phần tử) và scale lại kết quả f32 (ghi 40 MB rồi đọc lại) tốn thêm 0.25 đến 0.45 ms. Ở các cỡ layer này chi phí phụ lớn bằng phần tiết kiệm được.
2. **Attention, norm và phần còn lại vẫn ở fp16**, và với model 0.5B chúng chiếm tỷ trọng lớn của lượt forward.
3. Biến thể một scale để cuBLASLt áp scale ngay trong GEMM, bỏ được lượt rescale, nhưng tốc độ của nó phụ thuộc vào shape theo cách tôi không dự đoán được (4864 x 896 chậm gấp ba so với biến thể theo token).

## Cần gì để làm được

Một mức tăng tốc thật cần gộp việc lượng tử hóa activation vào nơi sinh ra activation (hoặc vào phần mở đầu của GEMM) và gộp epilogue scale vào lệnh store của GEMM, đúng loại gộp mà compiler này làm cho dequantisation và hàm kích hoạt. Nó cần lệnh `mma` FP8, mảnh còn thiếu trong toolchain này. Cho đến lúc đó layer này hữu ích vì **bộ nhớ** (một nửa số byte weight của fp16) và chất lượng, và làm baseline cho một đường compiler tương lai, không phải vì tốc độ. Với decode batch 1, kernel INT4 vẫn tốt hơn nhiều: weight FP8 gấp đôi số byte của INT4.
