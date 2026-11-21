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

## Cập nhật: tensor core INT8 cho được gì (2026-11-21)

Câu hỏi tự nhiên sau kết quả FP8 là liệu compiler, vốn *có thể* phát lệnh `mma` INT8 (LLVM 18 có lệnh này, chỉ thiếu lệnh FP8), có làm tốt hơn không. Ba phép đo.

**1. Phần cứng.** Một microbenchmark chỉ dùng thanh ghi cho các lệnh `mma.sync` (`benchmarks/mma_peak.cu`, 16 warp mỗi SM, không có lưu lượng bộ nhớ), RTX 4050:

| lệnh | đỉnh | so với fp16 / tích lũy fp32 |
|---|--:|--:|
| vào fp16, tích lũy fp32 (`m16n8k16`) | 44.1 TFLOP/s | 1.00x |
| vào fp16, tích lũy fp16 | 48.1 | 1.09x |
| **vào int8, tích lũy int32 (`m16n8k32`)** | **88.2 TOPS** | **2.00x** |
| vào fp8 e4m3, tích lũy fp32 | 96.2 | 2.18x |

cuBLAS fp16 đạt 37 đến 38 trên 44 (85 phần trăm). INT8 là đúng gấp đôi, và là một lệnh mà compiler này đã có thể nhắm tới.

**2. Thư viện.** `torch._int_mm` ban đầu cho 22 đến 24 TOPS, *chậm hơn* fp16, trông như ngõ cụt. Đó là hiệu ứng bố cục: với B theo cột (`weight.t()` của weight `[out, in]` theo hàng đúng là như vậy) cuBLASLt đạt **80 đến 83 TOPS, gấp 2.1 đến 2.2 lần fp16**; với B theo hàng nó âm thầm đi vào đường chậm. Một lời nhắc rằng con số của thư viện là nói về một đường đi, không phải về phần cứng.

**3. Layer và cả model.** `INT8W8A8Linear` (scale theo kênh cho weight, scale động theo token cho activation, cuBLASLt cho phép nhân) cư xử y như layer FP8: 1.44x fp16 với layer 2048 x 4096 x 4096, 0.74 đến 0.96x với các layer nhỏ của model 0.5B, vì lượng tử hóa activation và epilogue scale tốn khoảng 0.4 ms quanh một GEMM 0.86 ms. Prefill cả model 2048 token: Qwen2.5-0.5B 95.8 ms so với 86.3 ms ở fp16 (0.90x), Llama-3.2-1B 166.6 so với 178.5 ms (1.07x). Chất lượng chấp nhận được và ngang FP8: perplexity wikitext-2 19.20 so với 18.79 (+0.41) và 18.23 so với 17.91 (+0.33).

Vậy mức gấp đôi tồn tại trong lệnh và trong GEMM của thư viện, và bị mất ở các bước quanh nó. Đó là lý lẽ cho một kernel INT8 **do compiler sinh** với epilogue (scale hàng, scale cột, bias, lưu f16) gộp vào lệnh store, và lượng tử hóa activation gộp vào nơi sinh ra activation; khi gộp cả hai, phép tính ở trên gợi ý khoảng 1.0 ms thay vì 1.29 ms cho layer rộng 4096, tức 1.8x fp16, và một biến thể W4A8 (weight int4 nới sang int8, scale nhóm áp lên tổng riêng int32) cũng sẽ giảm một nửa công việc dequantisation của kernel W4A16 hiện tại. Chưa cái nào được xây; đó là cách hứa hẹn nhất để thắng cuBLAS ở chế độ bị giới hạn bởi tính toán, với độ chính xác (smoothing / xoay cho W4A8) là rủi ro chính.

## Cập nhật 2: kernel INT8 của chính compiler (2026-11-21)

Phép tính của lần cập nhật trước nói rằng mức 2x bị mất quanh GEMM, nên điều đầu tiên cần biết là đường Tensor Core của compiler có tiến gần đỉnh của lệnh không. Tôi đẩy một GEMM `i8 x i8 -> i32` thuần (`linalg.matmul_transpose_b`, B dạng `[N, K]` vì `ldmatrix` không chuyển vị được dữ liệu 8-bit) qua pipeline hiện có. Nó cần ba thay đổi nhỏ, một phát hiện về cấu hình và một sửa lỗi:

* matmul đã vector hóa là `contract(extsi a, extsi b, acc)`, và phần hạ xuống nvgpu từ chối `extsi`: `populateFoldArithExtensionPatterns` của LLVM 18 chỉ gộp `extf`, nên pass vector hóa có thêm một mẫu gộp các phép mở rộng số nguyên vào contraction (khoảng 20 dòng);
* lệnh là `m16n8k16` s8 (không nhanh hơn fp16); một tùy chọn pipeline mới `thread-tile-k=32` chọn `m16n8k32`;
* lần chạy đầu chết vì `misaligned address`: padding shared memory 8 phần tử là 16 byte với f16 nhưng 8 byte với int8, nên một hàng của tile không còn căn 16 byte cho `ldmatrix`; `smem-pad=16` sửa nó;
* pass ghi đè (`accumulate=false`) chỉ biết accumulator f32 và âm thầm vẫn cộng dồn vào C với i32; giờ nó xử lý cả hai. Tôi phát hiện ra vì chính test của tôi điền sẵn giá trị vào đầu ra.
* **sweep tìm ra một lỗi thật**: 48 trên 180 tổ hợp block / warp / tile K cho kết quả sai (75, 50 hoặc 100 phần trăm đầu ra), kể cả những cái trông nhanh nhất ở 60 TOPS. Bản đồ đầu ra theo từng fragment cho thấy các tile n8 xen kẽ bị sai; IR nvgpu của một cấu hình tốt và một cấu hình xấu giống hệt nhau về cấu trúc, nhưng ở cấu hình xấu tile B trong shared memory là `memref<64x64xi8>` với block N bằng 128. Pass đưa lên shared memory nhận diện tile toán hạng theo hình dạng, `[M-tile x K]` cho A và `[K x N-tile]` cho B, còn B chuyển vị là `[N-tile x K]`: nó chỉ được nhận ra khi block N tình cờ bằng block M hoặc warp M, nếu không thì pass đưa tile *warp* vào một buffer dùng chung cho mọi warp. Mở rộng phép kiểm tra hình dạng một dòng là sửa được; cùng sweep đó giờ cho 174 đúng tuyệt đối và 6 lỗi biên dịch rõ ràng (vượt ngân sách shared memory), không còn kết quả sai, và một test hồi quy phủ các tổ hợp từng sai.

Kết quả, mọi đầu ra **khớp từng bit** với cuBLASLt (RTX 4050, GPU được làm nóng, chế độ ghi đè, `benchmarks/int8_gemm_probe.py`):

| shape (M x K x N) | INT8 do compiler sinh | cuBLASLt INT8 | cuBLAS fp16 | so với fp16 | so với cuBLASLt |
|---|--:|--:|--:|--:|--:|
| 2048 x 4096 x 4096 | 61.7 TOPS | 69.9 | 38.5 | **1.60x** | 0.88x |
| 4096 x 4096 x 4096 | 64.8 | 77.0 | 38.0 | **1.70x** | 0.84x |
| 512 x 4096 x 4096 | 50.0 | 58.5 | 36.4 | 1.37x | 0.85x |
| 2048 x 4096 x 11008 (swizzle 16) | 59.6 | 77.5 | 38.3 | **1.56x** | 0.77x |
| 2048 x 4864 x 896 | 54.5 | 52.5 | 35.3 | 1.54x | **1.04x** |
| 2048 x 896 x 4864 | 38.6 | 56.1 | 35.1 | 1.10x | 0.69x |
| 2048 x 1536 x 8960 | 51.2 | 64.1 | 36.6 | 1.40x | 0.80x |

Biên dịch từ cùng bộ máy với kernel W4A16, với schedule chọn từ sweep 129 tổ hợp đều đúng tuyệt đối, nó đạt 59 đến 65 TOPS ở các shape lớn (ba phần tư đỉnh 88 TOPS của lệnh, 77 đến 88 phần trăm cuBLASLt, và vượt nó ở một shape) và **nhanh hơn cuBLAS fp16 từ 1.1 đến 1.7 lần ở mọi shape đã thử**. Weight lớn hơn L2 cần `block-swizzle=16` (2048 x 4096 x 11008: từ 21.7 lên 59.6 TOPS), một lời nhắc rằng thứ tự block quan trọng ngang tile.

Cái nó không phải là: một GEMM int32 thô, không có scale, không có epilogue và không có lượng tử hóa activation, nên chưa phải một layer và chưa phải mức tăng tốc nào người dùng thấy được. Hai lưu ý: các shape K nhỏ bị kìm bởi đầu ra int32 (4 byte mỗi phần tử), thứ mà một epilogue lưu f16 gộp sẽ giảm một nửa; và chỉ có một GPU, sm_89. Các bước tiếp theo là những bước phép tính đã nêu: gộp scale hàng / cột và bias vào lệnh store, và lượng tử hóa activation trong kernel sinh ra nó; rồi W4A8.

## Cập nhật 3: gộp dequantisation, và cả model (2026-11-21)

Một GEMM int32 thô chưa phải một layer: đầu ra phải thành f16 với scale hàng (của activation), scale cột (của weight) và bias. Là một kernel riêng thì nó tốn vòng đi về của int32: với layer 2048 x 4096 x 4096 là 33 MB ghi rồi đọc lại, tốn thời gian bằng khoảng một phần tư GEMM. Compiler đã gộp một activation (relu, silu, swiglu) vào lệnh store cuối cùng của các accumulator; một loại epilogue mới, `int8-dequant`, làm điều tương tự cho `y = f16(c * sx[m] * sw[n] + bias[n])`. Chương trình vẫn bình thường về mặt ngữ nghĩa, một `matmul` rồi một `linalg.generic` theo phần tử; pass viết lại mọi lệnh store accumulator `vector<2xi32>` thành cặp đã chuyển đổi, ghi bằng một lệnh store 4 byte, và xóa kernel theo phần tử cùng buffer int32. Khớp từng bit với tham chiếu f32, có và không có bias, và là một kernel trong test FileCheck.

**Một bài học về đo lường.** Benchmark đầu tiên nói kernel hợp nhất *chậm hơn* GEMM thô (1.93 ms so với 1.13 ms cho 2048 x 4096 x 4096). `ncu` nói không phải vậy (1.21 so với 1.25 ms, và DRAM ghi giảm một nửa). Thời gian thiết bị của profiler (1.09 ms) so với thời gian wall mỗi lần gọi (2.05 ms) chỉ ra mili giây còn thiếu nằm ở phía host: phần cấp phát của buffer int32 vẫn còn sau phép viết lại vì các `subview` chết còn trỏ vào nó, nên mỗi lần gọi đều `cudaMalloc` rồi `cudaFree` 33 MB. Xóa các subview chết là hết. Chỉ việc so thời gian thiết bị với thời gian wall mới cho thấy kernel chưa bao giờ là vấn đề.

GEMM cộng dequantisation, một lần launch, ra f16 (RTX 4050, GPU được làm nóng; cột fp16 là GEMM thường không có scale):

| shape (M x K x N) | cuBLAS fp16 | chỉ GEMM int32 | GEMM + dequant hợp nhất | so với fp16 |
|---|--:|--:|--:|--:|
| 2048 x 4096 x 4096 | 1.76 ms | 1.11 | 1.03 | **1.71x** |
| 4096 x 4096 x 4096 | 3.51 | 2.12 | 2.12 | 1.66x |
| 2048 x 4096 x 11008 | 5.06 | 3.09 | 2.96 | 1.71x |
| 2048 x 4864 x 896 | 0.48 | 0.33 | 0.30 | 1.61x |
| 2048 x 1536 x 8960 | 1.50 | 1.09 | 0.93 | 1.61x |
| 2048 x 896 x 4864 | 0.48 | 0.46 | 0.37 | 1.31x |
| 512 x 4096 x 4096 | 0.46 | 0.35 | 0.32 | 1.44x |

Kernel hợp nhất còn nhanh hơn GEMM int32 thô, vì nó ghi một nửa số byte, và các shape K nhỏ vốn yếu nhất khi là GEMM thô giờ được lợi nhiều nhất từ epilogue.

**Cả model.** `quantize_model(model, "int8_w8a8")` giờ dùng kernel này: activation được lượng tử hóa theo token (một lượt elementwise đã biên dịch), weight theo kênh đầu ra, GEMM và dequantisation là kernel đã biên dịch; các layer có N hoặc K không khớp tile của nó quay về cuBLASLt. Prefill 2048 token, cùng một lần chạy, fp16 so với INT8 đã biên dịch (và so với đường cuBLASLt trước đó):

| model | fp16 | INT8, đường cuBLASLt | INT8, kernel hợp nhất đã biên dịch | perplexity (fp16 -> INT8) |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B | 93.1 ms | 103.7 (0.90x) | 86.7 (**1.07x**) | 18.79 -> 19.20 |
| Llama-3.2-1B | 186.9 | 177.4 (1.05x) | 159.4 (**1.17x**) | 17.91 -> 18.23 |
| Qwen2.5-1.5B | 238.1 | 226.6 (1.05x) | 204.4 (**1.16x**) | 12.66 -> 13.06 |

Vậy cùng phép tính mà cuBLASLt không biến được thành mức tăng cả model lại là mức tăng tốc prefill 7 đến 17 phần trăm khi epilogue được gộp trong compiler. Nó thấp xa hơn mức 1.5 đến 1.7x của GEMM, và phép tính nói vì sao: attention, các norm, head (giữ ở fp16: với model 0.5B nó khoảng một phần sáu thời gian fp16), lượt lượng tử hóa activation và overhead host của một layer Python đều không được đụng tới. Các bước tiếp theo hiển nhiên là gộp lượng tử hóa activation vào nơi sinh ra activation, lượng tử hóa luôn head, và một biến thể W4A8 giảm một nửa số byte weight; cái giá về chất lượng (0.3 đến 0.4 điểm perplexity cho INT8 thuần theo token / theo kênh) là thứ cần theo dõi.
