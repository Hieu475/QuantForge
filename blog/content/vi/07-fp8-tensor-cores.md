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
| 2048 x 4096 x 4096 | 1.70 ms | 1.05 | 1.10 | **1.54x** |
| 4096 x 4096 x 4096 | 3.66 | 2.18 | 2.02 | **1.81x** |
| 2048 x 4096 x 11008 | 4.92 | 3.12 | 3.00 | 1.64x |
| 2048 x 4864 x 896 | 0.51 | 0.34 | 0.31 | 1.61x |
| 2048 x 1536 x 8960 | 1.55 | 1.07 | 0.96 | 1.62x |
| 2048 x 896 x 4864 | 0.51 | 0.45 | 0.40 | 1.29x |
| 512 x 4096 x 4096 | 0.47 | 0.36 | 0.33 | 1.43x |

Kernel hợp nhất còn nhanh hơn GEMM int32 thô, vì nó ghi một nửa số byte, và các shape K nhỏ vốn yếu nhất khi là GEMM thô giờ được lợi nhiều nhất từ epilogue.

**Cả model.** `quantize_model(model, "int8_w8a8")` giờ dùng kernel này: activation được lượng tử hóa theo token (một lượt elementwise đã biên dịch), weight theo kênh đầu ra, GEMM và dequantisation là kernel đã biên dịch; các layer có N hoặc K không khớp tile của nó quay về cuBLASLt. Prefill 2048 token, cùng một lần chạy, fp16 so với INT8 đã biên dịch (và so với đường cuBLASLt trước đó):

| model | fp16 | INT8, đường cuBLASLt | INT8, kernel hợp nhất đã biên dịch | perplexity (fp16 -> INT8) |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B | 87.0 ms | 95.5 (0.91x) | 79.3 (**1.10x**) | 18.79 -> 19.20 |
| Llama-3.2-1B | 178.5 | 166.5 (1.07x) | 151.7 (**1.18x**) | 17.91 -> 18.23 |
| Qwen2.5-1.5B | 236.1 | 218.8 (1.08x) | 196.7 (**1.20x**) | 12.66 -> 13.06 |

Vậy cùng phép tính mà cuBLASLt không biến được thành mức tăng cả model lại là mức tăng tốc prefill 10 đến 20 phần trăm khi epilogue được gộp trong compiler. Nó thấp xa hơn mức 1.5 đến 1.7x của GEMM, và phép tính nói vì sao: attention, các norm, head (giữ ở fp16: với model 0.5B nó khoảng một phần sáu thời gian fp16), lượt lượng tử hóa activation và overhead host của một layer Python đều không được đụng tới. Các bước tiếp theo hiển nhiên là gộp lượng tử hóa activation vào nơi sinh ra activation, lượng tử hóa luôn head, và một biến thể W4A8 giảm một nửa số byte weight; cái giá về chất lượng (0.3 đến 0.4 điểm perplexity cho INT8 thuần theo token / theo kênh) là thứ cần theo dõi.

## Cập nhật 4: dùng chung lượng tử hóa, một bản profile, và hai cái bẫy đo lường (2026-11-21)

**Dùng chung lượng tử hóa activation.** q, k và v đọc cùng một activation, gate và up cũng vậy; mỗi layer đều lượng tử hóa lại chúng. `quantize_model(..., "int8_w8a8")` giờ gộp chúng (một layer dùng chung đầu vào: một lượt lượng tử hóa, một GEMM rộng hơn, mỗi layer gốc lấy phần của mình). Nó chỉ tăng 1 đến 3 phần trăm prefill (0.5B: 80.6 lên 81.1 ms, trong sai số; Llama-3.2-1B 155.2 xuống 151.8; Qwen2.5-1.5B 203.5 xuống 201.3), nên việc lượng tử hóa lặp lại không phải là chỗ mất thời gian. Một lỗi của tôi trên đường đi: layer dùng chung giữ đầu ra của lần gọi cuối, khoảng 80 MB mỗi layer với gate / up ở 4096 hàng, làm model hết bộ nhớ; chúng được giải phóng ngay khi mọi phần đã được lấy, và một test so sánh đầu ra của model gộp và không gộp.

**Thời gian đi đâu** (Qwen2.5-0.5B, một lượt prefill 2048 token, thời gian thiết bị của kernel):

| | fp16 | INT8 |
|---|--:|--:|
| GEMM của các layer | 41.4 ms | **28.3** (1.46x) |
| `lm_head` (giữ ở fp16) | 14.3 | 14.3 |
| norm, rotary, silu x up, residual, lượng tử hóa... | 24.1 | 31.7 (+7.6 cho lượng tử hóa) |
| attention | 6.0 | 5.7 |

GEMM thật sự nhanh hơn 1.46x; điều ngăn cả model theo kịp là 24 ms, 28 phần trăm prefill fp16, là các phép theo phần tử của một model Hugging Face chạy eager, thứ không kernel GEMM nào đụng tới (engine suy luận gộp chúng), cộng một head mà prefill thật còn không tính cho mọi vị trí. Bản thân lượt lượng tử hóa tốn 7.6 ms; `torch.compile` đã làm nó thành một kernel ở mức 1.1 lần lý tưởng của một lượt bộ nhớ, nên một kernel viết tay không giúp được gì; chỉ việc gộp nó vào nơi sinh ra activation mới giúp (một kernel `silu(gate) * up` rồi lượng tử hóa, cho đầu vào của `down_proj`, là mảnh đơn lẻ lớn nhất: khoảng 5 ms ở đây).

**Hai cái bẫy đo lường.** (1) Một sweep `block-swizzle` đo từng biến thể ở một thời điểm khác nhau nói rằng swizzle giúp mọi shape layer từ 3 đến 28 phần trăm. Các vòng xen kẽ của mọi biến thể trong một phiên nói điều khác: nó nhanh hơn 2.3x với weight lớn hơn L2 (2048 x 2048 x 16384: 5.8 xuống 2.5 ms) và chậm hơn 12 đến 14 phần trăm với weight vừa trong L2 (2048 x 4096 x 4096), nên quy tắc giờ dựa trên kích thước. Xung nhịp của GPU laptop trôi 10 phần trăm giữa các lần đo, nhiều hơn phần lớn các khác biệt tôi đang cố đo. Phép gộp đầu vào dùng chung trước đó đã làm lộ ra điều này theo chiều ngược lại: với ngưỡng cũ, weight gate / up được nới rộng của Llama-3.2-1B (33.6 MB, ngay dưới ngưỡng) chạy không swizzle với thời gian gấp 2.9 lần và làm model chậm đi 32 phần trăm, cho đến khi tôi đo riêng shape đó. (2) Các bảng trong bài này đến từ các phiên khác nhau; hãy so sánh các cột trong cùng một hàng, đừng bao giờ so giữa các bảng.

Chưa làm, và vì sao: lượng tử hóa head (prefill thật chỉ cần vị trí cuối), W4A8 (giảm một nửa số byte weight nhưng cần phép đổi tỷ lệ int32 sang float theo từng nhóm ngay trong vòng K, một cách tích lũy mới trong compiler), và lượng tử hóa `silu * up` gộp, vốn là một kernel cho một họ model.

## Cập nhật 5: W4A8, và lý do tôi đưa ra để không làm nó là sai (2026-11-21)

Cập nhật 4 nói W4A8 "cần phép đổi tỷ lệ int32 sang float theo từng nhóm ngay trong vòng K, một cách tích lũy mới trong compiler". Không cần, nếu cách lượng tử hóa được thiết kế cho nó. *Lượng tử hóa lũy tiến* của QServe làm được: weight trước hết được lượng tử hóa sang INT8 với một scale float cho mỗi kênh đầu ra, rồi weight INT8 đó được lượng tử hóa sang INT4 theo nhóm 128 với scale **nguyên** (1 đến 16) và zero point nguyên. Weight mà tensor core cần là `(nibble - zero) * group_scale`, lại là một int8, nên nó dựng được trong thanh ghi lúc nạp tile, phép tích lũy vẫn là một int32 trên toàn bộ K, và scale float duy nhất còn lại là scale theo kênh trong epilogue đã có. Không phải đổi gì ở compiler.

**Kernel là một producer, không phải một pass.** Chương trình là một `linalg.generic` tính weight int8 từ các nibble đã đóng gói (`n, k` đọc byte `k / 2`, chọn nibble theo `k mod 2`, trừ zero point của nhóm, nhân với scale của nhóm; toàn bộ trong `i8`), rồi đến `matmul_transpose_b` và phép lưu `int8-dequant` của Cập nhật 3. Pass hợp nhất producer viết cho việc dequantise INT4 hóa ra không riêng cho nó: mọi phép store có backward slice thuần đều được tính lại tại các lần load của GEMM. Nó biên dịch được ngay lần đầu, khớp từng bit với tham chiếu số nguyên, trong **một kernel** (`W4A8GemmDq`). Weight là lưu lượng weight duy nhất: 4 bit mỗi weight.

**Lượng tử hóa, và một cái bẫy chất lượng.** Phiên bản đầu của tôi dùng dải 4-bit đối xứng theo nhóm, và model tệ hơn INT4 thường rất nhiều (perplexity Qwen2.5-0.5B 28.9 so với 22.9). Tôi đã đổ cho scale nhóm nguyên; chọn nó theo từng nhóm bằng bình phương sai số nhỏ nhất thay vì làm tròn lên chỉ đưa Qwen xuống 28.0 và Llama-3.2-1B không đổi. Thí nghiệm tách riêng tìm ra nguyên nhân: thay weight bằng giá trị dequantise và giữ activation ở fp16, INT4 đối xứng với scale *float* đã cho 26.3, bất đối xứng cho 22.9. INT4 của thư viện luôn có zero point theo nhóm; tôi đã bỏ nó để tiết kiệm một tensor. Với zero point nguyên theo nhóm (phép tìm chỉ giữ các cặp (scale, zero) mà `(q - z) * s` vẫn nằm trong int8, vì số học 8-bit của kernel bị quấn vòng chứ không bão hòa) riêng weight cho 23.1 và cả lớp W4A8 cho 23.7 (Qwen2.5-0.5B), 22.06 (Llama-3.2-1B, so với 22.28 của INT4 W4A16 với head INT8). Vậy chất lượng là của INT4, không tốt hơn.

**Tốc độ** (kernel, các vòng xen kẽ trong một phiên, L2 được xả, `benchmarks/w4a8_gemm.py`):

| shape (M x K x N) | cuBLAS fp16 | QF W4A16 | QF INT8 W8A8 | **QF W4A8** | W4A8 so với fp16 / W4A16 / INT8 |
|---|--:|--:|--:|--:|--:|
| 2048 x 4096 x 4096 | 1424 us | 1767 | 966 | **1286** (53 TOPS) | 1.11x / 1.37x / 0.75x |
| 2048 x 4096 x 11008 | 4359 | 5455 | 2630 | **3709** | 1.18x / 1.47x / 0.71x |
| 2048 x 11008 x 4096 | 4649 | 5031 | 2618 | **3682** | 1.26x / 1.37x / 0.71x |
| 2048 x 896 x 4864 | 449 | 711 | 304 | **381** | 1.18x / 1.87x / 0.80x |
| 2048 x 2048 x 8192 | 1639 | 2062 | 1022 | **1304** | 1.26x / 1.58x / 0.78x |

Nó vượt cuBLAS fp16 và kernel W4A16 ở prefill, và chậm hơn kernel INT8 từ 20 đến 30 phần trăm: việc giải nén là công việc ALU trong vòng nạp tile, lặp lại cho mỗi khối hàng, và producer tổng quát là mã vô hướng (một phép giải nén vector hóa như của đường W4A16 là việc nên thử tiếp theo).

Cả model, prefill 2048 token, head ở fp16, bốn biến thể của mỗi model trong một phiên (`benchmarks/hf_fp8_w8a8.py`):

| model | fp16 | INT4 W4A16 (+ head INT8) | INT8 W8A8 | **W4A8** |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B, ms (so với fp16) | 87.7 | 105.4 (0.83x) | 79.5 (1.10x) | **87.0 (1.01x)** |
| perplexity / bộ nhớ GPU | 18.79 / 1.01 GB | 22.91 / 0.67 | 19.20 / 0.64 | **23.72 / 0.47** |
| Llama-3.2-1B, ms | 184.2 | 210.1 (0.88x) | 152.2 (1.21x) | **172.7 (1.07x)** |
| perplexity / bộ nhớ GPU | 17.91 / 2.48 GB | 22.28 / 1.40 | 18.23 / 1.51 | **22.06 / 1.05** |
| Qwen2.5-1.5B, ms | 243.1 | hết bộ nhớ trong benchmark này | 199.6 (1.22x) | **233.5 (1.04x)** |
| perplexity / bộ nhớ GPU | 12.66 / 3.10 GB | | 13.06 / 1.79 | **14.70 / 1.19** |

(Biến thể INT4 W4A16 của 1.5B hết bộ nhớ trong benchmark Hugging Face nên không có số cho nó; các hàng 1.5B của những biến thể khác được chạy mỗi biến thể một tiến trình.) Tóm tắt trung thực: **W4A8 là bản thay thế cho lớp INT4 W4A16 ở prefill, giữ nguyên chất lượng** (nhanh hơn 1.2x với cùng số bit và gần cùng perplexity), và là định dạng nhỏ nhất ở đây, nhưng ở mức cả model nó chỉ chạm tốc độ fp16 vì 28 phần trăm prefill đó không phải GEMM; khi có thể chi thêm 1.5 lần bộ nhớ, INT8 W8A8 vừa nhanh hơn vừa tốt hơn W4A8 1.6 điểm perplexity ở model 1.5B (13.06 so với 14.70). Decode không đổi (bị giới hạn bởi băng thông: kernel W4A16 đã đọc cùng số byte). Weight đến từ việc lượng tử hóa model fp16 bằng làm tròn gần nhất; bù sai số kiểu GPTQ trên định dạng này chưa làm.

Test: `test_w4a8_quantisation_never_overflows_int8_and_the_linear_matches_its_reference` (weight nguyên nằm trong int8 với đầu vào nhiều outlier, lớp đã biên dịch bằng tham chiếu số nguyên tới mức làm tròn f16) và `test_w4a8_shared_input_fusion_gives_the_same_model_output`.

**Cập nhật 5b: GPTQ cho W4A8.** Làm tròn gần nhất để W4A8 thua INT4 W4A16 từ 0.3 đến 0.8 điểm perplexity ở hai trong ba model, nên tôi viết GPTQ cho định dạng này (`gptq_quantize_weight_w4a8`, `quantize_model_gptq(..., fmt="w4a8")`): weight được chia cho scale theo kênh để nằm trong bước int8, và với mỗi nhóm 128 cột, cặp (scale, zero point) nguyên được chọn từ các weight *đã được cập nhật bởi phần bù sai số của các cột trước*; phần còn lại là bù sai số từng cột như thường lệ với thừa số Cholesky của nghịch đảo Hessian. Perplexity wikitext-2 test, head fp16 ở mọi cột, 128 cửa sổ hiệu chỉnh 512 token:

| model | fp16 | INT4 W4A16 RTN | INT4 W4A16 GPTQ | W4A8 RTN | **W4A8 GPTQ** |
|---|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B | 18.79 | 22.91 | 20.45 | 23.72 | **21.06** |
| Llama-3.2-1B | 17.91 | 22.28 | 20.21 | 22.06 | **19.91** |
| Qwen2.5-1.5B | 12.66 | 14.29 | 13.37 | 14.70 | **13.68** |

GPTQ lấy lại 2.0 đến 2.7 điểm ở các model nhỏ và 1.0 ở model 1.5B, việc lượng tử hóa mất 49 đến 135 giây. W4A8 với GPTQ thua INT4 GPTQ 0.3 đến 0.6 ở các model Qwen và *hơn* 0.3 ở Llama-3.2-1B, nơi scale nguyên cùng phép tìm kiếm tình cờ khớp hơn; tôi sẽ không suy diễn thêm từ một model. (Với head fp16, INT4 W4A16 của 1.5B đo được 14.29; biến thể ở bảng trước dùng head INT8 và hết bộ nhớ trong benchmark đó.) Các lớp có kích thước không chia hết cho 128 giữ nguyên dense; `act_order` và tùy chọn thu hẹp dải không có cho định dạng này. Test: `test_gptq_w4a8_beats_the_w4a8_rtn_on_the_layer_output_error` (sai số đầu ra dưới 0.8 của RTN, tôn trọng dải int8) và một lần chạy end-to-end trên Llama nhỏ. `benchmarks/w4a8_gptq.py` tái tạo bảng này.
