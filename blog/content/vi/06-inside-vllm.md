---
title: QuantForge bên trong vLLM: một phép so sánh đầu-cuối công bằng, và 20 đến 30 phần trăm còn thiếu đi đâu
date: 2026-11-20
order: 6
summary: Một plugin lượng tử hóa của vLLM chạy các kernel đã sinh trong một engine phục vụ thật, throughput so với Marlin ở 1 đến 64 request đồng thời, và cách một chẩn đoán theo từng layer dẫn tới K-folding và các schedule mặc định theo shape (từ 70-80 phần trăm Marlin lên 73-88 phần trăm).
---

Các benchmark kernel trong [bài tổng quan](01-overview.html) so sánh GEMM tách rời, và các số đầu-cuối so sánh một model Hugging Face với một model khác. Không cái nào trả lời câu hỏi thực tế: **nếu tôi thay Marlin bằng các kernel đã sinh trong cùng một engine, với cùng một checkpoint, tôi mất hay được bao nhiêu throughput?** Một plugin vLLM cho phép làm thí nghiệm đó.

## Plugin

vLLM cho phép một package đăng ký một phương pháp lượng tử hóa. Plugin (`quantforge.vllm_plugin`) nhỏ vì nó tái dùng bộ máy GPTQ của vLLM:

* `QuantForgeConfig` kế thừa config GPTQ của vLLM và đáp ứng `quantization="quantforge"`. Nó nhận cùng các checkpoint (đối xứng hoặc bất đối xứng, group 32/64/128, `desc_act`);
* `QuantForgeLinearMethod` để vLLM tạo và nạp các tensor GPTQ đúng như với Marlin (tensor parallelism, các layer gộp q/k/v và gate/up chạy không đổi), rồi **chuyển đổi một lần sau khi nạp** sang layout QuantForge và gọi `torch.ops.quantforge.w4a16_gemm` trong `apply`;
* op đã chạy được dưới `torch.compile` và CUDA graph, thứ vLLM dùng cho decode; kernel của mọi row bucket được biên dịch trong các lượt warm-up của vLLM, trước khi capture graph;
* nó tự đăng ký qua entry point `vllm.general_plugins` (`pip install -e ./python`), vì engine của vLLM chạy ở một process riêng và pickle config sang đó: các lớp config phải nằm ở cấp module (phiên bản đầu của tôi định nghĩa chúng trong một hàm và lỗi pickle).

```python
from vllm import LLM
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", quantization="quantforge", dtype="float16")
```

## Cùng checkpoint, cùng engine: kết quả

Throughput là số token sinh ra mỗi giây cộng trên các request đồng thời (RTX 4050, tốt nhất trong 3 lần, hai lần chạy độc lập khớp nhau trong 1 đến 2 phần trăm):

| model | request đồng thời | Marlin (vLLM) | QuantForge | QuantForge / Marlin |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B GPTQ-Int4 | 1 | 298 | 231 | 0.78 |
| | 4 | 1153 | 893 | 0.77 |
| | 16 | 4188 | 3403 | 0.81 |
| | 64 | 10418 | 7747 | 0.74 |
| Qwen2.5-1.5B GPTQ-Int4 | 1 | 138 | 99 | 0.72 |
| | 16 | 2012 | 1542 | 0.77 |
| | 64 | 4929 | 3528 | 0.72 |

Kết quả đầu ra khớp nhau: trên năm prompt, bốn prompt sinh cùng 64 token greedy với cả hai kernel và prompt thứ năm khớp 34 token trước khi một thế gần hòa đổi chỗ (làm tròn fp16 khác nhau); chênh lệch tuyệt đối trung bình của log-xác suất các token sinh ra là 0.002 (tối đa 0.017). Vậy **plugin chạy đúng, và bên trong một engine phục vụ thật bản đầu của plugin đạt 70 đến 80 phần trăm throughput của Marlin** (bên dưới: những gì thu hẹp một phần khoảng cách), không phải 100 phần trăm mà bảng kernel batch 1 cho layer 4096 x 4096 có thể gợi ý.

## 20 đến 30 phần trăm còn thiếu đi đâu

Phỏng đoán đầu của tôi là chi phí phụ: op phát ba kernel mỗi GEMM (copy activation vào buffer đệm 16 hàng, GEMM, đổi kết quả f32 sang f16), Marlin một. Profile một lần gọi M = 1 xác nhận các kernel phụ (khoảng 2 micro giây thời gian thiết bị) nhưng cũng cho thấy chúng không thể là nguyên nhân chính, vì khoảng cách không thu hẹp ở model lớn hơn. So sánh kernel theo từng layer thì có (M = 1, micro giây, cùng phương pháp như các bảng kernel):

| layer của Qwen2.5-1.5B / 0.5B | K x N | Marlin | QuantForge | tỷ lệ |
|---|---|--:|--:|--:|
| qkv | 1536 x 2048 | 11.7 | 12.4 | 0.94 |
| gate / up (gộp) | 1536 x 17920 | 78.5 | 84.3 | 0.93 |
| gate / up (gộp, 0.5B) | 896 x 9728 | 27.6 | 28.4 | 0.97 |
| o_proj | 1536 x 1536 | 9.7 | 13.4 | **0.73** |
| **down_proj** | 8960 x 1536 | 40.8 | 66.2 | **0.62** |
| **down_proj (0.5B)** | 4864 x 896 | 15.2 | 27.7 | **0.55** |

Các layer rộng ổn. Các layer **K dài và N hẹp**, mà trong mọi transformer là `down_proj` của MLP, chậm hơn Marlin 40 đến 45 phần trăm. Với 128 hoặc 64 cột đầu ra mỗi block, N = 1536 cho 12 đến 24 block trên 20 SM: không đủ warp đang bay để stream weight ở tốc độ tối đa. Marlin chia chiều K giữa các block; kernel đã sinh thì không.

Điều này lật ngược một kết luận tôi đã viết trước đó ("split-K không phải đòn bẩy"): nó được đo ở 1 x 11008 x 4096, nơi N = 4096 đã cho 64 block. Các shape quan trọng cho decode trong một engine phục vụ thì khác, và phép so sánh engine tìm ra chúng ở nơi benchmark tách rời không tìm ra. Cách chữa chuẩn là split-K trong kernel, mà trong compiler này nghĩa là đụng vào sáu pass (tiling, mapping, shared-memory promotion, prefetch, hoist accumulator, store). Trước đó tôi thử những gì làm được mà không đụng compiler.

## Chữa mà không đụng compiler: K-folding

Ba thứ không có tác dụng: quét các tùy chọn tile và prefetch (không cấu hình nào dưới 67 micro giây cho layer 8960 x 1536, cao hơn 22 phần trăm so với 41 của Marlin), split-K nhiều stream (nhiều lần phát đồng thời vào các accumulator riêng, cộng sau: 72 xuống 56 micro giây cho layer đó, không có lợi cho 4864 x 896), và pipeline sâu hơn.

Cái hiệu quả là một cách sắp lại layout. Cắt K thành S lát và xếp cạnh nhau thành *cột* của một GEMM rộng hơn, với S lát của activation thành S *hàng*:

```
W' = [W_0 | W_1 | ... | W_{S-1}]   shape [K/S, S*N]        A' = x reshape thành [M*S, K/S]  (miễn phí)
C' = A' @ W'   ->   y[m] = sum_s C'[m*S + s, s*N:(s+1)*N]    (S khối chéo)
```

Kernel tính mọi cặp (lát, lát), nhiều hơn cần thiết S lần phép tính, nhưng một tile tensor-core vốn đã có 16 hàng, nên chừng nào `M * S <= 16` thì phần tính thừa là miễn phí: weight vẫn được đọc đúng một lần, bởi số block nhiều gấp S lần và với vòng K ngắn đi S lần. Hệ số được chọn theo từng shape layer bằng cách đo op thật một lần (có cache): S = 2 cho layer 4864 x 896, 7 cho 8960 x 1536. Layer có N >= 4096 đã đủ block và được để nguyên. Cái giá là một bản weight thứ hai cho các layer được fold (layout thường phục vụ các lần gọi nhiều hàng hơn), chỉ đáng kể với model nhỏ vì chỉ N <= 2048 mới được fold.

Mức kernel, một hàng, cả op trong một CUDA graph (micro giây): 4864 x 896 từ 37.7 xuống 23.3, 8960 x 1536 từ 68.5 xuống 43.0, 1536 x 1536 từ 14.4 xuống 12.6. Trong vLLM, cùng checkpoint, các lần chạy xen kẽ khớp nhau trong 1 phần trăm:

| model | request | Marlin (tok/s) | QuantForge (tok/s) | tỷ lệ: bản đầu | + K-fold | + schedule theo shape | + khối hẹp (hiện tại) |
|---|--:|--:|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B | 1 | 289 | 277 | 0.78 | 0.85 | 0.87 | 0.96 |
|  | 4 | 1112 | 1050 | 0.77 | 0.86 | 0.88 | 0.94 |
|  | 16 | 4066 | 3796 | 0.81 | 0.84 | 0.86 | 0.93 |
|  | 32 | 7289 | 6533 | n/a | n/a | n/a | 0.90 |
|  | 64 | 11076 | 9590 | 0.74 | 0.70 | 0.80 | 0.87 |
| Qwen2.5-1.5B | 1 | 141 | 122 | 0.72 | 0.85 | 0.84 | 0.87 |
|  | 4 | 546 | 446 | n/a | 0.74 | 0.80 | 0.82 |
|  | 16 | 2095 | 1702 | 0.77 | 0.80 | 0.82 | 0.81 |
|  | 32 | 3681 | 2896 | n/a | n/a | n/a | 0.79 |
|  | 64 | 5796 | 4300 | 0.72 | 0.68 | 0.73 | 0.74 |

Fold tăng 9 phần trăm (0.5B) và 15 phần trăm (1.5B) với một đến bốn request lúc mới đưa vào (khoảng 4 phần trăm bây giờ khi các schedule thường đã tốt hơn) và không đổi từ 16 request trở lên, nơi `M * S` vượt 16 và kernel thường chạy. Hai cột cuối thêm, trước hết, các schedule mặc định theo độ rộng (các tùy chọn tile của kernel 17 đến 64 hàng và của kernel decode đầu ra hẹp được chỉnh trên 4096 x 4096 và chậm hơn 1.5 đến 3 lần với các shape nhỏ của những model này; autotune chọn khối 32 / 64 cột và look-ahead ở đó, và tôi biến các kết quả thắng thành quy tắc, `benchmarks/sweep_skinny.py`), rồi một lượt sweep thứ hai thấy khối 32 và 64 cột nhanh hơn 11 đến 52 % so với khối 128 cột khi N <= 2048 ở 32 và 64 hàng (với đầu ra rộng không ứng viên nào trong 74 cái hơn mặc định quá 4 %, `benchmarks/sweep_skinny_blocks.py`) và một kernel epilogue gộp cho các layer được fold (3 kernel mỗi lần gọi thay vì 5 đến 7: nhanh hơn 20 đến 35 % mỗi layer, nhưng không thấy trong thời gian cả model, khoảng 1 đến 2 %, vì GPU vốn bận 97 % thời gian). Tổng cộng: model 0.5B đạt 0.87 đến 0.96 Marlin ở 1 đến 64 request, model 1.5B đạt 0.74 đến 0.87; tốc độ tuyệt đối tăng 4 đến 7 % so với cột trước. Tỷ lệ lấy từ một lần chạy Marlin và QuantForge liên tiếp; bản thân Marlin dao động khoảng 4 % giữa các lần chạy theo xung nhịp của laptop.

### Một cái bẫy khi đo

A/B đầu tiên của tôi không cho thấy khác biệt nào (241 so với 240 token mỗi giây, bốn lần chạy xen kẽ mỗi bên), và một profile của cùng lượt decode với cache biên dịch tắt cho 4.16 so với 3.81 mili giây mỗi token. Lý do: vLLM cache các graph đã biên dịch trên đĩa dưới một hash của cấu hình mà không biết gì về plugin, nên một graph biên dịch trước đó, với đường thường, bị dùng lại âm thầm, và đường fold không bao giờ chạy. Cùng cơ chế đó có thể đã cho kết quả sai (một graph cache có hệ số fold đóng cứng thành hằng số, dùng với weight fold khác). Plugin giờ cho mọi layer gọi cùng một op và đọc hệ số từ shape tensor lúc chạy, nên một graph cũ chỉ chậm, không bao giờ sai; sau khi nâng cấp plugin hãy xóa `~/.cache/vllm/torch_compile_cache` một lần.

## Split-K trong compiler: chạy đúng, nhưng không đáng

So sánh `ncu` chỉ ra vấn đề chia tải (Marlin: 20 block cố định), và câu trả lời sách giáo khoa là split-K. Tôi cài nó thành một tùy chọn của pipeline (`--quantforge-gpu-pipeline='... split-k=S'`, mặc định tắt), và việc cài đặt hóa ra cần ít thay đổi hơn tôi lo, vì MLIR đã có `linalg::tileReductionUsingForall`:

1. pass tiling cắt phép rút gọn K thành S đoạn bằng hàm đó: một `scf.forall (z)` có thân là một GEMM K nhỏ hơn bình thường, ghi vào một tensor kết quả từng phần, và một `linalg.generic` cộng S kết quả từng phần;
2. tiling khối hiện có chạy bên trong, tạo ra một forall lồng trong một forall; một phép biến đổi nhỏ gộp hai cái thành một `scf.forall (m, n, z)` để phần hạ xuống launch hiện có nhận được lưới 3 chiều;
3. phần còn lại của pipeline cần ba sửa, mỗi cái tìm ra bằng cách đọc IR lỗi: việc điền 0 buffer từng phần bị vector hóa thành một `vector<16x256x4xf32>` phía host (bỏ ở chế độ ghi đè, nơi mỗi block ghi đủ tile của nó); buffer từng phần được bố trí `[M, N, S]`, làm tile của một block không liên tục và khiến phần hạ xuống Tensor Core không nhận ra nó (giờ là `[S, M, N]`); và phép reduce phải được loại khỏi vector hóa để trở thành một kernel nhỏ riêng.

Nó **đúng**: kết quả khớp kernel không chia và tham chiếu dense (sai số tương đối 2e-4 ở mọi cấu hình, ba shape test trong bộ test, 38 trên 38 test FileCheck không đổi). Chỉ số scale và zero-point vẫn đúng khi vòng K bắt đầu ở `z * Kc` vì dequantisation được gộp qua offset của `extract_slice`, không qua bộ đếm vòng lặp. **Nó không đáng bật**, đo bằng thời gian thiết bị của tất cả kernel của nó (RTX 4050, schedule mặc định của op):

| shape (hàng x K x N) | không chia | split-K | K-folding (chỉ 16 hàng) |
|---|--:|--:|--:|
| 16 x 8960 x 1536 | 70.3 us | 60.0 (S=5) | 46.0 |
| 16 x 4864 x 896 | 28.3 | 26.6 (S=2) | 18.2 |
| 32 x 8960 x 1536 | 86.5 | 66.2 (S=5) | |
| 32 x 4864 x 896 | 36.9 | 28.7 (S=2) | |
| 64 x 4864 x 896 | 50.1 | 43.2 (S=2) | |
| 64 x 8960 x 1536 | 101.5 | 110.7 (S=5) | |
| 64 x 4096 x 4096 | 94.0 | 113.8 (S=4) | |
| 64 x 11008 x 4096 | 240.1 | 260.1 (S=4) | |
| 64 x 896 x 9728 | 51.3 | 165.6 (S=7) | |

Nó giúp các layer K dài, N hẹp nhanh hơn 14 đến 23 phần trăm ở 32 hàng, thua K-folding ở 16 hàng, và làm hầu hết layer chậm đi ở 64 hàng. Lý do là những thứ Marlin tránh bằng cách rút gọn ngay trong kernel: một kernel thứ hai cho phép reduce, thêm S lần M x N số thực được ghi rồi đọc lại, và một buffer cấp phát mỗi lần gọi, cũng sẽ phá việc capture CUDA graph (buffer tạm sẽ phải thành tham số của hàm). Sửa những điều đó (rút gọn trong kernel bằng atomic hoặc semaphore, buffer tạm truyền vào) mới là công việc thật của một kernel persistent kiểu Marlin, và trường hợp tốt nhất là 14 đến 23 phần trăm trên các layer chiếm một phần tư thời gian GEMM, tức khoảng 3 đến 5 phần trăm của một bước decode. Tôi dừng ở đó: mã nằm lại trong compiler như một tùy chọn thử nghiệm, có test, mặc định tắt, và chẩn đoán khoảng cách với Marlin ở 64 hàng là kết quả có ích.

## Các định dạng checkpoint khác

Plugin cũng đọc checkpoint AWQ 4-bit và GPTQ 8-bit (Qwen2.5-0.5B, cùng engine, token/s cộng trên các request, hai lần chạy khớp nhau trong 1 %; mỗi định dạng so với phương pháp Marlin của nó):

| định dạng | request | Marlin | QuantForge | QuantForge / Marlin | đầu ra so với Marlin |
|---|--:|--:|--:|--:|---|
| AWQ 4-bit (`awq_marlin`) | 1 | 307 | 270 | 0.88 | 287 trên 320 token greedy trùng, chênh log-xác suất tuyệt đối trung bình 0.0020 |
| | 4 | 1168 | 1040 | 0.89 | |
| | 16 | 4357 | 3811 | 0.87 | |
| | 64 | 11715 | 9509 | 0.81 | |
| GPTQ 8-bit (`gptq_marlin`) | 1 | 239 | 222 | 0.93 | 320 trên 320 trùng, chênh trung bình 0.0018 |
| | 4 | 919 | 856 | 0.93 | |
| | 16 | 3482 | 3235 | 0.93 | |
| | 64 | 10157 | 8600 | 0.85 | |

Đường GPTQ 8-bit cần một chi tiết chuyển đổi: GPTQ lưu byte không dấu, kernel nhận int8, nên cả weight lẫn zero point đều được dịch đi 128, giữ nguyên `q - zero`. AWQ khác GPTQ ở cách đóng gói (theo chiều đầu ra, với thứ tự nibble hoán vị), bộ chuyển đổi đảo ngược nó.

Các model khác (cùng engine, GPTQ 4-bit; token/s cộng trên các request, hai lần chạy khớp nhau trong 1 %):

| model | request | Marlin | QuantForge | QuantForge / Marlin | đầu ra so với Marlin |
|---|--:|--:|--:|--:|---|
| TinyLlama-1.1B (kiến trúc Llama, `desc_act=true`: act-order thật) | 1 | 232 | 209 | 0.90 | 320 trên 320 token greedy trùng, chênh log-xác suất tuyệt đối trung bình 0.0012 |
| | 4 | 904 | 804 | 0.89 | |
| | 16 | 3288 | 3003 | 0.91 | |
| | 64 | 8604 | 6547 | 0.76 | |
| Qwen2.5-3B | 1 | 81.5 | 71 | 0.87 | 320 trên 320 trùng, chênh trung bình 0.0019 |
| | 4 | 320 | 280 | 0.88 | |
| | 16 | 1215 | 1073 | 0.88 | |

K-folding không thay đổi gì cho model 3B (71 token/s có và không có): các layer hẹp của nó (2048 đầu ra) không lợi đủ để được fold. **Bộ nhớ:** lần chạy đầu của model 3B không khởi động được
trên card 6 GB (vLLM báo -0.32 GiB cho KV cache): op cấp một accumulator f32 cho toàn bộ số hàng của một lần gọi, và lượt profile của vLLM gửi 8192 token, tức 720 MB cho layer gate/up rộng 22016.
Op giờ xử lý đầu vào dài theo khối 2048 hàng ghi thẳng vào kết quả f16 (bộ nhớ tạm bị chặn, KV cache +0.1 GiB so với +0.24 của Marlin).

## Cập nhật: khoảng cách decode với Marlin đã trốn ở đâu (2026-11-22)

Khoảng cách decode còn lại (0.85 đến 0.9 của Marlin) không giống một vấn đề của kernel: theo từng lớp, các kernel nằm trong khoảng 0.88 đến 1.03 của Marlin, trừ `down_proj` trước khi K-fold. Bản tóm tắt `nsys` của cùng phép decode 128 token trong vLLM cho thấy điểm khác biệt: **33 313 kernel `float16_copy`** trong lần chạy QuantForge, mỗi cái 4 micro giây, mà Marlin không phát. Đếm kernel mỗi lần gọi op đã chốt vấn đề: một lần gọi decode thường gồm một bản sao đệm activation lên 16 hàng, GEMM vào bộ đệm f32, và một phép chuyển sang f16; có bias (q, k và v của Qwen) thì thêm hai kernel nhỏ để mở rộng nó vào accumulator, tổng cộng năm, so với một của Marlin. Với lớp qkv rộng 1152 ở một hàng, đó là 10.7 micro giây so với 6.4 của Marlin, trong đó GEMM chỉ 5.6.

Hai bước. Thứ nhất, một kernel fused sẵn có (epilogue của K-fold, với S = 1) cộng bias và ghi f16, đưa các lớp có bias từ năm kernel xuống ba. Rồi cách sửa đúng trong compiler: `epilogue=f16-store`, bộ máy của `int8-dequant` ở Cập nhật 3 (viết lại các lệnh lưu accumulator của GEMM thành các lệnh lưu f16 4 byte đã chuyển đổi và cộng bias, rồi xóa kernel elementwise và bộ đệm) áp dụng cho accumulator f32. Khi đó một GEMM thường là hai kernel, bản sao đệm và GEMM: qkv ở một hàng **từ 10.7 xuống 6.5 micro giây**, ngang 6.4 của Marlin; lớp gate / up từ 23.1 xuống 21.0 (Marlin 27.3). Nó cũng bỏ bộ đệm f32 khỏi prefill. Thay đổi đầu tiên làm một test fail có chủ đích: nó đòi hai cách (bắt đầu accumulator từ bias hay cộng bias sau) bằng nhau từng bit, hai thứ tự cộng f32 chỉ khớp tới một ulp f16; tôi nới nó thành dung sai cho trường hợp đó và giữ bằng nhau chính xác cho các activation.

vLLM 0.25.1, cùng checkpoint, token decode mỗi giây ở 1 / 4 / 16 / 64 request (QuantForge so với Marlin, xen kẽ trong một phiên): Qwen2.5-0.5B 286 / 1091 / 3974 / 10191 so với 303 / 1168 / 4333 / 11665, **0.94 / 0.93 / 0.92 / 0.87**; Qwen2.5-1.5B 128 / 461 / 1771 / 4711 so với 140 / 548 / 2070 / 5814, **0.91 / 0.84 / 0.86 / 0.81** (trước là 0.74 đến 0.87). Prefill của plugin tăng từ 20.6k lên 28.0k token mỗi giây ở 0.5B (Marlin 33.7k). Phương thức W4A8 thừa hưởng toàn bộ ở chế độ hai layout: decode 285 / 3988 ở 1 / 16 request trên 0.5B (0.94 / 0.93 của Marlin) với prefill 1.09x Marlin; chế độ một bản sao, dùng kernel decode riêng, cũng dùng epilogue mới (264 / 3952) và ở mức 0.87 / 0.92.


## Hạn chế của plugin

GPTQ 4-bit và 8-bit và AWQ 4-bit có zero point; không có layer MoE; AWQ không có zero point bị từ chối. Lần khởi động đầu của một model biên dịch một kernel cho mỗi shape layer và row bucket (khoảng 0.3 giây mỗi cái, sau đó được cache trên đĩa). Chỉ một kiến trúc GPU (sm_89) và hai model nhỏ được thử.
