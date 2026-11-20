---
title: QuantForge bên trong vLLM: một phép so sánh đầu-cuối công bằng, và 20 đến 30 phần trăm còn thiếu đi đâu
date: 2026-11-20
order: 6
summary: Một plugin lượng tử hóa của vLLM chạy các kernel đã sinh trong một engine phục vụ thật, throughput so với Marlin ở 1 đến 64 request đồng thời, và một chẩn đoán theo từng layer chỉ ra một điểm yếu của kernel (K dài, N hẹp).
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

Kết quả đầu ra khớp nhau: trên năm prompt, bốn prompt sinh cùng 64 token greedy với cả hai kernel và prompt thứ năm khớp 34 token trước khi một thế gần hòa đổi chỗ (làm tròn fp16 khác nhau); chênh lệch tuyệt đối trung bình của log-xác suất các token sinh ra là 0.002 (tối đa 0.014). Vậy **plugin chạy đúng, và bên trong một engine phục vụ thật QuantForge đạt 70 đến 80 phần trăm throughput của Marlin**, không phải 100 phần trăm mà bảng kernel batch 1 cho layer 4096 x 4096 có thể gợi ý.

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

Điều này lật ngược một kết luận tôi đã viết trước đó ("split-K không phải đòn bẩy"): nó được đo ở 1 x 11008 x 4096, nơi N = 4096 đã cho 64 block. Các shape quan trọng cho decode trong một engine phục vụ thì khác, và phép so sánh engine tìm ra chúng ở nơi benchmark tách rời không tìm ra. Việc kernel tiếp theo là một schedule split-K cho các layer này (một dải K cho mỗi `blockIdx.z`, cộng dồn bằng atomic hoặc một bước rút gọn thứ hai), cộng với gộp phép pad và phép đổi f16 vào kernel.

## Hạn chế của plugin

Chỉ GPTQ 4-bit: không AWQ, không 8-bit, không layer MoE. Lần khởi động đầu của một model biên dịch một kernel cho mỗi shape layer và row bucket (khoảng 0.3 giây mỗi cái, sau đó được cache trên đĩa). Chỉ một kiến trúc GPU (sm_89) và hai model nhỏ được thử.
