---
title: QuantForge: sinh kernel GEMM INT4 / INT8 / FP8 bằng MLIR
date: 2026-11-20
order: 1
summary: Dự án là gì, pipeline compiler được tổ chức ra sao, và toàn bộ kết quả: thời gian kernel so với Marlin, tốc độ decode đầu-cuối trên model thật, và bằng chứng về độ đúng.
---

## Vì sao lượng tử hóa weight-only, và vì sao cần compiler

Khi decode tự hồi quy, LLM đọc **mọi weight một lần cho mỗi token sinh ra** và làm rất ít phép tính với nó: ở batch size 1, mỗi weight chỉ tham gia một phép nhân-cộng. Bước decode bị giới hạn bởi băng thông bộ nhớ, không phải sức tính. Lưu weight dạng số nguyên 4 bit thay vì số thực 16 bit làm một bước decode rẻ hơn tới bốn lần về byte, miễn là dequantisation đủ rẻ để ẩn sau lưu lượng bộ nhớ. Đây là ý tưởng đằng sau GPTQ, AWQ và các kernel như Marlin.

Kernel viết tay như Marlin chạm tới giới hạn băng thông, nhưng được viết cho từng shape và bit-width cụ thể. Câu hỏi của QuantForge là: **compiler có thể tiến gần đến đâu khi sinh kernel từ một mô tả cấp cao và một tập pass ghép được, không cần tuning tay cho từng shape?** Pipeline bắt đầu từ mô tả ở mức tensor của

```
C = A @ dequant(unpack(W_packed))        # A: activation fp16, dequant(q) = (q - zero) * scale theo nhóm 128 hàng
```

và kết thúc bằng PTX được gọi từ PyTorch.

## Pipeline

Đầu vào là một hàm trong dialect `qf`:

```mlir
func.func @w4a16_gemm(%act: tensor<MxKxf16>, %wp: tensor<Kx(N/2)xi8>,
                      %scale: tensor<(K/G)xNxf16>, %zp: tensor<(K/G)xNxi8>,
                      %c: tensor<MxNxf32>) -> tensor<MxNxf32> {
  %w  = qf.unpack %wp : tensor<Kx(N/2)xi8> -> tensor<KxNxi8>      // hai INT4 trong một byte
  %wf = qf.dequant %w, %scale, %zp : ... -> tensor<KxNxf16>        // (q - zero) * scale, theo nhóm
  %r  = linalg.matmul ins(%act, %wf) outs(%c)
  return %r
}
```

`quantforge-opt --quantforge-gpu-pipeline` hạ nó xuống qua bốn giai đoạn. Tên dưới đây là tên pass thật:

| giai đoạn | điều xảy ra | pass |
|---|---|---|
| mức tensor | dequantisation thành `linalg.generic` theo phần tử, unpack và dequant được gộp | `fuse-unpack-dequant`, `lower-dequant-to-arith` |
| tiling và ánh xạ | chia tile block / warp, ánh xạ lên lưới GPU, swizzle tile thân thiện L2 (tùy chọn) | `tiling`, `forall-to-gpu` (`block-swizzle`), `thread-distribute` |
| phân cấp bộ nhớ | tile A đi qua shared memory với pipeline `cp.async` nhiều stage; **dequantisation được tính lại ngay khi tile weight được đưa vào**, nên chỉ byte INT4 đã đóng gói mới đi từ DRAM | `smem-promotion`, `fuse-producer-into-loads` |
| mức lệnh | load vector 16 byte, dequantisation `f16x2` đóng gói, accumulator trong register, `mma.sync.m16n8k16`, look-ahead cho load weight, epilogue gộp | `simplify-index-math`, `vectorize-global-loads`, `vectorize-dequant`, `hoist-vector-transfers`, `convert-vector-to-gpu`, `prefetch-weight-loads`, `fuse-epilogue`, `overwrite-output` |

Kết quả là một kernel duy nhất cho mỗi GEMM. Weight f16 đã dequantise không bao giờ được ghi ra bộ nhớ; đó là lý do kernel INT4 nhanh hơn GEMM fp16 của cuBLAS từ 3.2 đến 3.7 lần ở decode.

### Runtime

PTX sinh ra được đóng thành shared library và nạp qua một shim CUDA driver nhỏ. Bên trên có:

* PyTorch custom op `torch.ops.quantforge.w4a16_gemm` đăng ký trong dispatcher từ C++ (khoảng 45 micro giây thời gian host mỗi lần gọi từ Python, thay vì 210 đến 270 với launcher Python thuần ban đầu), chạy ở chế độ eager, dưới `torch.compile` và trong CUDA graph;
* cache kernel đã biên dịch bền vững trên đĩa và **row bucket** (M tới 16, 32, 64, 128, 256, ... mỗi nhóm một kernel) để server gặp prompt dài từ 1 đến 4096 chỉ biên dịch khoảng một chục kernel cho mỗi shape layer, thay vì một kernel cho mỗi độ dài;
* autotuner đo các tùy chọn tile một lần cho mỗi shape và lưu lại kết quả tốt nhất.

Bốn schedule kernel được chọn theo số hàng activation M:

| M | schedule |
|---|---|
| 1 đến 16 | *decode*: một tile Tensor Core 16 hàng cho mỗi block, N rải trên các block, weight chỉ đọc đúng một lần |
| 17 đến 32 / 33 đến 64 | *skinny*: tile block 32 hoặc 64 hàng |
| 65 đến 256 | tile 64 hoặc 128 hàng |
| trên 256 | *prefill*: tile 128 x 128, pipeline 3 stage cho weight lớn |

## Kết quả: kernel so với Marlin

Mọi số kernel là **thời gian device từ trace CUPTI**, tốt nhất của nhiều lần chạy, với weight đọc từ DRAM: L2 được flush trước mỗi lần gọi bằng cách *đọc* một buffer 256 MB (ghi vào nó để lại các dòng cache dirty làm kernel kế tiếp chậm 20 đến 50 phần trăm), và activation được chạm lại sau đó vì trong model thật layer trước vừa ghi chúng. Marlin là `marlin_gemm` của vLLM 0.25.1 trên cùng giá trị INT4 và scale. Group size 128.

{{chart:marlin_ratio}}

| M x K x N | Marlin (us) | QuantForge (us) | cuBLAS fp16 (us) | QuantForge / Marlin |
|---|--:|--:|--:|--:|
| 1 x 4096 x 4096 | 49.6 | 50.0 | 180.9 | 0.99x |
| 4 x 4096 x 4096 | 49.4 | 49.8 | 181.5 | 0.99x |
| 16 x 4096 x 4096 | 50.0 | 49.8 | 182.3 | 1.01x |
| 1 x 4096 x 11008 | 126.6 | 149.3 | 482.9 | 0.85x |
| 1 x 11008 x 4096 | 127.0 | 131.0 | 483.6 | 0.97x |
| 16 x 4096 x 11008 | 127.2 | 149.2 | 498.5 | 0.85x |
| 32 x 4096 x 4096 | 50.9 | 58.8 | 181.0 | 0.87x |
| 32 x 4096 x 11008 | 130.3 | 149.3 | 497.3 | 0.87x |
| 64 x 4096 x 4096 | 58.3 | 87.4 | 181.4 | 0.67x |
| 64 x 4096 x 11008 | 139.9 | 200.0 | 506.9 | 0.70x |
| 128 x 4096 x 4096 | 105.1 | 152.2 | 188.7 | 0.69x |
| 256 x 4096 x 4096 | 201.0 | 263.0 | 208.2 | 0.76x |
| 512 x 4096 x 4096 | 391.5 | 461.6 | 373.4 | 0.85x |
| 2048 x 4096 x 4096 | 1532.1 | 1775.8 | 1453.0 | 0.86x |

{{chart:kernel_us}}

Bảng nói lên điều gì:

* **Decode (M tới 16)**: ở 4096 x 4096 kernel sinh ra **ngang Marlin** (0.99 đến 1.01x, khoảng 50 micro giây, đúng là giới hạn DRAM: `ncu` đo được 9.2 MB đọc cho 8.65 MB weight và scale, khoảng 184 GB/s); với shape rộng 4096 x 11008 đạt 85 phần trăm, với 1 x 11008 x 4096 đạt 97 phần trăm. (Phiên bản đầu của bài này báo 1.30x ở 4096 x 4096; con số đó không qua được lần đo lại, xem [bài về decode](02-decode-instruction-bound.html).)
* **Prefill (M từ 512 trở lên)**: 85 đến 86 phần trăm của Marlin, chậm hơn cuBLAS fp16 khoảng 20 đến 25 phần trăm, trong khi cuBLAS đọc nhiều gấp bốn lần byte nhưng không phải dequantise: ở đây kernel bị giới hạn bởi tính toán và dequantisation tranh slot issue với Tensor Core.
* **M từ 32 đến 256** là điểm yếu, chậm hơn 13 đến 33 phần trăm. [Một bài riêng](04-mid-batch-postmortem.html) điều tra chi tiết.

## Kết quả: toàn bộ model

Kernel chỉ hữu ích nếu một model thật chạy đúng và nhanh hơn. `quantforge.nn.quantize_model` thay các layer `nn.Linear` của bất kỳ causal LM Hugging Face nào, gộp các phép chiếu q/k/v thành một GEMM và các phép chiếu gate/up của MLP SiLU-gated thành một GEMM với epilogue gộp `silu(gate) * up`. Bảng là decode batch 1 với CUDA graph, perplexity trên 8 x 512 token của tập test Wikitext-2, INT4 round-to-nearest đơn giản (group 128) với head INT8.

{{chart:e2e_speed}}

| model | ppl fp16 | tok/s fp16 | ppl INT4 | tok/s INT4 | weight fp16 -> INT4 |
|---|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 19.93 | 129 | 24.08 | 209 | 0.72 -> 0.19 GB |
| Qwen2.5-1.5B-Instruct | 13.48 | 49 | 15.31 | 97 | 2.62 -> 0.69 GB |
| Qwen2.5-3B-Instruct | không vừa 6 GB | | 14.29 | 61 | 5.55 -> 1.45 GB |
| Llama-3.2-1B-Instruct | 20.15 | 63 | 24.42 | 142 | 1.95 -> 0.51 GB |
| OLMo-2-0425-1B-Instruct | 19.32 | 60 | 21.31 | 138 | 2.15 -> 0.56 GB |
| Phi-3.5-mini-instruct (3.8B) | không vừa | | 9.62 | chưa đo | 7.25 -> 1.90 GB |
| Mistral-7B-Instruct-v0.3 (từ checkpoint NF4) | | | 8.51 (bitsandbytes NF4: 8.37) | 35.3 (bitsandbytes: 36.0) | đỉnh 4.65 GB |

Một model 3B và một model 7B chạy được trên card 6 GB vốn không chứa nổi chúng ở fp16. Cái giá về perplexity của 4 bit round-to-nearest là có thật (1 đến 4 điểm); [bài về GPTQ](05-gptq-and-checkpoints.html) cho thấy một quantiser thông minh hơn lấy lại được bao nhiêu. Mức tăng tốc giảm khi model lớn lên, vì lm_head, attention và khoảng 1400 kernel nhỏ của đồ thị Hugging Face chiếm phần lớn hơn của mỗi bước.

## Bằng chứng về độ đúng

Số tốc độ vô giá trị nếu kernel sai tinh vi, nên dự án dựa nhiều vào kiểm tra:

* mọi lần chạy đầu-cuối được so với một **bản sinh đôi dense**: cùng model nhưng weight INT4 được dequantise thành các layer `nn.Linear` fp16 bình thường. Perplexity khớp trong 0.002 và greedy decoding cho ra token giống hệt (48/48 hoặc 59/59) trên mọi model đã thử;
* checkpoint AWQ và GPTQ-Int4 thật của Qwen2.5-0.5B chạy qua thư viện khớp với bản sinh đôi dense: GPTQ 26.336 so với 26.335 perplexity, AWQ 26.689 so với 26.689;
* `compute-sanitizer` (memcheck và racecheck) sạch cho 54 biến thể pipeline, gồm các pipeline prefetch, epilogue gộp và mọi schedule ở trên;
* 37 test FileCheck bảo vệ cấu trúc code sinh ra (ví dụ "không có phép đổi int sang float vô hướng cho mỗi weight", lỗi hồi quy từng làm chậm 40 phần trăm).

## Cách dùng

```python
import torch, quantforge.nn as qnn
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16)
qnn.quantize_model(model, fmt="int4", group_size=128)          # int4 | int8 | fp8
qnn.quantize_model(model, "int8", only=("lm_head",), fuse=False)  # head INT8 không mất chất lượng
out = model.generate(ids.cuda(), max_new_tokens=64)            # chạy trên các kernel đã sinh
```

## Hạn chế, nói thẳng

* Mọi số đo đến từ **một GPU laptop** với xung nhịp dao động; chưa có kết quả trên GPU data-center.
* Baseline duy nhất là Marlin và cuBLAS. Machete, kernel Triton như gemlite, torchao và vLLM đầu-cuối chưa được so sánh.
* Kernel là PTX dựng bằng backend NVPTX của LLVM 18: không có Hopper `wgmma` hay TMA, không có activation FP8.
* M từ 32 đến 256 và prefill vẫn thua Marlin; các bài tiếp theo giải thích những gì đã thử.
