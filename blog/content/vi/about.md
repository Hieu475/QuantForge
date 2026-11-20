---
title: Giới thiệu
---
Blog này ghi lại **QuantForge**, một compiler kernel mã nguồn mở xây trên MLIR cho suy luận LLM lượng tử hóa weight-only (GEMM W4A16, W8A16 và FP8-weight trên GPU NVIDIA sm_80 trở lên), cùng các thí nghiệm xung quanh nó.

## Dự án gồm những gì

* Dialect MLIR `qf` (`qf.unpack`, `qf.dequant`) và khoảng hai mươi pass đưa phép nhân ma trận lượng tử hóa xuống PTX: tiling, đưa tile vào shared memory với pipeline `cp.async`, gộp dequantisation vào lúc load weight, load vector hóa, dequantisation `f16x2` đóng gói, accumulator giữ trong register, lowering Tensor Core `mma.sync`, prefetch weight, epilogue gộp.
* Một PyTorch custom op (`quantforge::w4a16_gemm`) với đường nóng bằng C++ dispatcher, chạy được với `torch.compile` và CUDA graph.
* `quantforge.nn` / `quantforge.gptq`: lượng tử hóa bất kỳ causal LM nào của Hugging Face (RTN hoặc GPTQ) hoặc nạp checkpoint AWQ / GPTQ thật.
* 37 test FileCheck, khoảng 70 test GPU đầu-cuối, và script `compute-sanitizer` (memcheck + racecheck) chạy 54 biến thể của pipeline.

## Cách đọc các con số

Mọi số đo đến từ **một GPU laptop** (RTX 4050, sm_89, 6 GB) vừa chạy desktop. Xung nhịp dao động tới 30 phần trăm giữa các phiên, nên tỷ lệ đo trong cùng một lần chạy (ví dụ so với Marlin) đáng tin hơn nhiều so với thời gian tuyệt đối. Chưa có kết quả trên GPU data-center (A100, H100); đây là hạn chế lớn nhất còn mở.

## Trạng thái

Prototype nghiên cứu. Nó hữu ích cho thí nghiệm và để hiểu compiler làm được và không làm được gì cho loại kernel này; nó không thay thế các engine suy luận production.
