---
title: Mixture-of-experts thành một lần launch kernel, một khoảng trống prefill chưa ai đo, và con số 0,66x chỉ là hiện vật
date: 2026-11-23
order: 10
summary: GEMM W4A16 theo nhóm qua hai pass của compiler (expert của mỗi khối hàng đọc từ mảng trên device, không đồng bộ với host), các số MoE đầu-cuối đầu tiên, prefill chỉ đạt 0,58 của Marlin MoE cho đến khi có hai bản sửa, và một chẩn đoán đuổi theo khoảng trống mà thực ra là KV cache.
---

Một layer mixture-of-experts là nhiều GEMM nhỏ cùng shape, khác weight. Launch một kernel cho mỗi expert làm phí GPU và buộc đồng bộ với host, mà CUDA graph cần shape tĩnh; lối ra là sắp các hàng theo expert, đệm đến cỡ tile hàng và chạy **một** kernel có `grid.z` là danh sách các bài toán khối hàng.

## Phía compiler

Hai pass (`ExpertBatchPass.cpp`, tùy chọn pipeline `expert-problems` và `expert-roles`) biến kernel dense thông thường thành kernel theo nhóm: `grid.z` trở thành chỉ số bài toán; kernel nhận thêm một đối số, mảng `int32` `meta` giữ expert của từng bài toán; một phần tử âm là đệm và block thoát ngay; mọi đối số weight, scale và zero point dịch đi `meta[z]` slab, còn đối số activation và đầu ra dịch đi `z` slab hàng. Mọi shape đều tĩnh và không có gì chờ host. Quanh hai GEMM theo nhóm (gate/up với epilogue SwiGLU, rồi down) là ba kernel CUDA nhỏ lo định tuyến, gom và kết hợp; plugin vLLM (`QuantForgeMoEMethod`) dùng bộ nạp weight và router của chính vLLM và chỉ thay phần tính expert.

## Cái đã đo, và một con số sai

OLMoE-1B-7B (64 expert, top-8), bước decode so với Marlin MoE của vLLM:

{{chart:moe_ratio}}

| request đồng thời | 1 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|
| Marlin MoE, token/s | 197 | 814 | 1556 | 2536 |
| QuantForge, token/s | 189 | 771 | 1475 | 2354 |
| tỉ lệ | 0,96 | 0,95 | 0,95 | 0,93 |

Phiên bản đầu của bảng này ghi **0,66** ở 32 request, và tôi đã dành nhiều ngày cho nó: trace `nsys`, roofline, phát lại định tuyến thật, các lần quét schedule đều đứng yên. Nó là hiện vật. Op đầu tiên ghi kết quả gate/up ra một buffer f32; buffer đó lấy từ KV cache, vLLM hết chỗ và preempt, và throughput của *cả lần chạy* giảm một phần ba trong khi mọi kernel nhanh như trước. Khi op không cần buffer đó nữa thì cùng phép đo cho 0,93. Bài học cũng nằm trong [bài W4A8](08-w4a8-beats-marlin.html): trước khi chẩn đoán một khoảng trống throughput trong vLLM, hãy so kích thước KV cache được báo và số lần preempt của hai lần chạy.

## Prefill: 0,58 lên 0,88

Mọi thứ ở trên là decode. Chưa ai đo prefill (16 prompt 1275 token, bước 1024 token): Marlin MoE của vLLM đạt 12117 / 11809 token/s còn op này 6873 / 6944, tức **0,58**. Hai nguyên nhân, cả hai đã sửa:

* Op xử lý batch lớn theo từng đoạn 256 token để chặn buffer gate/up f32, và mỗi đoạn lại đọc weight của cả 64 expert (khoảng 200 MB mỗi layer). Epilogue SwiGLU giờ ghi `silu(gate) * up` thẳng vào đầu ra f16 riêng của nó (`quantforge.swiglu_out`, một đối số hàm thứ hai mà pass gộp điền vào): không buffer f32, không kernel chuyển đổi, nên đoạn tăng lên 512 và weight được đọc ít hơn một nửa số lần.
* Các kernel theo nhóm dùng lại schedule theo số hàng của kernel dense, vốn đặt khối 64 hàng trong bài toán 128 hàng (weight bị đọc hai lần) với warp tile sai. Một lần quét tile block, warp và K với flush bằng đọc tìm ra một họ (tile K 64, một stage, nhìn trước 1, khối 128 cột) nhanh hơn schedule dense 17 đến 27 %; tile hàng là 16, 32 hoặc 64 theo số hàng trung bình mỗi expert (128 chưa bao giờ thắng).

Prefill giờ là 10616 / 10518 token/s, **0,88** của Marlin MoE, và văn bản sinh ra giống hệt Marlin MoE trên bốn prompt (log-xác suất token đầu lệch dưới 0,001).

## Một đường đã đo mà không xây

Kernel INT8 theo nhóm (kernel W4A8 qua cùng pass) nhanh hơn kernel W4A16 theo nhóm 5 đến 6 % ở 256 token (cả hai gần 192 GB/s của bộ nhớ) và 14 đến 15 % ở 512. Nhưng nó cần một kernel lượng tử hóa trước mỗi GEMM theo nhóm và một buffer gate/up f16 cộng một kernel SwiGLU-và-lượng-tử-lại ở giữa, khoảng 0,15 đến 0,2 ms lưu lượng mỗi layer, ăn gần hết phần lợi 0,34 ms: prefill sẽ từ 0,88 lên khoảng 0,92, chưa vượt 1, để đổi lấy một bộ nạp lượng tử hóa online mới trong vLLM. Không xây.

Giới hạn: một model chạy đầu-cuối (OLMoE), expert 4 bit dùng silu-gated, không có expert parallelism, shared expert chưa thử; shape Mixtral, Qwen-MoE và Qwen3-MoE biên dịch được vào kernel pack nhưng chưa chạy.
