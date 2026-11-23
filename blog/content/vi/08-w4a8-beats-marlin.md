---
title: Nhanh hơn Marlin chỉ khi dùng lõi INT8: W4A16 không làm được gì, và W4A8 tốn những gì
date: 2026-11-23
order: 8
summary: Đường W4A16 giữ số học chính xác vẫn ở mức 0,8 đến 0,94 của Marlin từ 32 request trở lên, dù thử mọi cách; kernel INT4 cho weight và INT8 cho activation với schedule đã đo vượt Marlin từ khoảng 128 request đồng thời. Các kết quả âm, và ba cái bẫy khi đo đã tốn nhiều ngày.
---

[Bài về vLLM](06-inside-vllm.html) kết thúc với một khoảng trống: ở 32 đến 256 request đồng thời, kernel W4A16 do compiler sinh ra chậm hơn Marlin. Bài này kể từ đó đến nay: một danh sách những thứ không lấp được khoảng trống, một thứ đã lấp được (bằng cách đổi phép tính chứ không phải schedule), và các lỗi trong phép đo trên đường đi.

## Những gì không lấp được khoảng trống W4A16

Tỉ lệ so với Marlin trong vLLM, Qwen2.5-1.5B, cùng weight, chạy xen kẽ: **0,94, 0,94, 0,89, 0,84, 0,80, 0,86, 0,85** ở 1, 16, 32, 64, 128, 192 và 256 request đồng thời. Profiler giải thích hình dạng đó:

| 128 x 4096 x 4096, weight lạnh (`ncu`) | Marlin | W4A16 do compiler sinh |
|---|--:|--:|
| thời gian | 115 us | 154 us |
| thanh ghi mỗi thread | 255 | 124 |
| block cùng cư trú mỗi SM | 1 (persistent) | khoảng 3 |
| băng thông DRAM | 50 % | 36 % |

Cả hai kernel đều bị giới hạn bởi độ trễ; Marlin giữ nhiều byte đang bay hơn với ít block nhưng to hơn, và nạp mảnh B thẳng từ bộ nhớ toàn cục vào thanh ghi (weight được xáo trước cho việc đó). Những thứ tôi đã thử mà không làm khoảng trống nhúc nhích: quét tile block, warp và K cho từng shape; split-K với atomics ở 128 hàng (165 us so với 155 us: thêm block không giúp gì, nên không phải do lượng tử hóa wave); một kernel duy nhất cho việc đệm hàng thay vì ba lần launch nhỏ (nằm trong nhiễu); tỉ lệ bộ nhớ giao cho vLLM (tỉ lệ y hệt ở 0,7 và 0,78, nên không phải hiệu ứng KV cache). Phần còn lại là định dạng weight, tức là bố cục khác và đường giải lượng tử khác, và tôi chưa làm.

## Đổi phép tính: W4A8

Lõi tensor INT8 làm gấp đôi việc của lõi fp16, và weight INT4 vẫn là lưu lượng weight duy nhất. Kernel (`W4A8GemmDq`) mở nibble thành int8 khi tile được đưa vào shared memory, cộng dồn int32 trên toàn K và áp một scale float cho mỗi kênh đầu ra ở epilogue. Định dạng là lượng tử hóa lũy tiến của QServe: INT8 theo kênh đầu ra, rồi INT4 theo nhóm 128 với scale và zero point *nguyên*, chọn sao cho `(q - z) * s` nằm trong int8 và phép tính 8 bit của kernel không tràn. Ba thứ làm nó nhanh chứ không chỉ đúng:

* **Scale K-major có thêm một hàng đệm.** 8 kênh đầu ra mà một thread giải lượng tử có 8 byte scale liền nhau theo thứ tự `[K/128, N]`, một lần nạp vector thay vì 8 byte cách nhau 32: nhanh hơn 3 đến 5 %. Các lần nạp trước của pipeline đọc scale của nhóm sau nhóm cuối, nên buffer có thêm một hàng toàn số 0 (bản đầu đọc quá cuối tensor).
* **Một planner biết đo.** Quy tắc schedule đã được chỉnh trên các layer có N <= 2048; ở 64 x 4096 x 4096 nó chạy 115 us trong khi schedule khác chạy 65 us. `int8_plan` đo một nhúm ứng viên cho mỗi (rows, K, N) đúng một lần, lưu cái thắng, và op chỉ tra cứu. Kernel pack mang theo các quyết định.
* **Kernel lượng tử hóa activation bản địa.** Lượng tử hóa activation theo token từng là một cặp kernel qua `torch.compile` cộng hai lần copy đệm; giờ một kernel CUDA lo amax, scale, làm tròn và đệm. Nó phải giống từng bit với biểu thức torch mà nó thay, và lúc đầu thì không: torch chia cho một số vô hướng bằng cách nhân với nghịch đảo float, và làm tròn half-to-even.

{{chart:w4a8_vs_marlin}}

| request đồng thời | 1 | 16 | 32 | 64 | 128 | 192 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|
| W4A16 (số học chính xác như Marlin) | 0,94 | 0,94 | 0,89 | 0,84 | 0,80 | 0,86 | 0,85 |
| W4A8, quay Hadamard (mặc định) | 0,92 | 0,93 | 0,97 | 0,98 | 1,01 | 1,11 | 1,16 |
| W4A8, không quay | 0,95 | 0,96 | 0,99 | 0,99 | 1,04 | 1,13 | 1,18 |

Đọc cho trung thực: W4A8 nhanh hơn Marlin từ khoảng 128 request đồng thời (và từ 64 hàng ở mức một GEMM), dưới đó nó chậm hơn 2 đến 8 %, vì đến 64 hàng nó dùng chính các kernel W4A16 như trước. Nó cũng không cùng phép tính: checkpoint của Marlin là W4A16, đường này lượng tử hóa activation xuống INT8. [Bài tiếp theo](09-hadamard-rotation.html) nói về cái giá chất lượng của việc đó, và phép quay biến cái giá thành lợi như thế nào.

## Ba cái bẫy khi đo

* **Flush L2 bằng cách đọc, không phải ghi.** Bản quét của tôi flush bằng `zero_()`; các dòng bẩn được ghi lại trong lúc kernel được đo làm *mọi* thời gian tăng 30 đến 50 us, và điều đó đổi cả schedule thắng. Cùng lỗi đó đã thổi phồng các số MoE đầu tiên.
* **vLLM đổi dtype mặc định của torch khi nạp model.** Planner tạo tensor thử ngẫu nhiên bằng `torch.rand`, vốn là f16 trong vLLM và f32 ở mọi nơi khác; kernel đọc buffer f16 như f32 và báo địa chỉ không hợp lệ *chỉ trong vLLM*. `compute-sanitizer python script.py` trên bước nạp engine tìm ra nó trong vài phút sau nhiều ngày nhìn mãi; cách sửa là truyền `dtype=` ở mọi nơi.
* **So throughput trong vLLM thực chất là so KV cache.** Mọi buffer phụ mà một op cấp phát đều làm KV cache nhỏ đi, vLLM preempt, và tỉ lệ lệch tới 30 %. [Bài về MoE](10-moe-grouped-gemm.html) có trường hợp tệ nhất.
