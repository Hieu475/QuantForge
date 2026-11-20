---
title: Vì sao batch 32 đến 256 chậm hơn Marlin? Một bài mổ xẻ với prototype CUDA viết tay
date: 2026-11-20
order: 4
summary: Profiling, ablation và một kernel kiểu Marlin viết tay để tìm ra điều gì giới hạn các kernel sinh ra ở batch cỡ vừa. Hầu hết giả thuyết đều sai, và các kết quả âm mới là phần hữu ích.
---

Bảng kernel trong [bài tổng quan](01-overview.html) có một vùng khó coi. Ở batch 1 đến 16 kernel sinh ra ngang Marlin; ở 512 hàng trở lên chúng đạt 85 phần trăm; nhưng ở **32, 64, 128 và 256 hàng chúng chỉ bằng 87, 67, 69 và 76 phần trăm Marlin**. Vùng này quan trọng trong thực tế: đó là thứ một server thấy khi có vài request đồng thời hoặc prompt ngắn. Bài này là cuộc điều tra. Kết luận không phải "đây là cách sửa"; nó là danh sách những thứ *không phải* nguyên nhân, kèm số đo.

| M | Marlin (us) | QuantForge (us) | tỷ lệ |
|---|--:|--:|--:|
| 32 | 50.9 | 58.8 | 0.87 |
| 64 | 58.3 | 87.4 | 0.67 |
| 128 | 105.1 | 152.2 | 0.69 |
| 256 | 201.0 | 263.0 | 0.76 |

(K = N = 4096, group 128, weight từ DRAM.) Ở M = 64 khối lượng tính là 2.1 GFLOP, khoảng 47 micro giây ở 45 TFLOP/s mà cuBLAS đạt trên card này; Marlin làm trong 59.

## Giả thuyết 1: schedule sai

Với M từ 17 đến 64 phiên bản đầu dùng tile 64 hàng dành cho prefill. Một lần quét theo block tile M (16, 32, 64, 128), warp tile, K tile, stage và look-ahead (tới khoảng 190 cấu hình mỗi M) tìm được các cấu hình tốt hơn: cho 32 hàng là tile block 32 hàng với K tile sâu 128 (**68.6 xuống 58.6 micro giây**), cho 64 hàng là tile 64 hàng với warp tile 32 x 64 và look-ahead một tile (**97 xuống 87 micro giây**). Cả hai giờ là mặc định, và autotuner tìm trong cùng không gian đó. Điều này đưa M = 32 từ 0.75x lên 0.87x và M = 64 từ 0.61x lên 0.67x của Marlin. Sau đó các lần quét chạm trần: ở M = 64, tám cấu hình tốt nhất nằm trong khoảng 87 đến 94 micro giây. Tiling đã cạn.

## Giả thuyết 2: không đủ song song

Lưới cho M = 64 chỉ có 64 block trên 20 SM. Nếu thiếu việc là vấn đề thì thêm block phải giúp. Nhưng không: bài toán 64 x 1024 x 16384 có đúng số weight bằng 64 x 4096 x 4096 nhưng nhiều gấp bốn lần block (256 thay vì 64, mỗi block có một phần tư vòng lặp K), và nó chạy 110 micro giây so với 102. Thời gian đi theo lượng việc mỗi SM, không theo số SM rảnh. Block tile 16 hàng nhỏ hơn (nhiều gấp bốn lần block, nhưng lưu lượng weight gấp bốn vì mỗi block đọc lại weight) *chậm hơn* 30 phần trăm.

## Giả thuyết 3: DRAM

`ncu` ở M = 64: thông lượng DRAM **52 phần trăm** đỉnh. Không phải giới hạn.

## Giả thuyết 4: Tensor Core đã bão hòa

Tensor pipe hoạt động: **51 phần trăm**, slot issue bận: **36 phần trăm**. Không cái nào bão hòa. Các lý do stall phân bố gần đều nhau, mỗi lý do tốn 1.0 đến 1.4 chu kỳ cho mỗi lệnh được issue: áp lực hàng đợi shared-memory (`mio_throttle` 1.38), Tensor pipe bận tạm thời (`math_pipe_throttle` 1.23), phụ thuộc độ trễ cố định (`wait` 1.34), độ trễ load global (`long_scoreboard` 1.20), barrier (0.96). Không có thủ phạm đơn lẻ, đó là dấu hiệu của một kernel mà các pha không chồng lên nhau tốt.

## Giả thuyết 5: công việc số nguyên của dequantisation

Histogram lệnh của vòng lặp chính kernel M = 64 cho thấy kẻ tình nghi hiển nhiên:

| opcode | lệnh warp | tỷ trọng |
|---|--:|--:|
| LOP3 | 697 K | 21 % |
| HMMA (việc hữu ích) | 524 K | 16 % |
| SHF | 345 K | 10 % |
| LDSM (`ldmatrix`) | 328 K | 10 % |
| HMUL2 + HADD2 | 524 K | 16 % |
| IMAD, SGXT, còn lại | phần còn lại | |

Khoảng 60 phần trăm số lệnh được thực thi là dequantisation; chỉ 16 phần trăm là lệnh Tensor Core. Marlin tránh được phần lớn số học tách nibble bằng cách sắp xếp trước nibble offline: ghép nibble `p` với nibble `p + 4` của một từ 32 bit, để một shift và một `LOP3` cho ra ngay một cặp `f16x2` (7 thay vì khoảng 16 phép số nguyên cho 8 weight).

Trước khi thiết kế lại layout weight tôi đã **làm ablation**: phát ra cách tách nibble rẻ với layout sai (nhưng được đo thời gian) và đo. Kết quả: **3 phần trăm ở M = 64, 2.6 phần trăm ở 512, 1.2 phần trăm ở 2048 x 4096 x 4096**. Kernel không bị giới hạn bởi số học số nguyên dù nó chiếm phần lớn số lệnh, vì slot issue chỉ bận 36 phần trăm. Đổi layout weight toàn hệ thống chỉ mua được 3 phần trăm. Bỏ.

## Giả thuyết 6: B nên ở lại trong register như Marlin

Khác biệt cấu trúc với Marlin là nơi các weight đã dequantise cư trú. Kernel sinh ra đưa tile B đã dequantise qua shared memory: dequantise, `st.shared`, `__syncthreads`, `ldmatrix`, `mma`. Marlin repack weight offline để mỗi thread load 16 byte đúng bằng các fragment B của tám tile 8 cột của nó, dequantise trong register và đưa thẳng vào `mma`: không shared memory, không barrier cho B. Cài đặt điều đó trong pipeline MLIR sẽ là dự án nhiều tuần, nên trước tiên tôi dựng **cận trên bằng tay**.

Prototype là một file CUDA duy nhất (`benchmarks/proto/bir_w4a16.cu` trong repository, khoảng 300 dòng):

* repack offline: mỗi bước `k16`, mỗi dải 64 cột, mỗi lane có một `uint4` chứa 32 nibble, sắp xếp để `(w >> 4t) & 0x000F000F` trực tiếp là một thanh ghi half2 B của lệnh `m16n8k16` (cách ghép nibble của Marlin);
* dequantisation: `lop3`, `sub.f16x2`, `mul.f16x2` đi thẳng vào toán hạng `mma`;
* A đi qua shared memory bằng `cp.async` và đọc bằng `ldmatrix`; bốn warp mỗi block, mỗi warp sở hữu một dải 64 cột trên mọi hàng M; K chia giữa các block và cộng dồn bằng atomic;
* một vòng load weight prefetch (`PF` bước), scale và zero point load một lần cho mỗi nhóm 128 hàng;
* được kiểm với reference f32 (sai số tối đa 0.0003).

Quét số warp mỗi block (2, 4, 8), số lát K (1 đến 8), số stage của A (2 đến 4) và khoảng prefetch (2 đến 8), các cấu hình tốt nhất:

| M | prototype | kernel sinh ra | Marlin |
|---|--:|--:|--:|
| 32 | 57.3 us | 58.8 us | 50.9 us |
| 64 | **75.8 us** | 87.4 us | 58.3 us |

Vậy **giữ B trong register đáng khoảng 13 phần trăm ở M = 64 và không đáng gì ở M = 32** trong một cài đặt thẳng thắn, và prototype vẫn chậm hơn Marlin 28 phần trăm ở M = 64. Nó dùng 254 register mỗi thread và chạy một warp mỗi scheduler, khiến nó bị giới hạn bởi độ trễ; phần còn lại là tuning occupancy và lập lịch của một kernel viết tay (áp lực register, hai block mỗi SM, lập lịch lệnh), loại công việc mà kernel viết tay nhận được, không phải thứ mà một thay đổi ở đường bộ nhớ của compiler sẽ tự mang lại.

## Tôi rút ra gì

1. Với các kernel này, *bảng lệnh là một chỉ báo kém về tốc độ*: dequantisation là 60 phần trăm số lệnh và đáng 3 phần trăm thời gian khi loại bỏ.
2. Quét tile rẻ và cần thiết, nhưng nó bão hòa; khi tám cấu hình tốt nhất nằm trong 8 phần trăm của nhau thì cấu trúc mới là giới hạn.
3. Một prototype viết tay là cách nhanh nhất để chặn một dự án compiler lớn. Vài giờ CUDA trả lời một câu hỏi mà lẽ ra tốn nhiều tuần việc MLIR: cách tiếp cận đó cùng lắm cho thêm 13 phần trăm ở M = 64.
4. Khoảng cách còn lại với Marlin ở batch cỡ vừa có lẽ không phải khoảng cách compiler-so-với-người mà là khoảng cách *độ sâu tuning*. Một kernel sinh ra với ngân sách register và lập lịch của Marlin sẽ cần một scheduler hiểu áp lực register, thứ mà pipeline này chưa có.

Các kernel sinh ra giữ nguyên, với mặc định đã tuning. Prototype được giữ trong repository như một thí nghiệm có tài liệu.
