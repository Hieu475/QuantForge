# QuantForge Phase 4: GPU Hardware Mapping

Tài liệu này mô tả quá trình ánh xạ mã MLIR (đã được tối ưu hóa ở các Phase trước) xuống kiến trúc phần cứng của GPU (Phase 4). Trọng tâm của Phase này là `GPUMappingPass`.

---

## Tổng quan

Sau khi mã được chia mảnh (tiling) đa tầng qua `TilingPass` (Block, Warp, Thread levels), cấu trúc lặp `scf.forall` và `scf.for` cần được ánh xạ vào mô hình thực thi của GPU (Grid, Block, Warp, Lane).

Mục tiêu của `GPUMappingPass` (`--quantforge-gpu-mapping`) là:
1. Ánh xạ các vòng lặp ngoài cùng (`scf.forall`) tương đương với các chiều của Grid/Block.
2. Thiết lập định mức luồng phần cứng thông qua các thuộc tính của hàm (`quantforge.block_size_{x,y,z}`).
3. Chèn các rào cản đồng bộ hóa bộ nhớ (`gpu.barrier`) vào các vòng lặp nạp dữ liệu từ SRAM (K-reduction loop).

## Cấu trúc Pass: `--quantforge-gpu-mapping`

**File:** `lib/Transforms/GPUMappingPass.cpp`

### Chức năng chính

1. **Hardware Mapping Strategy (Annotation-only):** 
   Pass này hoạt động theo cơ chế dán nhãn (annotation-based). Tính năng này tìm vòng lặp `scf.forall` khối ngoài cùng (block-level) và dán nhãn `gpu::GPUBlockMappingAttr` cho từng chiều.
   - Chiều 0 (`%m_blk`) -> `gpu.block_id.y`
   - Chiều 1 (`%n_blk`) -> `gpu.block_id.x`

2. **Cấu hình Kernel Dimensions:**
   Dựa trên các tham số cấu hình chia mảnh (tiling sizes), Pass thực hiện tính toán số luồng phần cứng và gán nó vào Attribute của `func.func` cha:
   - `quantforge.block_size_x` (mặc định = 32 lanes cho Tensor Cores)
   - `quantforge.block_size_y` (numWarpsN)
   - `quantforge.block_size_z` (numWarpsM)

3. **Memory Synchronization (gpu.barrier):**
   Trong kiến trúc Ampere/Hopper, việc nạp chéo (cooperative loading) dữ liệu từ Global Memory vào Shared Memory (SRAM) cần được đồng bộ hóa để tránh race conditions. Pass tự động dò tìm vòng lặp `scf.for` ở mức giảm chiều K (K-reduction loop) và chèn vào:
   - **Barrier 1:** Tại vị trí bắt đầu của `scf.for` thân K-loop (ngay sau khi dữ liệu đã được nạp từ Global lên Shared Memory) để đảm bảo toàn bộ block thread đã hoàn tất DMA copy.
   - **Barrier 2:** Tại vị trí kết thúc của `scf.for` thân K-loop (ngay trước lệnh `scf.yield`) để đảm bảo các thread đã tiêu thụ xong dữ liệu tại vòng lặp K hiện tại trước khi vòng lặp tiếp theo tiến hành ghi đè SRAM.

### IR Sinh Ra (Sau khi chạy Tiling và GPU Mapping)

```mlir
func.func @matmul_gpu_mapped(...) -> tensor<...> attributes {
  quantforge.block_size_x = 32 : index, 
  quantforge.block_size_y = 2 : index, 
  quantforge.block_size_z = 2 : index
} {
  %0 = scf.forall (%arg3, %arg4) = ... {
    
    // ...
    %1 = scf.for %arg6 = ... iter_args(...) {
      gpu.barrier // <-- Đồng bộ hóa tải dữ liệu (Barrier 1)
      
      // ... Vòng lặp tính toán nhân ma trận cho các vùng Warp/Thread
      %2 = scf.for ...
      
      %inserted_slice = tensor.insert_slice ...
      gpu.barrier // <-- Đồng bộ hóa tiêu thụ dữ liệu (Barrier 2)
      scf.yield %inserted_slice
    }
    scf.forall.in_parallel { ... }
    
  } {mapping = [#gpu.block<y>, #gpu.block<x>]} // <-- Ánh xạ sang cấu trúc Block GPU
  
  return %0
}
```

### Sử dụng

Để xem kết quả biến đổi thành công với Mapping Pass, cấu hình lệnh:

```bash
quantforge-opt --quantforge-tiling --quantforge-gpu-mapping input.mlir
```

*(Lưu ý: Quá trình chuyển đổi từ vòng lặp đã gán nhãn sang các khối API Low-level như `gpu.launch` sẽ được thực hiện bởi các pass bộ đệm - bufferization - nằm sau trong chuỗi Compile Pipeline của MLIR do đòi hỏi phải triệt tiêu Tensor Value Semantics trước).*
