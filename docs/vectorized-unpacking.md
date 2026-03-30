# QuantForge: Vectorized INT4 Unpacking

Tài liệu này mô tả 3 hướng phát triển của quá trình giải nén INT4 trong QuantForge, từ thiết kế ban đầu (có branch) đến IR sẵn sàng cho PTX.

---

## Bối cảnh

QuantForge biểu diễn INT4 packed weights dưới dạng `tensor<K x N x i8>`, trong đó mỗi byte chứa hai nibble (4-bit):

```
byte = [hi_nibble(7:4) | lo_nibble(3:0)]
```

Mục tiêu: giải nén về `tensor<K x 2N x i8>` (hoặc trực tiếp ra `f16` sau dequantization) **mà không cần branch, không tốn thanh ghi phụ, và sinh PTX instruction tối ưu**.

---

## Thiết kế Ban Đầu (Có Branch — Baseline)

Pass `--lower-unpack-to-arith` và `--fuse-unpack-dequant` sử dụng một `linalg.generic` duy nhất trên không gian vòng lặp `(K, N, 2)`:

```mlir
%tmp = linalg.generic
    {indexing_maps = [#map_KN, #map_KN2], iterator_types = [...]}
    ins(%packed : tensor<K x N x i8>)
    outs(%empty  : tensor<K x N x 2 x i8>) {
  ^bb0(%byte: i8, %_: i8):
    %low   = arith.andi  %byte, 0x0F
    %sh    = arith.shrui %byte, 4
    %high  = arith.andi  %sh,   0x0F
    %d2    = linalg.index 2           // ← đọc chỉ số vòng lặp thứ 3
    %cond  = arith.cmpi eq, %d2, 0
    %out   = arith.select %cond, %low, %high  // ← branch!
    linalg.yield %out
}
%result = tensor.collapse_shape %tmp [[0],[1,2]] …
```

**Vấn đề:**

| Vấn đề | Hệ quả |
|--------|--------|
| `linalg.index` trong loop body | Ngăn GPU vectorize vòng lặp nội tại |
| `arith.select` (conditional) | Warp divergence khi GPU cần chọn giữa low/high |
| Không gian vòng lặp 3 chiều `(K, N, 2)` | Làm rộng trip count, khó tile |

---

## Phase 1: Branch-Free Vectorized Unpacking

**Passes mới:**
- `--lower-unpack-branch-free`
- `--fuse-unpack-dequant-branch-free`

**File:** `lib/Transforms/LowerUnpackBranchFree.cpp`, `lib/Transforms/FuseUnpackDequant.cpp`

### Ý tưởng

Thay vì một kernel với branch, tách thành **hai kernel độc lập hoàn toàn** — một cho low nibble, một cho high nibble — rồi **interleave** output bằng `tensor.insert_slice` stride-2:

```
out[k][2j]   = low_nibble(packed[k][j])
out[k][2j+1] = high_nibble(packed[k][j])
```

### IR Sinh Ra

```mlir
// Kernel 1: Low nibble (chỉ andi — zero branch)
%lows = linalg.generic {maps=[#id,#id], iter=["par","par"]}
    ins(%packed : tensor<K x N x i8>) outs(%e0 : tensor<K x N x i8>) {
  ^bb0(%b: i8, %_: i8):
    %m   = arith.constant 15 : i8
    %lo  = arith.andi %b, %m
    linalg.yield %lo
}

// Kernel 2: High nibble (shrui + andi — zero branch)
%highs = linalg.generic {maps=[#id,#id], iter=["par","par"]}
    ins(%packed : tensor<K x N x i8>) outs(%e1 : tensor<K x N x i8>) {
  ^bb0(%b: i8, %_: i8):
    %s4  = arith.constant 4  : i8
    %m   = arith.constant 15 : i8
    %sh  = arith.shrui %b, %s4
    %hi  = arith.andi  %sh, %m
    linalg.yield %hi
}

// Interleave: stride-2 insert_slice
%buf = tensor.empty() : tensor<K x 2N x i8>
%r0  = tensor.insert_slice %lows  into %buf[0,0] [K,N] [1,2]
%out = tensor.insert_slice %highs into %r0  [0,1] [K,N] [1,2]
```

### Phiên bản Fused (Unpack + Dequant)

`--fuse-unpack-dequant-branch-free` kết hợp nibble extraction với dequantization ngay trong loop body:

```mlir
// Low nibble + dequant
%lo_fp = linalg.generic ins(%packed, %scale, %zp) outs(%e0 : tensor<K x N x f16>) {
  ^bb0(%b: i8, %s: f16, %z: i8, %_: f16):
    %nibble = arith.andi %b, 0x0F
    %fp     = arith.sitofp %nibble
    %zpfp   = arith.sitofp %z
    %r      = arith.mulf (arith.subf %fp, %zpfp), %s
    linalg.yield %r
}

// High nibble + dequant (tương tự, thêm shrui trước andi)
%hi_fp = linalg.generic …

// Interleave ra f16 output
%out = tensor.insert_slice %lo_fp into … [0,0] [K,N] [1,2]
%out = tensor.insert_slice %hi_fp into … [0,1] [K,N] [1,2]
```

### Lợi Ích

| Chỉ số | Baseline | Phase 1 |
|--------|----------|---------|
| `linalg.index` | 1 (per iteration) | 0 |
| `arith.select` | 1 (per iteration) | 0 |
| Số kernel | 1 (trên K×N×2) | 2 (trên K×N, độc lập) |
| Warp divergence | Có (branch trên `d2`) | Không |
| GPU vectorize | Bị chặn | Cho phép |

---

## Phase 2: PTX-Ready i32-Chunk Lowering

**Pass mới:** `--lower-unpack-to-nvvm`

**File:** `lib/Transforms/LowerUnpackToNVVM.cpp`

### Mục tiêu

Biểu diễn unpacking ở mức **scalar SCF loops** với ngữ nghĩa chính xác của PTX:

- Đọc **4 byte liên tiếp** một lần → tương đương `ld.global.u32` (1 memory transaction)
- Tái cấu trúc thành 1 thanh ghi `i32`
- Extract 8 nibbles bằng **constant shift immediates** → PTX: `shr.u32 reg, <imm>` + `and.b32`
- Không có `linalg.generic`, không có `linalg.index`

### Guards

| Điều kiện | Lý do |
|-----------|-------|
| Rank = 2 | Chỉ hỗ trợ weight matrix 2-D (LLM GEMV) |
| N tĩnh và N % 4 == 0 | Đảm bảo mỗi row chia đều thành i32 chunks |
| Doubling-shape | Same-shape semantic marker bị bỏ qua |

### IR Sinh Ra

```mlir
%out = tensor.empty() : tensor<K x 2N x i8>

%result = scf.for %k = 0 to K step 1 iter_args(%acc = %out) {
  %r2 = scf.for %ch = 0 to N/4 step 1 iter_args(%cur = %acc) {

    // Đọc 4 byte tại chunk*4 .. chunk*4+3
    %b0 = tensor.extract %packed[%k, %ch*4+0]
    %b1 = tensor.extract %packed[%k, %ch*4+1]
    %b2 = tensor.extract %packed[%k, %ch*4+2]
    %b3 = tensor.extract %packed[%k, %ch*4+3]

    // Pack → i32  (b0 | b1<<8 | b2<<16 | b3<<24)
    %w = arith.ori (extui b0), (shli (extui b1), 8)
       | (shli (extui b2), 16) | (shli (extui b3), 24)

    // Extract 8 nibbles — constant shifts:
    // n0 = trunci(andi(w,     0xF))       shift=0  → PTX: and.b32
    // n1 = trunci(andi(shrui(w, 4), 0xF)) shift=4  → PTX: shr.u32 imm + and.b32
    // n2 = trunci(andi(shrui(w, 8), 0xF)) shift=8
    // …
    // n7 = trunci(andi(shrui(w,28), 0xF)) shift=28

    // Ghi 8 nibble tại chunk*8 .. chunk*8+7
    %t0 = tensor.insert %n0 into %cur[%k, %ch*8+0]
    %t1 = tensor.insert %n1 into %t0 [%k, %ch*8+1]
    …
    scf.yield %t7
  }
  scf.yield %r2
}
```

### Ánh xạ sang PTX

```ptx
// Inner iteration (1 chunk = 4 packed bytes = 8 nibbles):
ld.global.u8   %r0, [addr+0]     // tensor.extract × 4
ld.global.u8   %r1, [addr+1]
ld.global.u8   %r2, [addr+2]
ld.global.u8   %r3, [addr+3]

// Pack (compiler thường fold thành ld.global.u32)
and.b32  %r0, %r0, 0xFF
shl.b32  %r1, %r1, 8
or.b32   %w,  %r0, %r1
…

// Extract nibbles — immediates tĩnh:
and.b32  %n0, %w, 0xF
shr.u32  %t,  %w, 4  ; and.b32 %n1, %t, 0xF
shr.u32  %t,  %w, 8  ; and.b32 %n2, %t, 0xF
…
shr.u32  %t,  %w, 28 ; and.b32 %n7, %t, 0xF

// Store
st.global.u8  [out+0], %n0
…
st.global.u8  [out+7], %n7
```

PTX compiler (`ptxas`) có thể tiếp tục thay thế `shr.u32 + and.b32` bằng `prmt.b32` khi thấy pattern đủ điều kiện.

---

## Phase 3: Late Dequantization & Tensor Core Fusion (Định hướng tương lai)

> **Trạng thái**: Chưa triển khai đầy đủ. Một số passes chuẩn bị sẵn trong Phase 3.
> Xem chi tiết: [docs/gpu-optimizations.md](gpu-optimizations.md)

### Vấn đề với kiến trúc hiện tại

Dù Phase 1 và Phase 2 đã loại bỏ branch và tối ưu bit manipulation, pipeline hiện tại vẫn **materialize** một `tensor<K x 2N x f16>` intermediate trước khi gọi `linalg.matmul`. Điều này nghĩa là:

```
HBM (INT4-packed) → Shared Mem → Registers (INT4) → f16 tensor (HBM!) → Tensor Core
                                                       ↑ write-back này phá vỡ mục tiêu
```

### Mục tiêu đúng đắn

```
HBM (INT4-packed) → Shared Mem → Registers
                                    ↓
                              unpack + dequant  ←─ on-the-fly trong register file
                                    ↓
                              mma.sync (Tensor Core)
                                    ↓
                              Kết quả f32/f16 accumulator
```

### Hướng triển khai

1. **Tiling**: `linalg.matmul` → tiled loops với tile size `(M_tile, N_tile, K_tile)` phù hợp Shared Memory
2. **GPU Mapping**: `scf.for` → `gpu.launch` / `gpu.func`
3. **NVGPU Dialect**:
   - `nvgpu.ldmatrix` — load INT4-packed tile từ Shared Memory vào registers (dạng đóng gói)
   - `nvgpu.mma.sync` — Tensor Core GEMM với accumulation
4. **Inline Dequant**: Chèn unpack + dequant giữa `nvgpu.ldmatrix` và `nvgpu.mma.sync`, hoàn toàn trong register file
5. **TMA (Tensor Memory Accelerator)** trên Hopper: async copy từ HBM → Shared Memory bằng `nvgpu.tma.async.load`

### Yêu cầu hạ tầng

| Thành phần | Trạng thái |
|-----------|-----------|
| `gpu.launch` mapping pass | Chưa có |
| Bufferization pipeline (`tensor` → `memref`) | Chưa có |
| `NVGPU dialect` integration | Chưa có |
| Tile size auto-tuning | Chưa có |

---

## Tóm tắt Kiến trúc Pass

```
qf.unpack
  ├── --lower-unpack-to-arith               (baseline, có branch)
  ├── --lower-unpack-branch-free            (Phase 1: branch-free, 2 generics)
  ├── --lower-unpack-to-nvvm                (Phase 2: SCF + i32-chunk, ~16 ALU ops)
  └── --lower-unpack-to-prmt             ★  (Phase 3: SCF + prmt.b32, ~10 ALU ops)

qf.unpack + qf.dequant
  ├── --fuse-unpack-dequant                 (baseline)
  └── --fuse-unpack-dequant-branch-free     (Phase 1: branch-free)

Optimization passes (Phase 3, chạy sau lowering): ★
  ├── --swizzled-unpack-indexing            (bank conflict prevention)
  ├── --register-layout-aware-unpack        (mma.sync fragment layout, skeleton)
  └── --canonicalize-dequant-zp             (symmetric quant fast-path)
```

★ Xem chi tiết Phase 3: [docs/gpu-optimizations.md](gpu-optimizations.md)

## Pipelines Đề Xuất

```bash
# Baseline (tương thích ngược)
quantforge-opt --convert-linalg-to-quantforge --fuse-unpack-dequant input.mlir

# Phase 1 — Branch-free (khuyên dùng cho GPU general)
quantforge-opt --convert-linalg-to-quantforge \
               --fuse-unpack-dequant-branch-free input.mlir

# Phase 2 — PTX-ready SCF (cho 2D static weight, N%4==0, pre-Ampere)
quantforge-opt --convert-linalg-to-quantforge \
               --lower-unpack-to-nvvm \
               --lower-dequant-to-arith input.mlir

# Phase 3 — Full Ampere/Hopper (sm_80+, khuyên dùng)
quantforge-opt --convert-linalg-to-quantforge \
               --lower-unpack-to-prmt \
               --swizzled-unpack-indexing \
               --lower-dequant-to-arith \
               --canonicalize-dequant-zp \
               input.mlir
```

## Test Files

| File | Pass | Số test |
|------|------|---------|
| `test/Transforms/lower_unpack_to_arith.mlir` | `--lower-unpack-to-arith` | 3 |
| `test/Transforms/fuse_unpack_dequant.mlir` | `--fuse-unpack-dequant` | 4 |
| `test/Transforms/lower_unpack_branch_free.mlir` | `--lower-unpack-branch-free`, `--fuse-unpack-dequant-branch-free` | 5 |
| `test/Transforms/lower_unpack_to_nvvm.mlir` | `--lower-unpack-to-nvvm` | 4 |
| `test/Transforms/lower_unpack_to_prmt.mlir` | `--lower-unpack-to-prmt` ★ | 4 |
| `test/Transforms/swizzled_unpack_indexing.mlir` | `--swizzled-unpack-indexing` ★ | 2 |
| `test/Transforms/register_layout_aware_unpack.mlir` | `--register-layout-aware-unpack` ★ | 2 |
| `test/Transforms/canonicalize_dequant_zp.mlir` | `--canonicalize-dequant-zp` ★ | 4 |

★ Phase 3 — xem [docs/gpu-optimizations.md](gpu-optimizations.md)
