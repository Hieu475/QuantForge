# QuantForge Development Roadmap

## 📊 Tổng quan tiến độ

| Phase | Tên | Trạng thái | Thời lượng dự kiến |
|-------|-----|------------|-------------------|
| 1 | PyTorch Baseline | ✅ HOÀN THÀNH | - |
| 2 | Dialect & Conversion Passes | ✅ HOÀN THÀNH | - |
| 3 | Vectorized Unpacking (Phase 1) | ✅ HOÀN THÀNH | - |
| 4 | PTX-Ready i32-Chunk Lowering (Phase 2) | ✅ HOÀN THÀNH | - |
| 5 | GPU Codegen & Tensor Core Fusion (Phase 3) | ❌ CHƯA LÀM | 6 tuần |

---

# PHASE 1: PyTorch Baseline ✅ HOÀN THÀNH

## Mục tiêu
Tạo ground truth để đối chiếu kết quả và benchmark latency baseline.

## Đã hoàn thành
- [x] File `python/quantforge/baseline.py`
- [x] Hàm `pack_int4()` và `unpack_int4()`
- [x] Hàm `gemv_int4_baseline()` cho W4A16 GEMV
- [x] Benchmark function với CUDA synchronization
- [x] Lưu golden tensors

## Output
```
test/golden_tensors.pt  # Chứa x, w_packed, scale, zero_point, golden
```

---

# PHASE 2: Dialect & Conversion Passes

## Tổng quan
```
TOSA/Linalg IR  →  ConvertToQuantForge  →  qf.unpack + qf.dequant + linalg.matmul
                           ↓
                   LowerQuantForgeOps   →  arith.shrui + arith.andi + arith.sitofp + arith.mulf
```

## Đã hoàn thành
- [x] `QuantForgeDialect.td` - Định nghĩa dialect
- [x] `QuantForgeOps.td` - Định nghĩa `qf.unpack` và `qf.dequant`
- [x] CMake build system
- [x] `quantforge-opt` tool

---

## Task 2.1: ConvertLinalgToQuantForge Pass

### Mục tiêu
Tự động nhận diện pattern quantized GEMV trong Linalg và chèn QuantForge ops.

### Files cần tạo

```
lib/Transforms/ConvertLinalgToQuantForge/
├── ConvertLinalgToQuantForge.cpp
└── CMakeLists.txt
```

### Chi tiết implementation

#### File: `lib/Transforms/ConvertLinalgToQuantForge/ConvertLinalgToQuantForge.cpp`

```cpp
//===----------------------------------------------------------------------===//
// ConvertLinalgToQuantForge Pass
// 
// Pattern: linalg.matmul với INT8 packed weights
// Output:  qf.unpack → qf.dequant → linalg.matmul (FP16)
//===----------------------------------------------------------------------===//

#include "QuantForge/Dialect/QuantForge/QuantForgeDialect.h"
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"
#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Pattern: Nhận diện linalg.matmul với weight là INT8 (packed INT4)
/// và scale/zero_point attributes → chèn qf.unpack + qf.dequant
class ConvertQuantizedMatmulPattern 
    : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // === Step 1: Kiểm tra điều kiện ===
    Value weight = op.getInputs()[1];  // B matrix
    auto weightType = weight.getType().dyn_cast<RankedTensorType>();
    
    if (!weightType)
      return failure();
    
    // Chỉ convert nếu weight là integer type (packed INT4)
    if (!weightType.getElementType().isInteger(8))
      return failure();
    
    // Kiểm tra attribute đánh dấu đây là quantized weight
    if (!op->hasAttr("quantized"))
      return failure();

    // === Step 2: Lấy scale và zero_point ===
    auto scaleAttr = op->getAttrOfType<DenseElementsAttr>("scale");
    auto zpAttr = op->getAttrOfType<DenseElementsAttr>("zero_point");
    
    if (!scaleAttr || !zpAttr)
      return failure();

    Location loc = op.getLoc();

    // === Step 3: Tạo qf.unpack ===
    // Input: (N, K/2) i8  →  Output: (N, K) i8
    auto unpackedShape = weightType.getShape().vec();
    unpackedShape.back() *= 2;  // Double last dimension
    
    auto unpackedType = RankedTensorType::get(
        unpackedShape, rewriter.getI8Type());
    
    auto unpackOp = rewriter.create<quantforge::UnpackOp>(
        loc, unpackedType, weight);

    // === Step 4: Tạo qf.dequant ===
    // Input: (N, K) i8, scale, zp  →  Output: (N, K) f16
    auto dequantType = RankedTensorType::get(
        unpackedShape, rewriter.getF16Type());
    
    // Materialize scale và zero_point thành tensor values
    auto scaleValue = rewriter.create<arith::ConstantOp>(loc, scaleAttr);
    auto zpValue = rewriter.create<arith::ConstantOp>(loc, zpAttr);
    
    auto dequantOp = rewriter.create<quantforge::DequantOp>(
        loc, dequantType, unpackOp.getResult(), scaleValue, zpValue);

    // === Step 5: Thay thế weight trong matmul ===
    SmallVector<Value> newInputs = {op.getInputs()[0], dequantOp.getResult()};
    
    // Tạo output tensor mới với FP16 type
    Value activation = op.getInputs()[0];
    auto actType = activation.getType().cast<RankedTensorType>();
    auto outputShape = op.getOutputs()[0].getType().cast<RankedTensorType>().getShape();
    
    auto newOutputType = RankedTensorType::get(
        outputShape, rewriter.getF16Type());
    
    // Empty tensor cho output
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, outputShape, rewriter.getF16Type());
    
    // Tạo matmul mới với FP16 weights
    auto newMatmul = rewriter.create<linalg::MatmulOp>(
        loc, newOutputType, newInputs, ValueRange{emptyOp});
    
    // Xóa attribute quantized
    newMatmul->removeAttr("quantized");
    newMatmul->removeAttr("scale");
    newMatmul->removeAttr("zero_point");

    rewriter.replaceOp(op, newMatmul.getResults());
    return success();
  }
};

/// Pass wrapper
struct ConvertLinalgToQuantForgePass
    : public PassWrapper<ConvertLinalgToQuantForgePass, OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToQuantForgePass)

  StringRef getArgument() const override { return "convert-linalg-to-quantforge"; }
  StringRef getDescription() const override {
    return "Convert quantized Linalg operations to QuantForge dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<quantforge::QuantForgeDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertQuantizedMatmulPattern>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::quantforge {

std::unique_ptr<Pass> createConvertLinalgToQuantForgePass() {
  return std::make_unique<ConvertLinalgToQuantForgePass>();
}

void registerConvertLinalgToQuantForgePass() {
  PassRegistration<ConvertLinalgToQuantForgePass>();
}

} // namespace mlir::quantforge
```

#### File: `lib/Transforms/ConvertLinalgToQuantForge/CMakeLists.txt`

```cmake
add_mlir_library(QuantForgeConvertLinalgToQuantForge
  ConvertLinalgToQuantForge.cpp

  DEPENDS
  QuantForgeOpsIncGen

  LINK_LIBS PUBLIC
  QuantForgeDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
)
```

### Test case

#### File: `test/Transforms/convert_linalg_to_quantforge.mlir`

```mlir
// RUN: quantforge-opt %s -convert-linalg-to-quantforge | FileCheck %s

// CHECK-LABEL: func.func @test_quantized_matmul
func.func @test_quantized_matmul(
    %activation: tensor<1x4096xf16>,
    %weight_packed: tensor<4096x2048xi8>) -> tensor<1x4096xf16> {
  
  %init = tensor.empty() : tensor<1x4096xf16>
  
  // Input: packed INT8 weights (mỗi byte chứa 2 INT4)
  // CHECK: qf.unpack
  // CHECK: qf.dequant
  // CHECK: linalg.matmul
  // CHECK-NOT: quantized
  %result = linalg.matmul {
    quantized,
    scale = dense<0.01> : tensor<4096xf16>,
    zero_point = dense<8> : tensor<4096xi8>
  } ins(%activation, %weight_packed : tensor<1x4096xf16>, tensor<4096x2048xi8>)
    outs(%init : tensor<1x4096xf16>) -> tensor<1x4096xf16>
  
  return %result : tensor<1x4096xf16>
}
```

### Tiêu chí hoàn thành Phase 2.1
- [ ] Pass compile thành công
- [ ] `quantforge-opt -convert-linalg-to-quantforge` chạy được
- [ ] Test case pass với FileCheck
- [ ] IR output chứa `qf.unpack` và `qf.dequant`

---

## Task 2.2: LowerQuantForgeOps Pass

### Mục tiêu
Hạ bậc `qf.unpack` và `qf.dequant` xuống các arith/math operations.

### Files cần tạo

```
lib/Transforms/LowerQuantForgeOps/
├── LowerQuantForgeOps.cpp
└── CMakeLists.txt
```

### Chi tiết implementation

#### File: `lib/Transforms/LowerQuantForgeOps/LowerQuantForgeOps.cpp`

```cpp
//===----------------------------------------------------------------------===//
// LowerQuantForgeOps Pass
//
// qf.unpack  →  arith.andi + arith.shrui
// qf.dequant →  arith.sitofp + arith.subf + arith.mulf
//===----------------------------------------------------------------------===//

#include "QuantForge/Dialect/QuantForge/QuantForgeDialect.h"
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"
#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// qf.unpack Lowering
//===----------------------------------------------------------------------===//

/// Lower qf.unpack to linalg.generic với arith ops
/// 
/// Mathematically:
///   low  = packed & 0x0F
///   high = (packed >> 4) & 0x0F
///   output[..., 2*i]   = low[..., i]
///   output[..., 2*i+1] = high[..., i]
class LowerUnpackOp : public OpConversionPattern<quantforge::UnpackOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quantforge::UnpackOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    
    auto inputType = input.getType().cast<RankedTensorType>();
    auto outputType = op.getResult().getType().cast<RankedTensorType>();
    
    // Constants
    Value c0x0F = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI8IntegerAttr(0x0F));
    Value c4 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI8IntegerAttr(4));

    // Create output tensor
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType());

    // Build linalg.generic để unpack
    // Input: (M, N/2), Output: (M, N)
    // Cần custom indexing map
    
    int64_t rank = inputType.getRank();
    SmallVector<AffineMap> indexingMaps;
    
    // Input map: (d0, d1, ..., d_last) -> (d0, d1, ..., d_last/2)
    // Cần handle special case cho last dimension
    
    SmallVector<AffineExpr> inputExprs;
    for (int64_t i = 0; i < rank - 1; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    // Last dim: floordiv by 2
    inputExprs.push_back(
        rewriter.getAffineDimExpr(rank - 1).floorDiv(2));
    
    indexingMaps.push_back(
        AffineMap::get(rank, 0, inputExprs, rewriter.getContext()));
    
    // Output map: identity
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{input}, ValueRange{emptyOp}, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value packed = args[0];
          
          // Get last dimension index to determine low/high
          // idx % 2 == 0 → low nibble, idx % 2 == 1 → high nibble
          Value lastIdx = b.create<linalg::IndexOp>(loc, rank - 1);
          Value c2 = b.create<arith::ConstantIndexOp>(loc, 2);
          Value remainder = b.create<arith::RemUIOp>(loc, lastIdx, c2);
          Value c0_idx = b.create<arith::ConstantIndexOp>(loc, 0);
          Value isLow = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, remainder, c0_idx);
          
          // low = packed & 0x0F
          Value low = b.create<arith::AndIOp>(loc, packed, c0x0F);
          
          // high = (packed >> 4) & 0x0F
          Value shifted = b.create<arith::ShRUIOp>(loc, packed, c4);
          Value high = b.create<arith::AndIOp>(loc, shifted, c0x0F);
          
          // Select based on index parity
          Value result = b.create<arith::SelectOp>(loc, isLow, low, high);
          
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// qf.dequant Lowering
//===----------------------------------------------------------------------===//

/// Lower qf.dequant to: output = (input - zero_point) * scale
class LowerDequantOp : public OpConversionPattern<quantforge::DequantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(quantforge::DequantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value scale = adaptor.getScale();
    Value zeroPoint = adaptor.getZeroPoint();
    
    auto inputType = input.getType().cast<RankedTensorType>();
    auto outputType = op.getResult().getType().cast<RankedTensorType>();
    
    // Create output tensor
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType());

    int64_t rank = inputType.getRank();
    
    // Indexing maps
    // Input: full tensor
    // Scale: per-channel (first dim)
    // ZeroPoint: per-channel (first dim)  
    // Output: full tensor
    
    SmallVector<AffineMap> indexingMaps;
    
    // Input: identity
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    
    // Scale: (d0) - broadcast
    indexingMaps.push_back(AffineMap::get(
        rank, 0, rewriter.getAffineDimExpr(0), rewriter.getContext()));
    
    // ZeroPoint: (d0) - broadcast
    indexingMaps.push_back(AffineMap::get(
        rank, 0, rewriter.getAffineDimExpr(0), rewriter.getContext()));
    
    // Output: identity
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{input, scale, zeroPoint},
        ValueRange{emptyOp}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value intVal = args[0];    // i8
          Value scaleVal = args[1];  // f16
          Value zpVal = args[2];     // i8
          
          // Convert int8 to f16
          Value intF16 = b.create<arith::SIToFPOp>(
              loc, b.getF16Type(), intVal);
          
          // Convert zero_point to f16
          Value zpF16 = b.create<arith::SIToFPOp>(
              loc, b.getF16Type(), zpVal);
          
          // (input - zero_point)
          Value diff = b.create<arith::SubFOp>(loc, intF16, zpF16);
          
          // diff * scale
          Value result = b.create<arith::MulFOp>(loc, diff, scaleVal);
          
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LowerQuantForgeOpsPass
    : public PassWrapper<LowerQuantForgeOpsPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerQuantForgeOpsPass)

  StringRef getArgument() const override { return "lower-quantforge-ops"; }
  StringRef getDescription() const override {
    return "Lower QuantForge dialect ops to Linalg/Arith";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    
    // QuantForge ops are illegal after this pass
    target.addIllegalDialect<quantforge::QuantForgeDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerUnpackOp, LowerDequantOp>(&getContext());
    
    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::quantforge {

std::unique_ptr<Pass> createLowerQuantForgeOpsPass() {
  return std::make_unique<LowerQuantForgeOpsPass>();
}

void registerLowerQuantForgeOpsPass() {
  PassRegistration<LowerQuantForgeOpsPass>();
}

} // namespace mlir::quantforge
```

### Test case

#### File: `test/Transforms/lower_quantforge_ops.mlir`

```mlir
// RUN: quantforge-opt %s -lower-quantforge-ops | FileCheck %s

// CHECK-LABEL: func.func @test_unpack_lowering
func.func @test_unpack_lowering(%packed: tensor<4096x2048xi8>) -> tensor<4096x4096xi8> {
  // CHECK-NOT: qf.unpack
  // CHECK: linalg.generic
  // CHECK: arith.andi
  // CHECK: arith.shrui
  %unpacked = qf.unpack %packed : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
  return %unpacked : tensor<4096x4096xi8>
}

// CHECK-LABEL: func.func @test_dequant_lowering
func.func @test_dequant_lowering(
    %input: tensor<4096x4096xi8>,
    %scale: tensor<4096xf16>,
    %zp: tensor<4096xi8>) -> tensor<4096x4096xf16> {
  // CHECK-NOT: qf.dequant
  // CHECK: linalg.generic
  // CHECK: arith.sitofp
  // CHECK: arith.subf
  // CHECK: arith.mulf
  %output = qf.dequant %input, %scale, %zp : 
      tensor<4096x4096xi8>, tensor<4096xf16>, tensor<4096xi8> -> tensor<4096x4096xf16>
  return %output : tensor<4096x4096xf16>
}
```

### Tiêu chí hoàn thành Phase 2.2
- [ ] Pass compile thành công
- [ ] `quantforge-opt -lower-quantforge-ops` chạy được
- [ ] Test cases pass
- [ ] Không còn `qf.*` ops trong output IR
- [ ] Output chứa `linalg.generic` với `arith.*` operations

---

## Task 2.3: Cập nhật Headers và Registration

### File: `include/QuantForge/Transforms/Passes.h`

```cpp
#ifndef QUANTFORGE_TRANSFORMS_PASSES_H
#define QUANTFORGE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::quantforge {

// === Phase 2 Passes ===
std::unique_ptr<Pass> createConvertLinalgToQuantForgePass();
std::unique_ptr<Pass> createLowerQuantForgeOpsPass();

// === Phase 3 Passes ===
std::unique_ptr<Pass> createTilingPass();
std::unique_ptr<Pass> createVectorizationPass();

// === Phase 4 Passes ===
std::unique_ptr<Pass> createGPUMappingPass();
std::unique_ptr<Pass> createLowerToNVVMPass();

// Registration
void registerConvertLinalgToQuantForgePass();
void registerLowerQuantForgeOpsPass();
void registerTilingPass();
void registerVectorizationPass();
void registerGPUMappingPass();
void registerLowerToNVVMPass();

inline void registerAllPasses() {
  registerConvertLinalgToQuantForgePass();
  registerLowerQuantForgeOpsPass();
  // registerTilingPass();
  // registerVectorizationPass();
  // registerGPUMappingPass();
  // registerLowerToNVVMPass();
}

} // namespace mlir::quantforge

#endif // QUANTFORGE_TRANSFORMS_PASSES_H
```

### File: `lib/Transforms/CMakeLists.txt`

```cmake
add_subdirectory(ConvertLinalgToQuantForge)
add_subdirectory(LowerQuantForgeOps)
# add_subdirectory(Tiling)        # Phase 3
# add_subdirectory(Vectorization)  # Phase 3
# add_subdirectory(GPUMapping)     # Phase 4
# add_subdirectory(LowerToNVVM)    # Phase 4
```

---

# PHASE 3: Tiling & Vectorization

## Tổng quan Pipeline
```
linalg.generic (with arith ops)
         ↓
    TilingPass          →  scf.for loops với tile sizes (128, 128, 64)
         ↓
VectorizationPass       →  vector.transfer_read/write + vector.contract
         ↓
    LoopUnrolling       →  Fully unrolled inner loops
```

---

## Task 3.1: TilingPass

### Mục tiêu
Chia nhỏ ma trận operations thành tiles phù hợp với GPU shared memory.

### Files cần tạo

```
lib/Transforms/Tiling/
├── TilingPass.cpp
└── CMakeLists.txt
```

### Chi tiết implementation

#### File: `lib/Transforms/Tiling/TilingPass.cpp`

```cpp
//===----------------------------------------------------------------------===//
// TilingPass - Tile linalg operations for GPU execution
//
// Target tile sizes:
//   - M dimension: 128 (rows per thread block)
//   - N dimension: 128 (columns per thread block)  
//   - K dimension: 64  (reduction dimension per iteration)
//
// Memory considerations:
//   - Shared memory per SM: ~164KB (Ampere)
//   - Tile A: 128 x 64 x 2 bytes (FP16) = 16KB
//   - Tile B: 64 x 128 x 2 bytes (FP16) = 16KB
//   - Total: 32KB per tile pair (fits comfortably)
//===----------------------------------------------------------------------===//

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct TilingPass : public PassWrapper<TilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass)

  StringRef getArgument() const override { return "quantforge-tiling"; }
  StringRef getDescription() const override {
    return "Tile linalg operations for GPU execution";
  }

  // Configurable tile sizes
  Option<int64_t> tileSizeM{*this, "tile-m", llvm::cl::desc("Tile size for M dimension"),
                            llvm::cl::init(128)};
  Option<int64_t> tileSizeN{*this, "tile-n", llvm::cl::desc("Tile size for N dimension"),
                            llvm::cl::init(128)};
  Option<int64_t> tileSizeK{*this, "tile-k", llvm::cl::desc("Tile size for K dimension"),
                            llvm::cl::init(64)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Tile sizes for [M, N, K]
    SmallVector<int64_t> tileSizes = {tileSizeM, tileSizeN, tileSizeK};

    // Find and tile all linalg.matmul operations
    SmallVector<linalg::LinalgOp> matmulOps;
    funcOp.walk([&](linalg::MatmulOp op) {
      matmulOps.push_back(op);
    });

    // Also handle linalg.generic that came from lowered dequant
    funcOp.walk([&](linalg::GenericOp op) {
      matmulOps.push_back(op);
    });

    IRRewriter rewriter(ctx);
    
    for (auto linalgOp : matmulOps) {
      // Use SCF tiling interface
      scf::SCFTilingOptions tilingOptions;
      tilingOptions.setTileSizes(tileSizes);
      
      FailureOr<scf::SCFTilingResult> tilingResult = 
          scf::tileUsingSCF(rewriter, 
                            cast<TilingInterface>(linalgOp.getOperation()),
                            tilingOptions);
      
      if (succeeded(tilingResult)) {
        rewriter.replaceOp(linalgOp, tilingResult->replacements);
      }
    }
  }
};

} // namespace

namespace mlir::quantforge {

std::unique_ptr<Pass> createTilingPass() {
  return std::make_unique<TilingPass>();
}

void registerTilingPass() {
  PassRegistration<TilingPass>();
}

} // namespace mlir::quantforge
```

### Test case

#### File: `test/Transforms/tiling.mlir`

```mlir
// RUN: quantforge-opt %s -quantforge-tiling="tile-m=128 tile-n=128 tile-k=64" | FileCheck %s

// CHECK-LABEL: func.func @test_tiled_matmul
func.func @test_tiled_matmul(
    %A: tensor<4096x4096xf16>,
    %B: tensor<4096x4096xf16>,
    %C: tensor<4096x4096xf16>) -> tensor<4096x4096xf16> {
  
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       linalg.matmul
  %result = linalg.matmul ins(%A, %B : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
                          outs(%C : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
  return %result : tensor<4096x4096xf16>
}
```

### Tiêu chí hoàn thành Task 3.1
- [ ] Pass compile thành công
- [ ] Output IR có 3 nested `scf.for` loops
- [ ] Inner matmul operates on 128x128x64 tiles
- [ ] Tile sizes configurable qua command line

---

## Task 3.2: VectorizationPass

### Mục tiêu
Vectorize các scalar operations thành vector operations cho SIMD execution.

### Files cần tạo

```
lib/Transforms/Vectorization/
├── VectorizationPass.cpp
└── CMakeLists.txt
```

### Chi tiết implementation

#### File: `lib/Transforms/Vectorization/VectorizationPass.cpp`

```cpp
//===----------------------------------------------------------------------===//
// VectorizationPass - Vectorize tiled linalg ops
//
// Strategy:
// 1. Vectorize linalg.generic → vector operations
// 2. Vectorize linalg.matmul → vector.contract
// 3. Convert tensor reads/writes → vector.transfer_read/write
//
// Target vector sizes:
//   - Native GPU vector width: 4 (for FP16, maps to half4)
//   - Warp-level: 32 threads × 4 elements = 128 elements per warp
//===----------------------------------------------------------------------===//

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct VectorizationPass 
    : public PassWrapper<VectorizationPass, OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizationPass)

  StringRef getArgument() const override { return "quantforge-vectorization"; }
  StringRef getDescription() const override {
    return "Vectorize linalg operations for GPU execution";
  }

  Option<int64_t> vectorSize{*this, "vector-size", 
                              llvm::cl::desc("Vector width"),
                              llvm::cl::init(4)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Step 1: Vectorize linalg operations
    RewritePatternSet patterns(ctx);
    
    // Add linalg vectorization patterns
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    
    // Configure vectorization options
    vector::VectorTransformsOptions vectorTransformOptions;
    vectorTransformOptions.setVectorTransformsOptions(
        vector::VectorContractLowering::OuterProduct);
    
    // Apply vectorization
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Step 2: Vectorize remaining linalg.generic ops
    funcOp.walk([&](linalg::GenericOp genericOp) {
      // Check if vectorizable
      if (!linalg::isaContractionOpInterface(genericOp))
        return;
      
      IRRewriter rewriter(ctx);
      rewriter.setInsertionPoint(genericOp);
      
      // Use linalg vectorization
      FailureOr<Operation *> vectorizedOp = 
          linalg::vectorize(rewriter, genericOp);
      
      if (succeeded(vectorizedOp)) {
        rewriter.eraseOp(genericOp);
      }
    });
  }
};

} // namespace

namespace mlir::quantforge {

std::unique_ptr<Pass> createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

void registerVectorizationPass() {
  PassRegistration<VectorizationPass>();
}

} // namespace mlir::quantforge
```

### Test case

#### File: `test/Transforms/vectorization.mlir`

```mlir
// RUN: quantforge-opt %s -quantforge-vectorization | FileCheck %s

// CHECK-LABEL: func.func @test_vectorized_matmul
func.func @test_vectorized_matmul(
    %A: tensor<128x64xf16>,
    %B: tensor<64x128xf16>,
    %C: tensor<128x128xf16>) -> tensor<128x128xf16> {
  
  // CHECK: vector.transfer_read
  // CHECK: vector.contract
  // CHECK: vector.transfer_write
  %result = linalg.matmul ins(%A, %B : tensor<128x64xf16>, tensor<64x128xf16>)
                          outs(%C : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %result : tensor<128x128xf16>
}
```

### Tiêu chí hoàn thành Task 3.2
- [ ] Pass compile thành công
- [ ] Scalar arith ops → vector arith ops
- [ ] linalg.matmul → vector.contract
- [ ] tensor reads/writes → vector.transfer_read/write

---

## Task 3.3: Lower Vector to SCF

### Mục tiêu
Lower vector operations xuống scalar loops (chuẩn bị cho GPU mapping).

### Sử dụng built-in passes

```bash
quantforge-opt input.mlir \
  -convert-vector-to-scf \
  -convert-scf-to-cf \
  -lower-affine
```

### Tiêu chí hoàn thành Task 3.3
- [ ] vector.contract → scf.for loops với arith ops
- [ ] Không còn vector dialect operations
- [ ] Output IR ready cho GPU mapping

---

# PHASE 4: GPU Lowering & PTX Generation

## Tổng quan Pipeline
```
scf.for loops + arith ops
         ↓
    GPUMappingPass      →  gpu.launch + gpu.thread_id/block_id
         ↓
SharedMemoryBuffering   →  gpu.alloc_shared + memref.load/store
         ↓
    LowerToNVVM         →  nvvm.* intrinsics + llvm.* ops
         ↓
    PTX Backend         →  .ptx assembly file
```

---

## Task 4.1: GPUMappingPass

### Mục tiêu
Map parallel loops to GPU grid/block/thread hierarchy.

### Files cần tạo

```
lib/Transforms/GPUMapping/
├── GPUMappingPass.cpp
└── CMakeLists.txt
```

### Chi tiết implementation

#### File: `lib/Transforms/GPUMapping/GPUMappingPass.cpp`

```cpp
//===----------------------------------------------------------------------===//
// GPUMappingPass - Map loops to GPU execution model
//
// Mapping Strategy:
//   - Outermost 2 loops → grid dimensions (blockIdx.x, blockIdx.y)
//   - Next 2 loops → block dimensions (threadIdx.x, threadIdx.y)
//   - Inner loops → sequential execution per thread
//
// Thread Block Size: 16 × 16 = 256 threads
// Grid Size: (N/128) × (M/128) blocks
//===----------------------------------------------------------------------===//

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct GPUMappingPass 
    : public PassWrapper<GPUMappingPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUMappingPass)

  StringRef getArgument() const override { return "quantforge-gpu-mapping"; }
  StringRef getDescription() const override {
    return "Map SCF loops to GPU execution model";
  }

  Option<int64_t> blockSizeX{*this, "block-x", 
                              llvm::cl::desc("Thread block size X"),
                              llvm::cl::init(16)};
  Option<int64_t> blockSizeY{*this, "block-y",
                              llvm::cl::desc("Thread block size Y"),
                              llvm::cl::init(16)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Step 1: Create GPU module
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    auto gpuModule = builder.create<gpu::GPUModuleOp>(
        moduleOp.getLoc(), "quantforge_kernels");

    // Step 2: Find and convert functions with parallel loops
    SmallVector<func::FuncOp> funcsToConvert;
    moduleOp.walk([&](func::FuncOp funcOp) {
      // Check if function has tiled loops suitable for GPU
      bool hasTiledLoops = false;
      funcOp.walk([&](scf::ForOp) {
        hasTiledLoops = true;
      });
      if (hasTiledLoops)
        funcsToConvert.push_back(funcOp);
    });

    for (auto funcOp : funcsToConvert) {
      convertFunctionToGPUKernel(funcOp, gpuModule, builder);
    }
  }

private:
  void convertFunctionToGPUKernel(func::FuncOp funcOp, 
                                   gpu::GPUModuleOp gpuModule,
                                   OpBuilder &builder) {
    Location loc = funcOp.getLoc();
    MLIRContext *ctx = builder.getContext();

    // Create GPU function
    builder.setInsertionPointToEnd(gpuModule.getBody());
    
    auto gpuFunc = builder.create<gpu::GPUFuncOp>(
        loc, funcOp.getName(), funcOp.getFunctionType());
    gpuFunc.setKnownBlockSizeAttr(
        builder.getDenseI32ArrayAttr({(int32_t)blockSizeX, 
                                       (int32_t)blockSizeY, 1}));

    // Clone function body into GPU function
    IRMapping mapping;
    funcOp.getBody().cloneInto(&gpuFunc.getBody(), mapping);

    // Replace scf.for loops with GPU execution model
    gpuFunc.walk([&](scf::ForOp forOp) {
      // Map to appropriate GPU dimension based on nesting level
      // This is simplified - real implementation needs more sophistication
    });

    // Add gpu.terminator
    builder.setInsertionPointToEnd(&gpuFunc.getBody().back());
    builder.create<gpu::ReturnOp>(loc);
  }
};

} // namespace

namespace mlir::quantforge {

std::unique_ptr<Pass> createGPUMappingPass() {
  return std::make_unique<GPUMappingPass>();
}

void registerGPUMappingPass() {
  PassRegistration<GPUMappingPass>();
}

} // namespace mlir::quantforge
```

### Test case

#### File: `test/Transforms/gpu_mapping.mlir`

```mlir
// RUN: quantforge-opt %s -quantforge-gpu-mapping | FileCheck %s

// CHECK: gpu.module @quantforge_kernels
// CHECK: gpu.func @gemv_kernel

module {
  func.func @gemv_kernel(%A: memref<4096x4096xf16>, 
                         %x: memref<4096xf16>,
                         %y: memref<4096xf16>) {
    // Tiled loops that should be mapped to GPU
    scf.for %i = %c0 to %c4096 step %c128 {
      scf.for %j = %c0 to %c4096 step %c128 {
        // Inner computation
      }
    }
    return
  }
}
```

### Tiêu chí hoàn thành Task 4.1
- [ ] Pass compile thành công
- [ ] Output có `gpu.module` và `gpu.func`
- [ ] Loops mapped to `blockIdx` và `threadIdx`
- [ ] `gpu.launch` wraps kernel invocation

---

## Task 4.2: Shared Memory Buffering

### Mục tiêu
Insert explicit shared memory allocations và data staging.

### Implementation outline

```cpp
// Detect tiles that should be staged in shared memory
// Insert:
//   %smem = gpu.alloc_shared() : memref<128x64xf16, #gpu.address_space<workgroup>>
//   
// Copy from global to shared:
//   gpu.memcpy async %smem, %global_slice
//   gpu.barrier
//
// Use shared memory in computation:
//   %val = memref.load %smem[%i, %j]
```

---

## Task 4.3: LowerToNVVM Pass

### Mục tiêu
Lower GPU dialect → NVVM dialect → LLVM IR → PTX.

### Pipeline command

```bash
quantforge-opt input.mlir \
  -gpu-kernel-outlining \
  -convert-scf-to-cf \
  -convert-gpu-to-nvvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -gpu-to-cubin="chip=sm_80"
```

### Custom pass for PTX intrinsics

#### File: `lib/Transforms/LowerToNVVM/LowerToNVVM.cpp`

```cpp
//===----------------------------------------------------------------------===//
// LowerToNVVM - Insert explicit PTX intrinsics for INT4 unpacking
//
// Key transformations:
// 1. arith.andi → nvvm.bitwise.and
// 2. arith.shrui → nvvm.shr.u32
// 3. Insert prmt for byte permutation (optimization)
// 4. Handle bank conflict avoidance patterns
//===----------------------------------------------------------------------===//

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct LowerToNVVMPass 
    : public PassWrapper<LowerToNVVMPass, OperationPass<gpu::GPUModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToNVVMPass)

  StringRef getArgument() const override { return "quantforge-lower-to-nvvm"; }
  StringRef getDescription() const override {
    return "Lower GPU operations to NVVM dialect with INT4 optimizations";
  }

  Option<std::string> targetChip{*this, "chip",
                                  llvm::cl::desc("Target GPU architecture"),
                                  llvm::cl::init("sm_80")};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp gpuModule = getOperation();
    
    // Apply standard GPU to NVVM conversion
    // Then apply custom INT4 optimizations
    
    gpuModule.walk([&](arith::AndIOp andOp) {
      // Check if this is part of INT4 unpacking pattern
      // If so, potentially optimize with prmt instruction
    });
  }
};

} // namespace

namespace mlir::quantforge {

std::unique_ptr<Pass> createLowerToNVVMPass() {
  return std::make_unique<LowerToNVVMPass>();
}

void registerLowerToNVVMPass() {
  PassRegistration<LowerToNVVMPass>();
}

} // namespace mlir::quantforge
```

### Tiêu chí hoàn thành Task 4.3
- [ ] Generate valid PTX assembly
- [ ] PTX có `shr.u32`, `and.b32` cho unpacking
- [ ] Kernel loads data với coalesced pattern
- [ ] No register spilling (check với `ptxas -v`)

---

## Task 4.4: PTX Compilation Pipeline

### Mục tiêu
End-to-end compilation từ MLIR đến CUBIN.

### Script: `scripts/compile_kernel.sh`

```bash
#!/bin/bash

INPUT=$1
OUTPUT=${2:-kernel.cubin}
CHIP=${3:-sm_80}

# Full MLIR to PTX pipeline
quantforge-opt $INPUT \
  -convert-linalg-to-quantforge \
  -lower-quantforge-ops \
  -quantforge-tiling="tile-m=128 tile-n=128 tile-k=64" \
  -quantforge-vectorization \
  -convert-vector-to-scf \
  -quantforge-gpu-mapping="block-x=16 block-y=16" \
  -gpu-kernel-outlining \
  -convert-scf-to-cf \
  -convert-gpu-to-nvvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir \
  | llc -mcpu=$CHIP -mtriple=nvptx64-nvidia-cuda \
  > kernel.ptx

# Compile PTX to CUBIN
ptxas -arch=$CHIP -o $OUTPUT kernel.ptx

echo "Generated: $OUTPUT"
```

### Tiêu chí hoàn thành Task 4.4
- [ ] Script chạy thành công end-to-end
- [ ] Generate valid .cubin file
- [ ] Can load với cuModuleLoad()

---

# PHASE 5: Verification & Profiling

## Task 5.1: Python Runtime

### Mục tiêu
Load và execute compiled kernel từ Python.

### Files cần tạo

```
python/quantforge/
├── compiler.py     # MLIR compilation wrapper
├── runtime.py      # CUDA kernel loading và execution
└── verify.py       # Correctness verification
```

### Implementation

#### File: `python/quantforge/compiler.py`

```python
"""
QuantForge Compiler - MLIR to PTX compilation
"""

import subprocess
import tempfile
from pathlib import Path


def compile_mlir_to_ptx(mlir_source: str, 
                        output_path: str = None,
                        tile_m: int = 128,
                        tile_n: int = 128,
                        tile_k: int = 64,
                        chip: str = "sm_80") -> str:
    """
    Compile MLIR source to PTX assembly.
    
    Args:
        mlir_source: MLIR IR as string or path to .mlir file
        output_path: Output PTX file path
        tile_m, tile_n, tile_k: Tiling parameters
        chip: Target GPU architecture
    
    Returns:
        Path to generated PTX file
    """
    # Write source to temp file if it's a string
    if not Path(mlir_source).exists():
        with tempfile.NamedTemporaryFile(suffix='.mlir', delete=False, mode='w') as f:
            f.write(mlir_source)
            input_path = f.name
    else:
        input_path = mlir_source
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.ptx')
    
    # Build pipeline command
    pipeline = [
        f'-convert-linalg-to-quantforge',
        f'-lower-quantforge-ops',
        f'-quantforge-tiling="tile-m={tile_m} tile-n={tile_n} tile-k={tile_k}"',
        f'-quantforge-vectorization',
        f'-convert-vector-to-scf',
        f'-quantforge-gpu-mapping',
        f'-gpu-kernel-outlining',
        f'-convert-scf-to-cf',
        f'-convert-gpu-to-nvvm',
        f'-convert-arith-to-llvm',
        f'-convert-func-to-llvm',
        f'-reconcile-unrealized-casts',
    ]
    
    cmd = f"quantforge-opt {input_path} {' '.join(pipeline)}"
    
    # Run MLIR pipeline
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MLIR compilation failed:\n{result.stderr}")
    
    # Translate to LLVM IR and compile to PTX
    llvm_ir = subprocess.run(
        "mlir-translate --mlir-to-llvmir",
        shell=True, input=result.stdout, capture_output=True, text=True
    )
    
    ptx = subprocess.run(
        f"llc -mcpu={chip} -mtriple=nvptx64-nvidia-cuda",
        shell=True, input=llvm_ir.stdout, capture_output=True, text=True
    )
    
    with open(output_path, 'w') as f:
        f.write(ptx.stdout)
    
    return output_path


def compile_mlir_to_cubin(mlir_source: str,
                          output_path: str = None,
                          chip: str = "sm_80") -> str:
    """
    Compile MLIR source to CUBIN.
    """
    ptx_path = compile_mlir_to_ptx(mlir_source, chip=chip)
    
    if output_path is None:
        output_path = ptx_path.replace('.ptx', '.cubin')
    
    subprocess.run(
        f"ptxas -arch={chip} -o {output_path} {ptx_path}",
        shell=True, check=True
    )
    
    return output_path
```

#### File: `python/quantforge/runtime.py`

```python
"""
QuantForge Runtime - Load and execute compiled kernels
"""

import ctypes
import numpy as np
import torch
from cuda import cuda, cudart


class QuantForgeKernel:
    """
    Wrapper for compiled QuantForge kernel.
    """
    
    def __init__(self, cubin_path: str, kernel_name: str = "gemv_kernel"):
        """
        Load kernel from CUBIN file.
        
        Args:
            cubin_path: Path to .cubin file
            kernel_name: Name of kernel function
        """
        self.cubin_path = cubin_path
        self.kernel_name = kernel_name
        
        # Initialize CUDA
        err, = cuda.cuInit(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        # Load module
        err, self.module = cuda.cuModuleLoad(cubin_path.encode())
        assert err == cuda.CUresult.CUDA_SUCCESS, f"Failed to load module: {err}"
        
        # Get kernel function
        err, self.kernel = cuda.cuModuleGetFunction(
            self.module, kernel_name.encode())
        assert err == cuda.CUresult.CUDA_SUCCESS, f"Failed to get function: {err}"
    
    def __call__(self, 
                 x: torch.Tensor,
                 w_packed: torch.Tensor,
                 scale: torch.Tensor,
                 zero_point: torch.Tensor,
                 block_dim: tuple = (16, 16, 1),
                 grid_dim: tuple = None) -> torch.Tensor:
        """
        Execute kernel.
        
        Args:
            x: Activation tensor (1, K) FP16
            w_packed: Packed INT4 weights (N, K//2) INT8
            scale: Scale factors (N,) FP16
            zero_point: Zero points (N,) INT8
            block_dim: Thread block dimensions
            grid_dim: Grid dimensions (auto-calculated if None)
        
        Returns:
            Output tensor (1, N) FP16
        """
        N = w_packed.shape[0]
        K = w_packed.shape[1] * 2  # Unpacked dimension
        
        # Allocate output
        y = torch.empty(1, N, dtype=torch.float16, device=x.device)
        
        # Calculate grid dimensions
        if grid_dim is None:
            grid_dim = (
                (N + 127) // 128,
                1,
                1
            )
        
        # Get device pointers
        args = [
            x.data_ptr(),
            w_packed.data_ptr(),
            scale.data_ptr(),
            zero_point.data_ptr(),
            y.data_ptr(),
            N, K
        ]
        
        # Pack arguments
        arg_types = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
        kernel_args = (ctypes.c_void_p * len(args))()
        for i, (arg, arg_type) in enumerate(zip(args, arg_types)):
            kernel_args[i] = ctypes.cast(
                ctypes.pointer(arg_type(arg)), 
                ctypes.c_void_p
            )
        
        # Launch kernel
        err, = cuda.cuLaunchKernel(
            self.kernel,
            grid_dim[0], grid_dim[1], grid_dim[2],
            block_dim[0], block_dim[1], block_dim[2],
            0,  # Shared memory
            0,  # Stream
            kernel_args,
            0   # Extra
        )
        assert err == cuda.CUresult.CUDA_SUCCESS, f"Kernel launch failed: {err}"
        
        # Synchronize
        err, = cuda.cuCtxSynchronize()
        assert err == cuda.CUresult.CUDA_SUCCESS
        
        return y
    
    def __del__(self):
        if hasattr(self, 'module'):
            cuda.cuModuleUnload(self.module)
```

#### File: `python/quantforge/verify.py`

```python
"""
QuantForge Verification - Compare kernel output with golden reference
"""

import torch
import numpy as np
from typing import Dict, Tuple

from .baseline import gemv_int4_baseline


def load_golden_tensors(path: str = "test/golden_tensors.pt") -> Dict[str, torch.Tensor]:
    """Load golden tensors from Phase 1."""
    return torch.load(path)


def verify_correctness(
    kernel_output: torch.Tensor,
    golden: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Tuple[bool, Dict[str, float]]:
    """
    Verify kernel output against golden reference.
    
    Args:
        kernel_output: Output from compiled kernel
        golden: Golden reference from PyTorch baseline
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Tuple of (passed, metrics_dict)
    """
    # Ensure same device
    kernel_output = kernel_output.cpu().float()
    golden = golden.cpu().float()
    
    # Compute metrics
    metrics = {}
    
    # 1. Cosine similarity (target: > 0.99)
    cos_sim = torch.nn.functional.cosine_similarity(
        kernel_output.flatten().unsqueeze(0),
        golden.flatten().unsqueeze(0)
    ).item()
    metrics['cosine_similarity'] = cos_sim
    
    # 2. Max absolute error
    max_abs_err = (kernel_output - golden).abs().max().item()
    metrics['max_abs_error'] = max_abs_err
    
    # 3. Mean absolute error
    mean_abs_err = (kernel_output - golden).abs().mean().item()
    metrics['mean_abs_error'] = mean_abs_err
    
    # 4. Relative error
    rel_err = ((kernel_output - golden).abs() / (golden.abs() + 1e-8)).mean().item()
    metrics['mean_rel_error'] = rel_err
    
    # 5. Element-wise match
    allclose = torch.allclose(kernel_output, golden, rtol=rtol, atol=atol)
    metrics['allclose'] = allclose
    
    # Pass criteria
    passed = cos_sim > 0.99 and allclose
    
    return passed, metrics


def run_verification(
    kernel,
    golden_path: str = "test/golden_tensors.pt"
) -> Tuple[bool, Dict[str, float]]:
    """
    Run full verification suite.
    """
    # Load golden data
    data = load_golden_tensors(golden_path)
    x = data['x'].cuda()
    w_packed = data['w_packed'].cuda()
    scale = data['scale'].cuda()
    zero_point = data['zero_point'].cuda()
    golden = data['golden']
    
    # Run kernel
    output = kernel(x, w_packed, scale, zero_point)
    
    # Verify
    passed, metrics = verify_correctness(output, golden)
    
    print("=== Verification Results ===")
    print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f} (target: > 0.99)")
    print(f"Max Absolute Error: {metrics['max_abs_error']:.6e}")
    print(f"Mean Absolute Error: {metrics['mean_abs_error']:.6e}")
    print(f"Mean Relative Error: {metrics['mean_rel_error']:.6e}")
    print(f"torch.allclose: {metrics['allclose']}")
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    
    return passed, metrics
```

---

## Task 5.2: Benchmark Script

### File: `python/quantforge/benchmark.py`

```python
"""
QuantForge Benchmark - Compare kernel performance
"""

import torch
import time
from typing import Callable, Dict

from .baseline import gemv_int4_baseline, benchmark as torch_benchmark


def benchmark_kernel(
    kernel: Callable,
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    warmup: int = 10,
    repeat: int = 100
) -> float:
    """
    Benchmark compiled kernel.
    
    Returns:
        Average latency in seconds
    """
    # Warmup
    for _ in range(warmup):
        kernel(x, w_packed, scale, zero_point)
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.time()
    for _ in range(repeat):
        kernel(x, w_packed, scale, zero_point)
    torch.cuda.synchronize()
    
    return (time.time() - start) / repeat


def run_benchmark(
    kernel: Callable,
    M: int = 1,
    K: int = 4096,
    N: int = 4096
) -> Dict[str, float]:
    """
    Run full benchmark suite comparing baseline vs compiled kernel.
    """
    device = torch.device("cuda")
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w_full = torch.randint(0, 16, (N, K), dtype=torch.int8, device=device)
    
    # Pack weights
    low = w_full[..., 0::2] & 0x0F
    high = w_full[..., 1::2] & 0x0F
    w_packed = (low | (high << 4)).to(torch.int8)
    
    scale = torch.randn(N, dtype=torch.float16, device=device) * 0.01
    zero_point = torch.randint(0, 16, (N,), dtype=torch.int8, device=device)
    
    # Benchmark baseline
    baseline_latency = torch_benchmark(
        gemv_int4_baseline, x, w_packed, scale, zero_point
    )
    
    # Benchmark compiled kernel
    kernel_latency = benchmark_kernel(
        kernel, x, w_packed, scale, zero_point
    )
    
    # Calculate metrics
    speedup = baseline_latency / kernel_latency
    
    # Memory bandwidth
    w_bytes = w_packed.numel() * 1  # INT8
    x_bytes = x.numel() * 2         # FP16
    total_bytes = w_bytes + x_bytes
    
    baseline_bw = total_bytes / baseline_latency / 1e9
    kernel_bw = total_bytes / kernel_latency / 1e9
    
    results = {
        'baseline_latency_ms': baseline_latency * 1000,
        'kernel_latency_ms': kernel_latency * 1000,
        'speedup': speedup,
        'baseline_bandwidth_GBps': baseline_bw,
        'kernel_bandwidth_GBps': kernel_bw,
    }
    
    print("=== Benchmark Results ===")
    print(f"Baseline Latency: {results['baseline_latency_ms']:.3f} ms")
    print(f"Kernel Latency:   {results['kernel_latency_ms']:.3f} ms")
    print(f"Speedup:          {results['speedup']:.2f}x")
    print(f"\nBaseline Bandwidth: {results['baseline_bandwidth_GBps']:.1f} GB/s")
    print(f"Kernel Bandwidth:   {results['kernel_bandwidth_GBps']:.1f} GB/s")
    
    # Check if meets target
    if speedup >= 1.5:
        print(f"\n✅ Target speedup (1.5x) achieved!")
    else:
        print(f"\n⚠️ Below target speedup (1.5x)")
    
    return results
```

---

## Task 5.3: Nsight Compute Profiling

### Script: `scripts/profile_kernel.sh`

```bash
#!/bin/bash

KERNEL_PATH=${1:-build/kernel.cubin}
KERNEL_NAME=${2:-gemv_kernel}

# Run Nsight Compute profiling
ncu --set full \
    --target-processes all \
    --export profile_report \
    --force-overwrite \
    python -c "
from quantforge.runtime import QuantForgeKernel
from quantforge.verify import load_golden_tensors

kernel = QuantForgeKernel('$KERNEL_PATH', '$KERNEL_NAME')
data = load_golden_tensors()
kernel(data['x'].cuda(), data['w_packed'].cuda(), 
       data['scale'].cuda(), data['zero_point'].cuda())
"

# Print key metrics
echo "=== Key Metrics ==="
ncu --import profile_report.ncu-rep --page details --csv | grep -E "Memory Throughput|Bank Conflicts|Register"
```

### Metrics to check

| Metric | Target | Command |
|--------|--------|---------|
| Memory Throughput | > 75% peak | `gpu__compute_memory_throughput` |
| L1 Bank Conflicts | < 5% | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` |
| Register Spilling | 0 bytes | `sm__sass_spill_bytes` |
| Occupancy | > 50% | `sm__warps_active.avg.pct_of_peak_sustained_active` |

---

## Task 5.4: End-to-End Test Script

### File: `test/run_e2e_test.py`

```python
#!/usr/bin/env python3
"""
QuantForge End-to-End Test

Usage:
    python test/run_e2e_test.py
"""

import sys
sys.path.insert(0, 'python')

from quantforge.compiler import compile_mlir_to_cubin
from quantforge.runtime import QuantForgeKernel
from quantforge.verify import run_verification
from quantforge.benchmark import run_benchmark


def main():
    print("=" * 60)
    print("QuantForge End-to-End Test")
    print("=" * 60)
    
    # Step 1: Compile MLIR to CUBIN
    print("\n[1/3] Compiling MLIR to CUBIN...")
    mlir_source = "test/Dialect/QuantForge/gemv_quantized.mlir"
    cubin_path = compile_mlir_to_cubin(mlir_source, chip="sm_80")
    print(f"   Generated: {cubin_path}")
    
    # Step 2: Load kernel
    print("\n[2/3] Loading kernel...")
    kernel = QuantForgeKernel(cubin_path, "gemv_kernel")
    print("   Kernel loaded successfully")
    
    # Step 3: Verify correctness
    print("\n[3/3] Verifying correctness...")
    passed, metrics = run_verification(kernel)
    
    if not passed:
        print("\n❌ Correctness verification FAILED")
        return 1
    
    # Step 4: Benchmark
    print("\n[Bonus] Running benchmark...")
    results = run_benchmark(kernel)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Correctness: PASSED (cosine_similarity = {metrics['cosine_similarity']:.6f})")
    print(f"✅ Speedup: {results['speedup']:.2f}x vs PyTorch baseline")
    print(f"✅ Bandwidth: {results['kernel_bandwidth_GBps']:.1f} GB/s")
    
    # Final verdict
    success = (
        passed and 
        results['speedup'] >= 1.5 and 
        results['kernel_bandwidth_GBps'] > 100
    )
    
    if success:
        print("\n🎉 ALL TARGETS MET - PROJECT COMPLETE!")
        return 0
    else:
        print("\n⚠️ Some targets not met")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

# CHECKLIST TỔNG HỢP

## Phase 2: Dialect & Passes
- [ ] Task 2.1: ConvertLinalgToQuantForge Pass
  - [ ] Pattern matcher cho quantized matmul
  - [ ] Insert qf.unpack và qf.dequant
  - [ ] Test với FileCheck
- [ ] Task 2.2: LowerQuantForgeOps Pass
  - [ ] Lower qf.unpack → arith ops
  - [ ] Lower qf.dequant → arith ops
  - [ ] Test với FileCheck
- [ ] Task 2.3: Update headers và registration

## Phase 3: Tiling & Vectorization
- [ ] Task 3.1: TilingPass
  - [ ] Tile sizes configurable (128, 128, 64)
  - [ ] Generate scf.for loops
- [ ] Task 3.2: VectorizationPass
  - [ ] Vectorize arith ops
  - [ ] Generate vector.contract
- [ ] Task 3.3: Lower Vector to SCF

## Phase 4: GPU & PTX
- [ ] Task 4.1: GPUMappingPass
  - [ ] Map to blockIdx/threadIdx
  - [ ] Generate gpu.launch
- [ ] Task 4.2: Shared Memory Buffering
- [ ] Task 4.3: LowerToNVVM Pass
- [ ] Task 4.4: PTX Compilation Pipeline

## Phase 5: Verification
- [ ] Task 5.1: Python Runtime
  - [ ] compiler.py
  - [ ] runtime.py
  - [ ] verify.py
- [ ] Task 5.2: Benchmark Script
- [ ] Task 5.3: Nsight Compute Profiling
- [ ] Task 5.4: End-to-End Test

---

# SUCCESS CRITERIA

| # | Tiêu chí | Mục tiêu |
|---|----------|----------|
| 1 | Correctness | cosine_similarity > 0.99 |
| 2 | Self-containment | No cuBLAS/cuDNN calls |
| 3 | Latency Speedup | 1.5-2x vs baseline |
| 4 | Bandwidth | > 75% theoretical peak |
| 5 | Hardware Cost | Bank conflicts < 5%, Spill = 0 |
