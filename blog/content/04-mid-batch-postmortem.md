---
title: Why are batches of 32 to 256 slower than Marlin? A post-mortem with a hand-written CUDA prototype
date: 2026-11-20
order: 4
summary: Profiling, ablations and a hand-written Marlin-style kernel to find out what limits the generated kernels at medium batch size. Most hypotheses were wrong, and the negative results are the useful part.
---

The kernel table in the [overview](01-overview.html) has one ugly region. At batch size 1 to 16 the generated kernels match Marlin; at 512 rows and more they reach 85 percent; but at **32, 64, 128 and 256 rows they are 87, 67, 69 and 76 percent of Marlin**. This regime matters in practice: it is what a server sees with a few concurrent requests or short prompts. This post is the investigation. The conclusion is not "here is the fix"; it is a list of things that are *not* the problem, with measurements.

| M | Marlin (us) | QuantForge (us) | ratio |
|---|--:|--:|--:|
| 32 | 50.9 | 58.8 | 0.87 |
| 64 | 58.3 | 87.4 | 0.67 |
| 128 | 105.1 | 152.2 | 0.69 |
| 256 | 201.0 | 263.0 | 0.76 |

(K = N = 4096, group 128, weights from DRAM.) At M = 64 the arithmetic is 2.1 GFLOP, about 47 microseconds at the 45 TFLOP/s cuBLAS reaches on this card; Marlin does it in 59.

## Hypothesis 1: the schedule is wrong

For M between 17 and 64 the first version used a 64-row tile intended for prefill. A sweep over block tile M (16, 32, 64, 128), warp tile, K tile, stages and look-ahead (up to about 190 configurations per M) found better ones: for 32 rows a 32-row block tile with a 128-deep K tile (**68.6 to 58.6 microseconds**), for 64 rows a 64-row tile with a 32 x 64 warp tile and one-tile look-ahead (**97 to 87 microseconds**). Both are now the defaults, and the autotuner searches the same space. That moved M = 32 from 0.75x to 0.87x and M = 64 from 0.61x to 0.67x of Marlin. Then the sweeps plateaued: at M = 64, the best eight configurations lie between 87 and 94 microseconds. Tiling is exhausted.

## Hypothesis 2: not enough parallelism

The grid for M = 64 has only 64 blocks on 20 SMs. If starvation were the problem, more blocks should help. They do not: a 64 x 1024 x 16384 problem has exactly the same number of weights as 64 x 4096 x 4096 but four times as many blocks (256 instead of 64, each with a quarter of the K loop), and it takes 110 microseconds against 102. Time follows the work per SM, not the number of idle SMs. Smaller 16-row block tiles (four times the blocks, but four times the weight traffic, since each block re-reads the weights) are 30 percent *slower*.

## Hypothesis 3: DRAM

`ncu` at M = 64: DRAM throughput **52 percent** of peak. Not the limit.

## Hypothesis 4: the Tensor Cores are saturated

Tensor pipe active: **51 percent**, issue slots busy: **36 percent**. Neither is saturated. The stall reasons are spread almost evenly, each costing 1.0 to 1.4 cycles per issued instruction: shared-memory queue pressure (`mio_throttle` 1.38), the Tensor pipe being momentarily busy (`math_pipe_throttle` 1.23), fixed-latency dependences (`wait` 1.34), global-load latency (`long_scoreboard` 1.20), barriers (0.96). No single culprit, which is the signature of a kernel whose phases do not overlap well.

## Hypothesis 5: the integer work of the dequantisation

The instruction histogram of the M = 64 kernel's main loop shows what looks like the obvious suspect:

| opcode | warp instructions | share |
|---|--:|--:|
| LOP3 | 697 K | 21 % |
| HMMA (the useful work) | 524 K | 16 % |
| SHF | 345 K | 10 % |
| LDSM (`ldmatrix`) | 328 K | 10 % |
| HMUL2 + HADD2 | 524 K | 16 % |
| IMAD, SGXT, others | the rest | |

About 60 percent of the executed instructions are dequantisation; only 16 percent are Tensor Core instructions. Marlin avoids most of the unpacking arithmetic by pre-arranging the nibbles offline: pair nibble `p` with nibble `p + 4` of a 32-bit word, so that one shift and one `LOP3` produce a ready `f16x2` pair (7 instead of about 16 integer operations per 8 weights).

Before redesigning the weight layout I **ablated it**: emit the cheap unpacking with the wrong (but timed) layout and measure. Result: **3 percent at M = 64, 2.6 percent at M = 512, 1.2 percent at 2048 x 4096 x 4096**. The kernels are not limited by the integer arithmetic even though it dominates the instruction count, because the issue slots are only 36 percent busy. A weight-layout change across the whole system would have bought 3 percent. Abandoned.

## Hypothesis 6: B should stay in registers like in Marlin

The structural difference with Marlin is where the dequantised weights live. The generated kernels stage the dequantised B tile through shared memory: dequantise, `st.shared`, `__syncthreads`, `ldmatrix`, `mma`. Marlin repacks the weights offline so that each thread loads 16 bytes that are exactly its B fragments of eight 8-column tiles, dequantises them in registers and feeds `mma` directly: no shared memory and no barrier for B. Implementing that inside the MLIR pipeline would be a multi-week project, so I first built the **upper bound by hand**.

The prototype is a single CUDA file (`benchmarks/proto/bir_w4a16.cu` in the repository, about 300 lines):

* offline repack: per `k16` step, per 64-column strip, each lane gets one `uint4` holding 32 nibbles, ordered so that `(w >> 4t) & 0x000F000F` is directly a half2 B register of the `m16n8k16` instruction (the Marlin nibble pairing);
* dequantisation: `lop3`, `sub.f16x2`, `mul.f16x2` straight into the `mma` operands;
* A staged through shared memory with `cp.async` and read with `ldmatrix`; four warps per block, each warp owning a 64-column strip over all M rows; K split across blocks and accumulated with atomics;
* a ring of prefetched weight loads (`PF` steps), scales and zero points loaded once per 128-row group;
* verified against an f32 reference (max error 0.0003).

Sweeping warps per block (2, 4, 8), K slices (1 to 8), A stages (2 to 4) and prefetch distance (2 to 8), the best configurations:

| M | prototype | generated kernel | Marlin |
|---|--:|--:|--:|
| 32 | 57.3 us | 58.8 us | 50.9 us |
| 64 | **75.8 us** | 87.4 us | 58.3 us |

So **keeping B in registers is worth about 13 percent at M = 64 and nothing at M = 32** in a straightforward implementation, and the prototype is still 28 percent slower than Marlin at M = 64. It uses 254 registers per thread and runs one warp per scheduler, which leaves it latency-bound; closing the rest is occupancy and scheduling tuning of a hand-written kernel (register pressure, two blocks per SM, instruction scheduling), the kind of work that hand-written kernels receive, not something a change to the compiler's memory path would deliver by itself.

## What I take from this

1. For these kernels the *instruction mix is a bad predictor of speed*: dequantisation is 60 percent of the instructions and worth 3 percent of the time when removed.
2. Tile sweeps are cheap and necessary, but they saturate; when the best eight configurations are within 8 percent of each other, the structure is the limit.
3. A hand-written prototype is the fastest way to bound a large compiler project. A few hours of CUDA answered a question that would have cost weeks of MLIR work: the approach would have gained at most 13 percent at M = 64.
4. The remaining gap to Marlin at medium batch is probably not a compiler-versus-human gap but a *tuning-depth* gap. A generated kernel with Marlin's register budget and scheduling would need a scheduler that understands register pressure, which this pipeline does not have.

The generated kernels stay as they are, with the tuned defaults. The prototype is kept in the repository as a documented experiment.
