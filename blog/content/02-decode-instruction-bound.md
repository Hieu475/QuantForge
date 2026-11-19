---
title: Why the decode kernel was instruction-bound, and how it went from 53 to 38 microseconds
date: 2026-11-20
order: 2
summary: A performance investigation in four steps: duplicated scale loads, a pass that stopped matching, a magic-number dequantisation, and wave quantisation. Plus the benchmarking pitfalls that nearly fooled me on a laptop GPU.
---

A batch-1 decode GEMM reads each INT4 weight once, so it should run at the speed of the memory system: for a 4096 x 4096 layer that is 8.4 MB of weights plus 0.5 MB of scales, about 47 microseconds at the 190 GB/s the card's specification allows (a read microbenchmark reached 183 GB/s). The first working kernel needed **53 microseconds**; the final one needs 38 (see the caveat at the end of the post). This post is the story of the missing 15, because the causes were not the ones I first suspected.

## Step 0: measure the right thing

Three measurement mistakes would have sent me in the wrong direction; they are worth listing because they apply to any kernel work on a laptop GPU.

1. **Wall-clock timing hides the kernel.** Launching from Python costs 45 to 270 microseconds, more than the kernel itself. All numbers here are device time from the CUPTI trace (the PyTorch profiler).
2. **A warm L2 makes the kernel look 2x faster than it is.** The RTX 4050 has 24 MB of L2, so an 8 MB INT4 weight stays resident between benchmark iterations and the benchmark measures the cache. Before every call I flush L2, and in a real model the weights always come from DRAM.
3. **How you flush matters.** Writing a 256 MB buffer (a `memset`) leaves dirty lines in L2; their write-back slows the *next* kernel by 20 to 50 percent. Reading the buffer evicts cleanly. The activations then have to be re-touched, otherwise an M = 16 case looks 20 percent slower than it is, because a harness that happens to copy the activations (to pad them) warms them for free.

On top of that the GPU shares a desktop and its clocks drift by up to 30 percent between sessions (cuBLAS fp16 reached 25 TFLOP/s in one session and 44 in another). Rule: compare kernels **interleaved in the same run**, never against a number from yesterday.

## Step 1: count instructions, not bytes

`ncu` gives the instruction mix per opcode (`sass__inst_executed_per_opcode`). For the first kernel it showed about **7.7 thread-instructions per weight**, including one `I2F` (integer-to-float conversion) per weight. A kernel that must hit memory speed with 8 MB of weights in 45 microseconds has to issue its instructions for 16.7 million weights in that time, on 20 SMs; at 7.7 instructions per weight it simply runs out of issue slots before it runs out of bandwidth. The kernel was **instruction-bound**, not memory-bound.

## Step 2: the scale loads that were loaded eight times

The dequantisation of one 16-byte work item (8 columns) needs eight scales and eight zero points. A pass (`vectorize-global-loads`) merges those eight scalar loads into one 16-byte load. It did not fire for the larger K tiles, because the index expression that maps a K row to its group (`k / 128`) was not folded to a constant: it was computed from values the pass did not recognise as bounded. The result was **eight duplicate scale and zero-point loads per iteration**. Fixing the bound analysis (the upper bound was derived from launch block-size arguments instead of from the tile sizes) took 1 x 4096 x 4096 from 53 to 46 microseconds, measured on its own.

## Step 3: a pass that silently stopped matching

The fix of step 2 had a side effect that cost more than it gained. After it, the scale/zero-point loads of the eight work items of a thread became *identical* (same address, because the group is the same), so common-subexpression elimination merged them. The vectoriser then replaced only the **first** of the now-identical loads; the packed-dequantisation pass (`vectorize-dequant`, step 4) pattern-matches "eight consecutive lanes extracted from the same vector load", and for seven of the eight items it no longer saw that shape. They fell back to the scalar path: `I2F` per weight again.

Replacing *all* duplicates of the load fixed it. Measured against the same 53-microsecond baseline, the kernel dropped to **38 microseconds**, below Marlin's 49.4 for the same layer. The lesson is about compilers, not GPUs: a performance property that depends on a chain of pattern-matching passes is fragile, so the project now has a FileCheck test whose only job is to assert that the generated PTX contains no `cvt.rn.f16` (no scalar int-to-half conversion per weight). It guards exactly the failure described above.

## Step 4: dequantisation in two instructions per two weights

The packed form of the dequantisation is the well-known trick that Marlin uses. A nibble `q` (0 to 15) placed in the low mantissa bits of the half-precision constant `1152 = 0x6480` gives **exactly `1152 + q`**, with no conversion instruction. For two weights at once, on a 32-bit register holding two half-precision lanes:

```text
packed  = (nibble_lo) | (nibble_hi << 16) | 0x64806480        // one LOP3-style combination
zero'   = (zero byte ^ 0x80) | 0x6400 per lane                  // 1152 + zero_point, built the same way
diff    = packed - zero'          // sub.f16x2: exact, an integer in [-15, 15]
weight  = diff * scale            // mul.f16x2: the only rounding
```

Both operands carry the same offset, the subtraction is exact (integers below 2048), and the single rounding happens in the multiplication, so the result is **bit-identical** to the textbook `(q - zero) * scale` computed in fp16. Per pair of weights that is one `sub.f16x2` and one `mul.f16x2`; the integer instructions around them unpack the nibbles.

## Step 5: wave quantisation, the last 10 percent

The wide shapes behave differently from 4096 x 4096. A sweep of the grid size against the bandwidth showed:

* when the grid fits in the resident blocks of the GPU (80 blocks of 128 columns at K tile 128 on this card) the kernel reads at **about 183 GB/s**;
* as soon as there is a second, partial wave, it drops to **about 151 GB/s**.

A 4096 x 11008 layer has 86 column blocks for 80 slots. Two things helped: a smaller K tile (64 rather than 128, which lowers the shared memory per block and lets more blocks be resident) and a **one-tile look-ahead for the weight loads** (`prefetch-distance=1`: the global loads for the next tile are issued before the current tile is consumed, kept in rotating registers). The look-ahead removed 19 percent on layers with a long K loop. Two tiles of look-ahead or more did nothing: the loads were already early enough, which told me memory-level parallelism was not the limit either (DRAM throughput 146 against a 176 to 200 GB/s ceiling, with the long-scoreboard stalls reduced but the time unchanged).

Things that were tried and **did not help**: split-K (a layer with K much larger than N only runs 4 percent slower, so there is no big tail to recover), a second shared-memory stage for decode, sinking the dequantisation stores, and wider block tiles (N = 256) to fit the grid in one wave.

## Summary

| change | 1 x 4096 x 4096 |
|---|--:|
| first working kernel | 53 us |
| fold the group index, one vector load for scale / zero point (alone) | 46 us |
| replace all duplicated loads so the packed dequantisation matches again (measured against the 53 us kernel) | 38 us |
| Marlin, same layer | 49.4 us |

The pattern was the same each time: **look at the instruction mix first**. Bandwidth arithmetic said the kernel should be at 45 microseconds; only the opcode histogram said why it was not.

## A caveat about the 38 microseconds

8.9 MB in 38 microseconds is 235 GB/s, above the 183 GB/s that a pure-read microbenchmark reached and above the card's 190 GB/s specification. Marlin on the same layer under the same method moves its bytes at 180 GB/s. I do not have a complete explanation: the most likely one is that a part of the weights is still served from the 24 MB L2 despite the flush (the flush reads a 256 MB buffer, but the replacement policy is not strictly LRU, and the two kernels touch memory in different orders). So the *1.30x* at 4096 x 4096 should be read as "at least on par"; the wide shapes (4096 x 11008: 22.5 MB in 139 microseconds, 162 GB/s, 0.91x of Marlin) are the cleaner comparison. This is exactly the kind of thing that an A100 or H100 measurement with a larger, differently managed L2 would clarify.
