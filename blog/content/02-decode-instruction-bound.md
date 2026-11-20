---
title: What makes a decode kernel fast: an ablation, and an erratum about a too-good figure
date: 2026-11-20
order: 2
summary: Why the decode kernel was instruction-bound, what each pass is worth (measured cold, today), the benchmarking pitfalls of a laptop GPU, and the correction of an earlier claim of 1.30x over Marlin that did not reproduce.
---

> **Erratum (2026-11-20).** An earlier version of this post, and of the project README, claimed that the generated decode kernel needed **38 microseconds** for a 4096 x 4096 layer, "1.30x faster than Marlin". **That figure is wrong**: it does not reproduce. Re-running the same benchmark harness on the very commit that recorded it gives 50.2 microseconds, and the harness today gives 50.0 microseconds against Marlin's 49.6, a ratio of **0.99**. The number was a measurement artefact of the development period that I cannot reconstruct exactly. Everything below has been re-measured today. The page you are reading replaces the old story ("from 53 to 38 microseconds") with the part that did survive: the diagnosis method and an ablation.

A batch-1 decode GEMM reads each INT4 weight once, so it should run at the speed of the memory system. For a 4096 x 4096 layer that is 8.4 MB of weights plus 0.25 MB of scales and 0.25 MB of zero points: 8.65 MB, about 45 microseconds at the 192 GB/s the card's specification allows.

## Step 0: measure the right thing, and check it

These pitfalls apply to any kernel work on a laptop GPU, and the erratum above shows why they matter.

1. **Wall-clock timing hides the kernel.** Launching from Python costs 45 to 270 microseconds, more than the kernel. All numbers are device time from the CUPTI trace (the PyTorch profiler).
2. **A warm L2 makes the kernel look faster than it is.** The RTX 4050 has 24 MB of L2, so an 8 MB INT4 weight stays resident between benchmark iterations and the benchmark measures the cache. Before every call I flush L2.
3. **How you flush matters.** Writing a 256 MB buffer leaves dirty lines in L2 whose write-back slows the *next* kernel by 20 to 50 percent; reading the buffer evicts cleanly. The activations then have to be touched again, otherwise a harness that happens to copy them (to pad them) hands the kernel a hot A.
4. **Clocks drift.** The GPU shares a desktop and its clocks move by up to 30 percent between sessions. Compare kernels *interleaved in the same run*, never against a number from yesterday. (The erratum is a case in point.)
5. **Verify that the bytes come from DRAM.** `ncu --cache-control all --metrics dram__bytes_read.sum` on the decode kernel reports **9.2 MB** read from DRAM for the 8.65 MB of weights and scales; at 50 microseconds that is about **184 GB/s**, essentially the card's DRAM limit. Replacing the 256 MB flush with a 1 GB one changes neither kernel's time (50.3 microseconds for both). The script is `benchmarks/l2_residency_check.py`.

So at 4096 x 4096 the generated kernel and Marlin are **both at the DRAM limit**, and a tie is exactly what one should expect.

## Step 1: count instructions, not bytes

While the kernel was being developed, `ncu` gave the instruction mix per opcode (`sass__inst_executed_per_opcode`): about **7.7 thread-instructions per weight**, including one `I2F` (integer-to-float conversion) per weight. A kernel that has to stream 16.7 million weights in 45 microseconds on 20 SMs has roughly 10 instruction slots per weight, so at 7.7 it is close to running out of issue slots before it runs out of bandwidth: the decode kernel is *instruction-bound before it is latency-bound*. (Instruction counts do not depend on cache state, so this observation is not affected by the erratum.)

## Step 2: the scale loads that were loaded eight times, and the pass that stopped matching

The dequantisation of one 16-byte work item (8 columns) needs eight scales and eight zero points. A pass (`vectorize-global-loads`) merges the eight scalar loads into one 16-byte load. Two bugs interacted:

* the index expression that maps a K row to its group (`k / 128`) was not folded to a constant for larger K tiles, because the bound analysis used launch block-size arguments instead of the tile sizes: **eight duplicate scale and zero-point loads per iteration**;
* after that was fixed, the loads of a thread's eight work items became *identical*, common-subexpression elimination merged them, and the vectoriser replaced only the **first** of them. The packed-dequantisation pass (next step) matches "eight consecutive lanes extracted from the same vector load", and for seven of the eight items it no longer saw that shape; they fell back to the scalar path with one `I2F` per weight.

Replacing *all* the duplicates restored the packed form. The lesson is about compilers: a performance property that depends on a chain of pattern-matching passes is fragile, so the project has a FileCheck test whose only job is to assert that the PTX contains no `cvt.rn.f16`. (I no longer quote microsecond gains for these two fixes, because the original measurements are the ones that did not survive.)

## Step 3: dequantisation in two instructions per two weights

The packed form is the well-known trick that Marlin uses. A nibble `q` (0 to 15) placed in the low mantissa bits of the half-precision constant `1152 = 0x6480` gives **exactly `1152 + q`**, with no conversion instruction. For two weights at once, in a 32-bit register with two half-precision lanes:

```text
packed  = (nibble_lo) | (nibble_hi << 16) | 0x64806480        // one LOP3-style combination
zero'   = (zero byte ^ 0x80) | 0x6400 per lane                  // 1152 + zero_point, built the same way
diff    = packed - zero'          // sub.f16x2: exact, an integer in [-15, 15]
weight  = diff * scale            // mul.f16x2: the only rounding
```

Both operands carry the same offset, the subtraction is exact (integers below 2048), and the single rounding happens in the multiplication, so the result is **bit-identical** to the textbook `(q - zero) * scale` computed in fp16.

## Step 4: an ablation, measured today (cold L2, 1 x 4096 x 4096)

The best way to say what each pass is worth is to switch it off. Same kernel, same shape, L2 flushed, device time:

| configuration | time | relative |
|---|--:|--:|
| all passes (the autotuned decode kernel) | **50 us** | 1.00x |
| `vectorize-dequant=false` (scalar dequantisation, one `I2F` per weight) | 58 us | 1.16x slower |
| `prefetch-distance=0` (no look-ahead for the weight loads) | 61 us | 1.22x slower |
| `vectorize-loads=false` (scalar weight / scale / zero-point loads) | **156 us** | **3.1x slower** |
| Marlin (reference) | 49.6 us | |
| cuBLAS fp16 on pre-dequantised weights (4x the bytes) | 181 us | |

With the weights already in L2 (no flush) the full kernel takes 49 microseconds: at this size it is latency-bound (few warps, each with a long dependent chain), not bandwidth-bound, which is why cold and warm timings are so close.

The ranking is informative. **Vectorised loads are the big one**: without 16-byte loads the kernel issues three times more load instructions and runs three times slower. The packed dequantisation and the one-tile look-ahead are each worth 16 to 22 percent. Passes that did *not* help at all in this regime: a second shared-memory stage for decode, sinking the dequantisation stores, look-ahead of two tiles or more (the loads are already early enough), and split-K.

## Step 5: wave quantisation on the wide shapes

The wide shapes behave differently. During development a sweep of the grid size against the achieved bandwidth showed about 183 GB/s while the grid fits in the resident blocks of the GPU (80 blocks of 128 columns at K tile 128 on this card), and about 151 GB/s as soon as a partial second wave exists: a 4096 x 11008 layer has 86 column blocks for 80 slots. A smaller K tile (64 instead of 128, which lowers the shared memory per block) with the one-tile look-ahead recovers part of the loss. These are development-time measurements; the cold results on the wide shapes today are in the overview: 0.85 of Marlin for 1 x 4096 x 11008 and 0.97 for 1 x 11008 x 4096.

## What to take from this

1. At the DRAM limit a tie with a hand-written kernel is the right result, and a claim of beating it by 30 percent on a memory-bound kernel should be checked with `dram__bytes_read` before it is believed.
2. Count instructions first: the bandwidth arithmetic said the kernel should be at 45 microseconds, only the opcode histogram explained why early versions were not.
3. Ablate, do not narrate: the table above is more informative than any story about steps.
4. Keep a regression test for every performance property that depends on pass interplay.
