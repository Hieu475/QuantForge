"""Run-to-run spread of the QuantForge / Marlin kernel-time ratio (independent invocations of compare_marlin.py).

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/kernel_repeat.py --shape 1x4096x4096 --shape 64x4096x4096 --repeats 5
"""
import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ns = ap.parse_args()
    here = Path(__file__).resolve().parent
    for sh in ns.shape:
        ratios, qf, mar = [], [], []
        for _ in range(ns.repeats):
            out = subprocess.run([sys.executable, str(here / "compare_marlin.py"), "--shape", sh, "--group-size", "128", "--autotune"],
                                 capture_output=True, text=True).stdout
            m = re.search(rf"{sh}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", out)
            if m:
                mar.append(float(m.group(1))); qf.append(float(m.group(2))); ratios.append(float(m.group(4)))
        if len(ratios) > 1:
            print(f"{sh:>18}: Marlin {statistics.mean(mar):7.1f} +- {statistics.stdev(mar):4.1f} us   QuantForge {statistics.mean(qf):7.1f} +- "
                  f"{statistics.stdev(qf):4.1f} us   ratio {statistics.mean(ratios):.3f} +- {statistics.stdev(ratios):.3f}  (n = {len(ratios)})", flush=True)


if __name__ == "__main__":
    main()
