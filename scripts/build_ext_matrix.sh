#!/usr/bin/env bash
# Build the C++ torch runtime (`quantforge-pack ext`) for a matrix of Python and torch versions into one pack directory.
# Each build uses a throwaway uv venv with the default PyPI wheel of that torch (the CUDA it carries decides the tag).
#   scripts/build_ext_matrix.sh OUT_DIR "3.12:2.8.0 2.9.1 2.10.0" "3.10 3.11 3.13:2.9.1 2.11.0"
# Each argument is "pythons:torch versions". Failures are logged and skipped. Needs uv, ninja-capable compiler and CUDA toolkits.
set -u
out=$(realpath -m "$1"); shift
here=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
log="$out/build_ext_matrix.log"
mkdir -p "$out"
: > "$log"
for spec in "$@"; do
  pys=${spec%%:*}; torches=${spec#*:}
  for py in $pys; do
    for tv in $torches; do
      v="$work/py$py-t$tv"
      echo "== python $py torch $tv" | tee -a "$log"
      if ! uv venv -q --python "$py" "$v" >>"$log" 2>&1; then echo "   venv failed" | tee -a "$log"; continue; fi
      if ! uv pip install -q --python "$v/bin/python" "torch==$tv" ninja >>"$log" 2>&1; then echo "   torch install failed" | tee -a "$log"; rm -rf "$v"; continue; fi
      if PYTHONPATH="$here/python" "$v/bin/python" -I -c "import sys; sys.path.insert(0, '$here/python'); from quantforge import pack; pack.main(['ext', '--out', '$out'])" >>"$log" 2>&1; then
        tail -1 "$log" | cut -c1-200
      else
        echo "   ext build failed (see $log)" | tee -a "$log"
      fi
      rm -rf "$v"
    done
  done
done
rm -rf "$work"
echo "done; runtimes in $out/ext:"; ls "$out/ext"
