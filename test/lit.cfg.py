"""lit configuration for all QuantForge FileCheck tests (test/Dialect, test/Transforms).

Normally driven through CMake (`check-quantforge` target, `ctest -L lit`), which
generates lit.site.cfg.py with the tool locations. Running lit directly on the
source tree also works:

    QUANTFORGE_BUILD_DIR=<build> lit test            # FILECHECK=<path> to override
"""

import os
import shutil

import lit.formats

config.name = "QuantForge"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.excludes = ["E2E", "CMakeLists.txt", "Inputs"]


def _site(name):
    return getattr(config, name, None)


# --- locate the tools -----------------------------------------------------------
tools_dir = _site("quantforge_tools_dir")
if not tools_dir:  # not configured through CMake
    build_dir = os.environ.get(
        "QUANTFORGE_BUILD_DIR",
        os.path.join(os.path.dirname(__file__), "..", "build"),
    )
    tools_dir = os.path.join(os.path.abspath(build_dir), "tools", "quantforge-opt")

quantforge_opt = os.path.join(tools_dir, "quantforge-opt")
if not os.path.exists(quantforge_opt):
    lit_config.fatal(
        f"quantforge-opt not found at {quantforge_opt}; build it first "
        "(cmake --build <build>) or set QUANTFORGE_BUILD_DIR"
    )

filecheck = os.environ.get("FILECHECK")
if not filecheck:
    llvm_tools = _site("llvm_tools_dir")
    candidates = [os.path.join(llvm_tools, "FileCheck")] if llvm_tools else []
    candidates += [shutil.which(n) for n in ("FileCheck", "FileCheck-18")]
    filecheck = next((c for c in candidates if c and os.path.exists(c)), None)
if not filecheck:
    lit_config.fatal("FileCheck not found (install llvm-18-tools or set FILECHECK)")

config.substitutions.append(("quantforge-opt", quantforge_opt))
config.substitutions.append(("FileCheck", filecheck))
config.environment["PATH"] = os.pathsep.join(
    [tools_dir, config.environment.get("PATH", "")]
)
