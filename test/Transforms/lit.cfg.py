"""lit.cfg.py — LIT test configuration for QuantForge Transforms tests."""

import os
import lit.formats

# ---------------------------------------------------------------------------
# Basic LIT configuration
# ---------------------------------------------------------------------------
config.name = "QuantForge-Transforms"
config.test_format = lit.formats.ShTest(preamble_commands=[])
config.suffixes = [".mlir"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Source directory containing the .mlir test files
config.test_source_root = os.path.dirname(__file__)

# Build output directory (set by CMake or inferred)
build_dir = os.environ.get(
    "QUANTFORGE_BUILD_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "build"),
)
build_dir = os.path.abspath(build_dir)

# ---------------------------------------------------------------------------
# Tool substitutions
# ---------------------------------------------------------------------------
quantforge_opt = os.path.join(build_dir, "tools", "quantforge-opt", "quantforge-opt")
filecheck = os.environ.get("FILECHECK", "FileCheck")

config.substitutions.append(("quantforge-opt", quantforge_opt))
config.substitutions.append(("FileCheck", filecheck))

# Make sure quantforge-opt is on PATH for RUN lines that use bare names
config.environment["PATH"] = (
    os.path.join(build_dir, "tools", "quantforge-opt")
    + os.pathsep
    + config.environment.get("PATH", "")
)
