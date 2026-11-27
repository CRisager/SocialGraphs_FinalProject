"""Data pipeline and analyses for the 1970-1980 songs project (see README.md)."""

import os

# Workaround for a broken MKL OpenMP layer in our conda environment (MKL 2026.1 with
# llvm-openmp 23): with MKL's default "INTEL" threading layer, any numpy.linalg call
# (used by matplotlib's tight_layout, for one) kills the process with Windows error
# 0xc06d007f. The TBB layer works and gives the same results. setdefault, so an
# explicit setting wins; it has no effect without MKL. It must be set before numpy is
# first imported.
os.environ.setdefault("MKL_THREADING_LAYER", "TBB")
