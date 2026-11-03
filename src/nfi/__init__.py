"""nfi: neuro-fuzzy classifiers under class imbalance.

macOS note: LightGBM and PyTorch each bundle an OpenMP runtime; loading both
crashes or deadlocks the process.  scripts/setup_macos.sh repoints LightGBM at
PyTorch's copy, which is why running outside that environment may hang.
"""
