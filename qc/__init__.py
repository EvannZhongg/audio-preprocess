"""Offline quality control for pipeline_v3 output.

A read-only companion to the pipeline: nothing in this package writes into
the pipeline's output tree, imports ray, or mutates any production module.
It answers five questions about a finished (or in-progress) v3 run:

  1. what fraction of the raw corpus survived stage 1, and stage 2
  2. how the final segment durations are distributed
  3. whether the final segments really do contain a single speaker
  4. whether background noise really was removed
  5. whether the final output still holds adjacent same-speaker segments
     that look like they should have been merged

Entry point is `python -m qc.main`; see qc/README.md for the CLI.

Deliberately empty of imports so that `python -m qc.main --help` stays
instant -- the heavy model/torch imports live behind lazy imports in
qc/models_bundle.py and qc/gpu_worker.py, which the parquet-only analyzers
never touch.
"""
