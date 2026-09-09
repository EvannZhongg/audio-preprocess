"""Live progress monitor for a running pipeline_v3 job.

A read-only, resident companion to the pipeline (and a lightweight sibling of
`qc/`): every few minutes it rescans the output tree's stage-1 / stage-2
parquet, joins stage 1's finished files back against the manifest to get the
progress denominator right, writes a human-readable report to
`status/status.log`, and once an hour pushes a condensed version to a WeCom
group.

Nothing here writes into the pipeline's output tree; the only files it creates
live in this directory (`status.log`, `history.jsonl`, `scan_cache.json`,
`state.json`).

Entry point is `python status/monitor.py --output ... --manifest ...`; see
status/README.md for the CLI and for what every number means.

Deliberately empty of imports, mirroring `qc/__init__.py`, so that
`--help` stays instant and so importing one submodule never drags in the
others' dependencies (notably `requests`, which reporter.py loads lazily).
"""
