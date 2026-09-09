# `qc/` — offline quality control for `pipeline_v3` output

A read-only companion to the pipeline. It never writes into the output tree,
never imports `ray`, and does not modify a single line of existing code — point
it at a finished (or still-running) `pipeline_v3` output root and it answers five
questions about the data.

```bash
# everything: full-dataset statistics + model re-checks on a 2000-segment sample per shard
python -m qc.main \
    --output      /path/to/pipeline_v3/output \
    --manifest    /path/to/manifest \
    --config      configs/config_for_a10.json
```

---

## What it reports

| # | Section | Scope | Needs GPU |
|---|---|---|---|
| 1 | Effective data rate: raw → stage 1 → stage 2 | full dataset | no |
| 2 | Final output duration distribution | full dataset | no |
| 3 | Single-speaker re-check (two independent methods) | sample | yes |
| 4 | Background-removal re-check | sample | yes |
| 5 | Adjacent same-speaker mergeability | full dataset (+ sampled embeddings) | partly |

Sections 1, 2 and the structural half of 5 read only parquet, so they run
anywhere — including a box with no CUDA and no `torch`.

### 1. Effective data rate

Three levels, each measured **both** by file count and by seconds:

- **level 0** the raw corpus, from the `build_manifest.py` manifest
- **level 1** stage-1 segments (`segments_part-*.parquet`)
- **level 2** stage-2 kept segments (`stage2_segments_part-*.parquet`) — the final output

Also broken out: stage-1 failure types, all five stage-2 `dropped_by_*` reasons
with rows and hours (not mutually exclusive — one segment can trip several), and
retriable `asr_access_failed` rows counted separately since the next run
reprocesses them.

> **Why both file and duration rates?** A stage-1 file fails as a *unit*: when
> any chunk fails, the actor discards that file's already-successful segments too
> (`pipeline_v2_ray/actors/v2_stage_1.py:134-144`). An 8-hour file that failed on
> its last chunk contributes zero seconds while counting as one failed file, so
> the duration rate is structurally lower than the file rate. The gap between
> them is an upper bound on what a chunk-level retry could recover.

### 2. Duration distribution

Two bucketings side by side, both by segment count and by summed duration:

- **coarse** `<4 / 4-8 / 8-15 / >=15s` — same buckets as `tmp/stat_parquet.py`,
  so the numbers are directly comparable with figures already circulated from
  that script
- **fine** `0-3 / 3-5 / 5-7 / 7-9 / 9-12 / 12-15 / 15+s` — same as
  `misc/analyze_output.py`

Plus mean/P50/P90/P99, a per-language and per-shard split, and a count of
segments falling outside the production `min/max_segment_length`.

### 3. Single-speaker re-check

Two methods, applied independently to every sampled segment:

| Method | What it does | Character |
|---|---|---|
| Embedding self-consistency | slides 1.1s windows, embeds each, takes min cosine vs. the segment embedding; multi-speaker when below `inter_similarity_threshold` | production's own test, re-run — cheap, same standard the data was filtered with |
| pyannote diarization | runs diarization on the isolated segment and counts speakers | genuinely independent, much slower |

The report gives each method's detection rate, their 2×2 agreement matrix, the
same matrix split by segment length, and the drift against the
`speaker_min_similarity` already stored in the parquet.

The **diarization-only** cell is the interesting one: segments production's own
test passed but an independent model considers multi-speaker.

### 4. Background-removal re-check

Stage 1 runs separation/denoise before segmenting, so the exported wavs should be
clean speech. This measures what actually came out:

- **DNSMOS BAK** — the background component, graded `clean` / `mild_residual` /
  `clear_residual` via `--bak-pass` / `--bak-warn`. **New to QC**: production
  calls the same model but keeps only `OVRL` and discards SIG/BAK
  (`pipeline_v2/steps/metrics.py:81-83`), and an overall score can stay
  respectable while the background is clearly audible.
- **brouhaha SNR / C50** — objective noise and reverberation, comparable against
  the production `fixed_snr_threshold` / `fixed_c50_threshold`.

Also reported: drift vs. the recorded `dnsmos`/`c50`/`snr`, and scoring-failure
counts including brouhaha's `-420.69` sentinel (excluded from every statistic).

### 5. Adjacent mergeability

Rebuilds adjacency among **stage-2 surviving** segments inside each chunk and
classifies every adjacent pair by the *first* production merge condition it
fails (`pipeline_v2/steps/segment.py:120-140`), so the categories partition the
pairs:

| Category | Meaning |
|---|---|
| `different_speaker` | different `speaker_id` |
| `no_embedding_short_segment` | one side is under 1s, so it never had an embedding and **can never merge** — not a missed merge |
| `gap_too_large` | `gap >= merge_gap` |
| `merged_too_long` | merged length `>= max_segment_length` |
| `mergeable_by_structure` | every geometric condition passes — a candidate missed merge |

Embedding similarity (the one remaining condition) is measured on a sample and
reported as a separate pass rate, which is then used to extrapolate the
structural count into an estimated missed-merge figure.

---

## CLI

| Flag | Default | Notes |
|---|---|---|
| `--output` | *required* | the `--output` you passed to `main_v3_ray.py` |
| `--manifest` | – | raw-data manifest; without it level 0 is unavailable |
| `--config` | – | production config json — **strongly recommended**, see below |
| `--shards` | all | comma-separated shard names |
| `--steps` | all | subset of `yield,duration,merge,speaker,background` |
| `--sample-n` | `2000` | segments to re-check; **`0` = every segment** |
| `--sample-scope` | `per-shard` | `per-shard` keeps a big shard from crowding out the sample |
| `--pair-sample-n` | `2000` | adjacent pairs to score embeddings for |
| `--workers` | `32` | processes for the parquet passes |
| `--gpu-workers` | `1` | processes for the model passes |
| `--devices` | `cuda:0` | comma-separated, round-robined; `cpu` forces CPU |
| `--merge-buckets` | `64` | hash buckets for the chunk regroup |
| `--bak-pass` / `--bak-warn` | `4.0` / `3.0` | QC-only background grade cutoffs |
| `--report-dir` | `<output>/_qc_reports` | |
| `--work-dir` | `<report-dir>/_work` | scratch + resumable cache |
| `--keep-work` | off | keep scratch (needed to resume later) |
| `--quiet` | off | skip stdout; still writes the files |

### Always pass `--config`

Thresholds are read from the production config through
`PipelineParams.from_config` — production's own parser — so QC's verdicts use
exactly the bar the data was produced with. The values genuinely differ per
profile:

| config | `merge_gap` | `min_segment_length` | `intra_sim` | `inter_sim` |
|---|---|---|---|---|
| `config_for_a10.json` | 0.8 | 2.0 | 0.63 | 0.69 |
| `config_for_a100.json` | 0.6 | 2.0 | 0.64 | 0.69 |
| `config_for_v100_for_ko.json` | 1.0 | 1.5 | 0.68 | 0.50 |

Without `--config` the run falls back to built-in defaults and **says so loudly**
in the report — those numbers answer a different question than your data. Model
re-checks (3, 4) additionally *require* `--config`, since the model paths,
pyannote cache and auth token all live in it.

---

## Common invocations

```bash
# statistics only -- no GPU, no torch needed, full dataset
python -m qc.main --output /path/out --steps yield,duration,merge

# exhaustive re-check across two GPUs (slow: hours, not minutes)
python -m qc.main --output /path/out --config configs/config_for_a10.json \
    --steps speaker,background --sample-n 0 \
    --gpu-workers 2 --devices cuda:0,cuda:1 --keep-work

# resume an interrupted exhaustive run -- reuses every cached verdict
python -m qc.main --output /path/out --config configs/config_for_a10.json \
    --steps speaker,background --sample-n 0 --keep-work \
    --work-dir /path/out/_qc_reports/_work

# one shard, tighter background bar
python -m qc.main --output /path/out --config configs/config_for_a10.json \
    --shards manifest_part-00000 --bak-pass 4.3 --bak-warn 3.5
```

## Output

Three files per run in `--report-dir`, all rendered from the same data so they
cannot disagree:

- `qc_report_<ts>.txt` — aligned tables for reading in a terminal (also printed to stdout)
- `qc_report_<ts>.json` — the complete result, machine-readable
- `qc_report_<ts>.md` — for pasting into a doc

---

## Known biases (stated in the report too)

- **DNSMOS on short segments.** DNSMOS needs 9.01s and self-concatenates shorter
  clips (`models/dnsmos.py:157-159`), so BAK on a 2s segment reflects a looped
  signal. Grades are therefore also broken out by length band.
- **pyannote on short segments.** It tends to report a single speaker on very
  short audio regardless of content, so the `<2s` band of the cross-check matrix
  is explicitly flagged unreliable.
- **Sub-1s segments have no embedding.** `EmbeddingRefiner` skips them
  (`pipeline_v2/steps/embedding_refinement.py:85-90`), so they are reported as
  *not checked* rather than as failures, and their pairs are structurally
  unmergeable rather than missed merges.
- **`use_brouhaha=false`.** Stage 1 then stored placeholder c50/snr instead of
  measured values, so QC's brouhaha numbers are real but have nothing meaningful
  to compare against; the drift figures should be ignored for such a run.
- **Re-check drift is expected to be small but non-zero.** Production scores the
  pre-export waveform; QC reads the exported wav.

## Sampling and resumability

Sampling is by SHA-1 of `utt_id`, keeping the N smallest hashes. This is
uniform, streams in O(N) memory, is **identical across runs and machines**, and
has a useful property: the sample for N is a subset of the sample for N+k, so
raising `--sample-n` reuses every cached verdict instead of invalidating it.

Each GPU worker appends verdicts to its own jsonl in `--work-dir`, flushed per
record. A restart with the same `--work-dir` skips whatever is already scored, so
an interrupted multi-hour run resumes rather than restarting. A crash truncates
at most the final line, which the reader tolerates. The cache holds only scores
— never audio or transcript text.

## Implementation notes

- **Nothing is reimplemented that production already does.** The keep-segment
  predicate is transcribed from `tmp/stat_stage2.py:35-44`; thresholds come from
  `PipelineParams.from_config`; the speaker-consistency check drives
  `EmbeddingRefiner`'s own windowing and embedding. A QC tool that measures
  something subtly different from production is worse than useless — it is
  misleading.
- **`qc/models_bundle.py` is the only file that touches production models.** The
  analyzers are pure statistics, so an upstream signature change has one place to
  be fixed. It also asserts at load time that the mirrored constants still match
  `embedding_refinement`, turning a silent behaviour change into a visible
  warning.
- **Memory is O(buckets), not O(rows).** Workers return fixed-size counters and
  histograms, never rows, so a shard with millions of segments does not have to
  fit in the parent process. Percentiles are histogram-estimated for the same
  reason.
- **Chunk-aware GPU dispatch.** Work is grouped by chunk wav, so a 40-segment
  chunk is decoded once rather than forty times, and all three re-checks share
  that single decode.
- **Cross-part chunks are handled.** The driver flushes every 100k rows
  regardless of chunk boundaries, so one chunk's segments can straddle two
  parquet files. Requirement 5 re-partitions by `hash(chunk_audio_path)` before
  grouping, otherwise every flush boundary would lose a pair.
- **Secrets never reach a report.** `configs/config_for_*.json` holds
  `huggingface_token` and API keys in plain text. Only an explicit numeric
  allow-list is echoed, and the renderer re-scans the payload and refuses to
  write if a credential-like key ever appears.
- **Untrusted paths are vetted.** `chunk_audio_path` comes from parquet QC did not
  write, so it is resolved and confirmed to stay inside `--output` before being
  opened.

## Relationship to `tmp/`

The ad-hoc scripts in `tmp/` (`stat_parquet.py`, `stat_stage2.py`,
`stat_json.py`) are left untouched. This module is a superset of their
statistics with the semantics pinned to production; `tmp/stat_stage2.py` remains
useful as an independent oracle — it produces byte-identical keep-rate numbers.
