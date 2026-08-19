"""pipeline_v3: multi-stage streaming Ray pipeline.

Runs several processing stages (today: stage_1 = raw-audio decode/VAD/export,
stage_2 = remote ASR + v1 post-processing) in ONE process, each with its own
Ray custom resource (slot_stage_1, slot_stage_2, ...) and actor pool, so a
file finishing stage N is immediately queued into stage N+1 instead of
waiting for the whole shard to finish stage N first.

Reuses pipeline_v2_ray's actors, segment schemas and resume/writer functions
verbatim (pipeline_v2_ray itself is untouched); pipeline_v3 only adds the
multi-stage scheduling layer on top. See pipeline_v3/stages.py for the
per-stage plugin registry -- that's the extension point for a future
stage_3.
"""
