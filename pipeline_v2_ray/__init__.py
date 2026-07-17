"""Ray cluster runner for PipelineV2.

A persistent Ray actor per pipe_slot auto-detects its hardware and loads the
matching pipeline config (see configs/pipeline_v2_ray.yaml). Each actor runs
max_concurrency files at once on separate threads: their ffmpeg decode (CPU)
and mp3 export (CPU) overlap freely while a per-actor lock keeps exactly one
file on the GPU, so the GPU never waits on IO. The distributed cache (JuiceFS)
is a POSIX mount, so all paths are used directly.
"""
