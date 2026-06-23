# -*- coding: utf-8 -*-
"""日志额外标签。JsonFormatter 会把 record 上命中 `extra_tags` 的字段
挂到结构化日志里。需要新字段时，加到 `extra_tags` 并扩展 `make_extra_tags`。
"""


extra_tags: list[str] = ["audio_file"]


def make_extra_tags(audio_file: str = "") -> dict:
    return {"audio_file": audio_file}
