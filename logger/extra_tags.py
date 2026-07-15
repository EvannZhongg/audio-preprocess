# -*- coding: utf-8 -*-
"""日志额外标签。JsonFormatter 会把 record 上命中 `extra_tags` 的字段
挂到结构化日志里。需要新字段时，加到 `extra_tags` 并扩展 `make_extra_tags`。

`audio_file` 存的是 relative_path(相对 audio root，可跨目录唯一区分，比
basename 更可追溯）；`version` 是产出该记录的 pipeline 版本。version 由调用方
传入（而非在此 import 常量），以保持 logger 包无重依赖。
"""


extra_tags: list[str] = ["audio_file", "version"]


def make_extra_tags(audio_file: str = "", version: str = "") -> dict:
    return {"audio_file": audio_file, "version": version}
