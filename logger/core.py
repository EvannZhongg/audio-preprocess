# -*- coding: utf-8 -*-
"""日志核心模块。

写死默认值：INFO 级别，JSON 格式，滚动写入 `logs/app.log`，仅文件输出。
"""

import logging
import os
from typing import Optional

from concurrent_log_handler import ConcurrentRotatingFileHandler

from .formatter import JsonFormatter


_LOG_FILE = "logs/app.log"
_LOG_LEVEL = logging.INFO
_LOG_MAX_MB = 100
_LOG_BACKUP = 5

_logger: Optional[logging.Logger] = None
_configured = False


def setup_logger() -> logging.Logger:
    global _logger, _configured
    if _configured and _logger is not None:
        return _logger

    _logger = logging.getLogger("app")
    _logger.setLevel(_LOG_LEVEL)
    _logger.handlers.clear()

    log_file = os.path.abspath(_LOG_FILE)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = ConcurrentRotatingFileHandler(
        log_file,
        maxBytes=_LOG_MAX_MB * 1024 * 1024,
        backupCount=_LOG_BACKUP,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    _logger.addHandler(file_handler)

    _logger.propagate = False
    _configured = True
    return _logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    root = setup_logger()
    if name is None:
        return root
    return logging.getLogger(f"app.{name}")
