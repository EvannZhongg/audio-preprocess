# -*- coding: utf-8 -*-
"""日志包 - 提供统一的日志接口"""

from .core import get_logger
from .extra_tags import make_extra_tags


# 获取根logger实例
_root_logger = get_logger()

# 提供便捷的日志方法
def debug(msg, extra=None):
    """记录DEBUG级别日志"""
    _root_logger.debug(msg, stacklevel=2, extra=extra)

def info(msg, extra=None):
    """记录INFO级别日志"""
    _root_logger.info(msg, stacklevel=2, extra=extra)

def warning(msg, extra=None):
    """记录WARNING级别日志"""
    _root_logger.warning(msg, stacklevel=2, extra=extra)

def warn(msg, extra=None):
    """记录WARNING级别日志"""
    _root_logger.warning(msg, stacklevel=2, extra=extra)

def error(msg, extra=None):
    """记录ERROR级别日志"""
    _root_logger.error(msg, stacklevel=2, extra=extra)

def critical(msg, extra=None):
    """记录CRITICAL级别日志"""
    _root_logger.critical(msg, stacklevel=2, extra=extra)

def exception(msg, extra=None):
    """记录异常信息"""
    _root_logger.exception(msg, stacklevel=2, extra=extra)


__all__ = ['get_logger', 'debug', 'info', 'warning', 'error', 'critical', 'exception', 'make_extra_tags']