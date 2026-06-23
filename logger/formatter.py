# -*- coding: utf-8 -*-
"""日志格式化器模块"""

import json
import logging
from datetime import datetime

from .extra_tags import make_extra_tags, extra_tags

class JsonFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def __init__(self, time_fmt='%Y-%m-%d %H:%M:%S'):
        super().__init__()
        self.time_fmt = time_fmt
    
    def format(self, record):
        log_data = {
            'timestamp': f"{datetime.fromtimestamp(record.created).strftime(self.time_fmt)}.{int(record.msecs):03d}",
            'level': record.levelname,
            'file': f"{record.filename}:{record.lineno}",
            'process': record.process,
            'content': record.getMessage(),
        }
        
        for key, value in record.__dict__.items():
            if key in extra_tags:
                log_data[key] = value
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        try:
            return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            error_data = {
                'timestamp': f"{datetime.fromtimestamp(record.created).strftime(self.time_fmt)}.{int(record.msecs):03d}",
                'level': 'ERROR',
                'file': f"{record.filename}:{record.lineno}",
                'process': record.process,
                'content': log_data.get('message', ''),
                'exception': f"JsonFormatter format error: {str(e)}" 
            }
            return json.dumps(error_data, ensure_ascii=False, separators=(',', ':'))
