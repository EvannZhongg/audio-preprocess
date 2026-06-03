# meta_info_config.py
import json
from pathlib import Path
from typing import Any, Dict

META_INFO_TEMPLATE = {
    "pipeline_version": "1.0",
    "origin": {
        "raw_audio_path": "",
        "sample_rate": 24000,
        "duration": 0.0
    },
    "sentences": [
        {
            "utt_id": "",
            "spk_id": "",
            "speaker_min_similarity": 0.0,
            "time_range": {
                "duration": 0,
                "start_time": 0,
                "end_time": 0
            },
            "transcription_info": {
                "text": "今天天气真好…",
                "val_text": "今天天气真好。",
                "norm_text": "今天天气真好。",
                "language": "zh",
                "wer": 0.15,
                "avg_char_duration": 0.2,
                "speaking_rate": -1.0,
                "alignment_score": -1.0,
                "abnormal_silence_count": -1
            },
            "audio_quality_info": {
                "snr": 20,
                "c50": 40,
                "dnsmos": 3
            },
            "text_quality_info": {
                "ppl": -1.0,
                "spell_score": -1.0,
                "llm_quality": -1.0,
                "semantic_completeness": -1.0,
                "tts_suitability": -1.0
            },
            "domain_info": {
                "text_domain": {
                    "domain": "unknown",
                    "scenario": "unknown",
                    "style": "unknown"
                },
                "acoustic_domain": {
                    "environment": "unknown",
                    "background": "unknown",
                    "quality": "unknown"
                },
                "speaker_domain": {
                    "gender": "unknown",
                    "age_group": "unknown",
                    "accent": "unknown"
                }
            },
            "speaker_info": {},
            "paralinguistics_info": {}
        }
    ]
}

class MetaConfig:
    def __init__(self, config_data: Dict[str, Any] = None):
        self._config = config_data if config_data is not None else self.create_empty_template()

    @classmethod
    def create_empty_template(cls) -> Dict[str, Any]:
        """创建一个空的配置模板"""
        import copy
        return copy.deepcopy(META_INFO_TEMPLATE)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'MetaConfig':
        """
        从JSON文件加载配置。
        
        Args:
            file_path: 配置文件路径。
            
        Returns:
            SimpleMetaConfig实例。
            
        Raises:
            FileNotFoundError: 当文件不存在时。
            JSONDecodeError: 当文件不是有效的JSON时。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data)

    def save_to_file(self, file_path: str, indent: int = 2) -> None:
        """
        将当前配置保存到JSON文件。
        
        Args:
            file_path: 要保存的文件路径。
            indent: JSON缩进，使得文件易于阅读。
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True) # 自动创建目录

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, ensure_ascii=False, indent=indent)
        print(f"配置已保存至: {file_path}")

    def get_config(self) -> Dict[str, Any]:
        """获取当前的配置字典副本"""
        import copy
        return copy.deepcopy(self._config)

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新整个配置"""
        self._config = new_config

    def update_origin(self, **kwargs) -> None:
        """更新origin部分的字段"""
        self._config['origin'].update(kwargs)

    def add_sentence(self, sentence_data: Dict[str, Any]) -> None:
        """向句子列表中添加一个句子"""
        self._config['sentences'].append(sentence_data)

    def clear_sentences(self) -> None:
        """清空句子列表"""
        self._config['sentences'].clear()

    def __repr__(self):
        return f"SimpleMetaConfig(version={self._config['pipeline_version']}, sentences={len(self._config['sentences'])})"
