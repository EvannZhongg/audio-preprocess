"""
Domain classifier using Qwen3-Omni multimodal LLM via venus llmproxy.

Two API call types:
1. classify_file_level(audio_b64, transcript)
   -> {"text_domain": {...}, "acoustic_domain": {...}}
   Single multimodal call combining text + audio for file-level domains.

2. classify_speaker(audio_b64)
   -> {"speaker_domain": {...}}
   Audio-only call per speaker for biological / identity attributes.

All values come from predefined enums passed via config; LLM is constrained
to pick from those values. Failures degrade gracefully to "unknown".
"""
import base64
import io
import json as _json
import logging
import re as _re
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _wav_b64(waveform, sample_rate: int, max_seconds: Optional[float] = None) -> str:
    """Slice waveform to max_seconds, encode as base64 WAV."""
    import soundfile as sf
    if max_seconds is not None:
        n = int(max_seconds * sample_rate)
        if n < len(waveform):
            waveform = waveform[:n]
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def slice_audio(waveform, sample_rate: int, start_sec: float, end_sec: float,
                max_seconds: Optional[float] = None):
    """Slice waveform between [start_sec, end_sec], optionally cap to max_seconds."""
    start = max(0, int(start_sec * sample_rate))
    end = min(len(waveform), int(end_sec * sample_rate))
    seg = waveform[start:end]
    if max_seconds is not None:
        n = int(max_seconds * sample_rate)
        if n < len(seg):
            seg = seg[:n]
    return seg


class Qwen3OmniDomainClassifier:
    """Multimodal domain classifier using Qwen3-Omni via venus llmproxy."""

    def __init__(
        self,
        api_url: str,
        api_token: str,
        model_id: str,
        text_enums: Dict[str, List[str]],
        acoustic_enums: Dict[str, List[str]],
        speaker_enums: Dict[str, List[str]],
        timeout: int = 60,
        max_retries: int = 2,
    ):
        self.api_url = api_url
        self.api_token = api_token
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.text_enums = text_enums
        self.acoustic_enums = acoustic_enums
        self.speaker_enums = speaker_enums

    # -------------------------- prompt builders --------------------------

    def _build_file_prompt(self, transcript: str) -> str:
        text_dom_list = ", ".join(self.text_enums.get("domain", []))
        text_scen_list = ", ".join(self.text_enums.get("scenario", []))
        text_style_list = ", ".join(self.text_enums.get("style", []))
        ac_env_list = ", ".join(self.acoustic_enums.get("environment", []))
        ac_bg_list = ", ".join(self.acoustic_enums.get("background", []))
        ac_q_list = ", ".join(self.acoustic_enums.get("quality", []))

        return (
            "请根据以下文本和音频，对该音频文件进行**两组**分类（文本侧 + 声学侧）。\n\n"
            "【1. 文本侧 text_domain】\n"
            "结合文本内容和它所暗示的演绎风格，分别选取：\n"
            f"- domain（大类，从中选一个）: {text_dom_list}\n"
            f"- scenario（细化场景，从中选一个）: {text_scen_list}\n"
            f"- style（演绎风格，从中选一个）: {text_style_list}\n\n"
            "【2. 声学侧 acoustic_domain】\n"
            "根据音频判断该音频在什么物理环境下录制、录音质量如何，分别选取：\n"
            f"- environment（环境，从中选一个）: {ac_env_list}\n"
            f"- background（背景音情况，从中选一个）: {ac_bg_list}\n"
            f"- quality（录音质量，从中选一个）: {ac_q_list}\n\n"
            f"【转录文本（可能截断为前若干段）】\n{transcript[:2500]}\n\n"
            "请只输出 JSON，所有字段值必须严格从给出的候选列表中选择，不要其他内容：\n"
            '{"text_domain": {"domain": "...", "scenario": "...", "style": "..."}, '
            '"acoustic_domain": {"environment": "...", "background": "...", "quality": "..."}}'
        )

    def _build_speaker_prompt(self) -> str:
        sp_g = ", ".join(self.speaker_enums.get("gender", []))
        sp_a = ", ".join(self.speaker_enums.get("age_group", []))
        sp_acc = ", ".join(self.speaker_enums.get("accent", []))

        return (
            "请仅根据音频片段，识别该说话人的生物学和身份特征。\n\n"
            "【speaker_domain】\n"
            f"- gender（性别，从中选一个）: {sp_g}\n"
            f"- age_group（年龄段，从中选一个）: {sp_a}\n"
            f"- accent（口音，从中选一个）: {sp_acc}\n\n"
            "请只输出 JSON，所有字段值必须严格从给出的候选列表中选择，不要其他内容：\n"
            '{"speaker_domain": {"gender": "...", "age_group": "...", "accent": "..."}}'
        )

    # -------------------------- API call --------------------------

    def _post(self, prompt: str, audio_b64: str) -> Optional[Dict]:
        import requests

        messages = [
            {"role": "system", "content": "你是一个专业的音频和文本分类助手。"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "venus_multimodal_url", "venus_multimodal_url": {
                    "mimeType": "audio/wav",
                    "url": f"data:audio/wav;base64,{audio_b64}",
                }},
            ]},
        ]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 256,
        }

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url, headers=headers,
                    data=_json.dumps(payload), timeout=self.timeout,
                )
                resp.raise_for_status()
                result = resp.json()
                if "choices" in result and result["choices"]:
                    out_text = result["choices"][0]["message"]["content"]
                elif "data" in result and "text" in result["data"]:
                    out_text = result["data"]["text"]
                else:
                    return None
                out_text = out_text.replace("```json", "").replace("```", "").strip()
                m = _re.search(r"\{.*\}", out_text, _re.DOTALL)
                if m:
                    return _json.loads(m.group())
                return None
            except Exception as e:
                if attempt == self.max_retries:
                    logger.warning(f"Domain classifier failed after {attempt} retries: {e}")
                    return None
        return None

    # -------------------------- helpers --------------------------

    def _validate_against_enum(self, value: str, allowed: List[str]) -> str:
        """Snap LLM output to enum; on mismatch return 'unknown' or first 'unknown'-like value."""
        if not value or not isinstance(value, str):
            return "unknown" if "unknown" in allowed else (allowed[-1] if allowed else "unknown")
        if value in allowed:
            return value
        # case-insensitive match
        v_low = value.lower().strip()
        for a in allowed:
            if a.lower() == v_low:
                return a
        # not in enum - fallback
        return "unknown" if "unknown" in allowed else (allowed[-1] if allowed else "unknown")

    def _validate_text_domain(self, parsed: Optional[Dict]) -> Dict[str, str]:
        td = (parsed or {}).get("text_domain", {}) if isinstance(parsed, dict) else {}
        return {
            "domain": self._validate_against_enum(td.get("domain"), self.text_enums.get("domain", [])),
            "scenario": self._validate_against_enum(td.get("scenario"), self.text_enums.get("scenario", [])),
            "style": self._validate_against_enum(td.get("style"), self.text_enums.get("style", [])),
        }

    def _validate_acoustic_domain(self, parsed: Optional[Dict]) -> Dict[str, str]:
        ad = (parsed or {}).get("acoustic_domain", {}) if isinstance(parsed, dict) else {}
        return {
            "environment": self._validate_against_enum(ad.get("environment"), self.acoustic_enums.get("environment", [])),
            "background": self._validate_against_enum(ad.get("background"), self.acoustic_enums.get("background", [])),
            "quality": self._validate_against_enum(ad.get("quality"), self.acoustic_enums.get("quality", [])),
        }

    def _validate_speaker_domain(self, parsed: Optional[Dict]) -> Dict[str, str]:
        sd = (parsed or {}).get("speaker_domain", {}) if isinstance(parsed, dict) else {}
        return {
            "gender": self._validate_against_enum(sd.get("gender"), self.speaker_enums.get("gender", [])),
            "age_group": self._validate_against_enum(sd.get("age_group"), self.speaker_enums.get("age_group", [])),
            "accent": self._validate_against_enum(sd.get("accent"), self.speaker_enums.get("accent", [])),
        }

    # -------------------------- public API --------------------------

    def classify_file_level(self, audio_b64: str, transcript: str) -> Dict[str, Dict[str, str]]:
        """Single multimodal call: text_domain + acoustic_domain."""
        if not audio_b64:
            return {
                "text_domain": self._validate_text_domain(None),
                "acoustic_domain": self._validate_acoustic_domain(None),
            }
        prompt = self._build_file_prompt(transcript or "")
        parsed = self._post(prompt, audio_b64)
        return {
            "text_domain": self._validate_text_domain(parsed),
            "acoustic_domain": self._validate_acoustic_domain(parsed),
        }

    def classify_speaker(self, audio_b64: str) -> Dict[str, Dict[str, str]]:
        """Audio-only multimodal call: speaker_domain."""
        if not audio_b64:
            return {"speaker_domain": self._validate_speaker_domain(None)}
        prompt = self._build_speaker_prompt()
        parsed = self._post(prompt, audio_b64)
        return {"speaker_domain": self._validate_speaker_domain(parsed)}
