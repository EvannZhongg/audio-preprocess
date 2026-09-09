"""
Text quality scoring module.

Provides 3 scorers:
1. PerplexityScorer       - Token-level perplexity using a small LM (Qwen2.5-0.5B)
2. SpellChecker           - Multilingual spell/grammar checker via language-tool-python
3. Qwen3OmniTextScorer    - LLM-based text_quality + semantic_completeness +
                            tts_suitability (single API call, three scores)

Note: rule-based semantic completeness was removed. Semantic completeness is
now evaluated by the LLM scorer alongside text_quality and tts_suitability.
"""
import hashlib
import json as _json
import logging
import math
import re as _re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Perplexity Scorer (local Qwen2.5-0.5B)
# =============================================================================

class PerplexityScorer:
    """Compute token-level perplexity via a small causal LM.

    Lower PPL = more fluent text. Used to filter ASR hallucinations and gibberish.
    Returns -1 on any failure (don't drop segment, downstream decides).
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.torch = torch
        logger.info(f"Loading PPL model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    @lru_cache(maxsize=10000)
    def _score_cached(self, text: str) -> float:
        return self._compute_ppl(text)

    def _compute_ppl(self, text: str) -> float:
        if not text or not text.strip():
            return -1.0
        try:
            with self.torch.no_grad():
                enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = enc["input_ids"].to(self.device)
                if input_ids.shape[1] < 2:
                    return -1.0
                out = self.model(input_ids, labels=input_ids)
                loss = out.loss.item()
                ppl = math.exp(loss) if loss < 50 else 1e21
                return float(ppl)
        except Exception as e:
            logger.warning(f"PPL compute failed: {e}")
            return -1.0

    def score(self, text: str) -> float:
        return self._score_cached(text)


# =============================================================================
# 2. Spell Checker (language-tool-python)
# =============================================================================

class SpellChecker:
    """Multilingual spell/grammar checker.

    Uses language-tool-python (LanguageTool) which supports many languages
    including Russian, English, German, French. Lazy-init the LanguageTool
    instance per language because each one starts a Java process.

    Score = 1 - (num_errors / max(num_words, 1)), clamped to [0, 1].
    """

    # map our config language codes to LanguageTool codes
    _LANG_MAP = {
        "ru": "ru-RU",
        "en": "en-US",
        "de": "de-DE",
        "fr": "fr-FR",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
    }

    def __init__(self):
        self._tools: Dict[str, object] = {}
        self._lock = threading.Lock()

    def _get_tool(self, language: str):
        lt_lang = self._LANG_MAP.get(language, "en-US")
        with self._lock:
            tool = self._tools.get(lt_lang)
            if tool is None:
                try:
                    import language_tool_python
                    tool = language_tool_python.LanguageTool(lt_lang)
                    self._tools[lt_lang] = tool
                    logger.info(f"Initialized LanguageTool for: {lt_lang}")
                except Exception as e:
                    logger.warning(f"Failed to init LanguageTool for {lt_lang}: {e}")
                    self._tools[lt_lang] = False  # mark as failed
            elif tool is False:
                return None
            return tool

    @lru_cache(maxsize=10000)
    def _score_cached(self, text: str, language: str) -> float:
        return self._compute_score(text, language)

    def _compute_score(self, text: str, language: str) -> float:
        if not text or not text.strip():
            return -1.0
        tool = self._get_tool(language)
        if tool is None:
            return -1.0
        try:
            matches = tool.check(text)
            # CJK 没空格, split() 会把整段当 1 个词导致严重失真; 用字符数除以 4 估算"句长单位"
            if language in ("zh", "ja", "ko"):
                cleaned = _re.sub(r"\s+", "", text)
                # 大致 4 个汉字 ≈ 1 个英文 word, 给一个合理的归一化基数
                normalize_unit = max(len(cleaned) / 4.0, 1.0)
            else:
                normalize_unit = max(len(text.split()), 1)
            error_rate = len(matches) / normalize_unit
            score = max(0.0, min(1.0, 1.0 - error_rate))
            return round(score, 4)
        except Exception as e:
            logger.warning(f"Spell check failed: {e}")
            return -1.0

    def score(self, text: str, language: str = "en") -> float:
        return self._score_cached(text, language)

    def close(self):
        for tool in self._tools.values():
            if tool and tool is not False:
                try:
                    tool.close()
                except Exception:
                    pass


# =============================================================================
# 3. Qwen3-Omni Text Quality + Semantic Completeness + TTS Suitability Scorer
# =============================================================================

class Qwen3OmniTextScorer:
    """LLM-based text quality scorer using Qwen3-Omni via venus llmproxy.

    Returns three scores in a single API call (saves 66% API cost vs three calls):
      - text_quality (0-10): overall textual quality (fluency, grammar, hallucination)
      - semantic_completeness (0-10): whether the text forms a complete, coherent
        utterance (catches truncated/fragmented segments)
      - tts_suitability (0-10): whether text is suitable for TTS training data

    Pattern mirrors `audio_analysis.py`: ThreadPoolExecutor + retries + caching.
    """

    _PROMPT_TEMPLATE = (
        "请对以下文本进行三个维度的评分（0-10分）：\n\n"
        "【评分维度】\n"
        "- text_quality: 文本整体质量。考量流畅度、语法正确性、"
        "是否有 ASR 转写错误（如重复词、乱码、不通顺）、信息量。\n"
        "  10分=完美的书面文本；7-9=自然口语句子；4-6=有小问题但可读；"
        "0-3=严重错误/无意义/重复幻觉。\n"
        "- semantic_completeness: 语义完整度。文本是否构成一个完整的、"
        "语义连贯的句子或表达单元。考量：句子是否有完整的主谓结构（或符合该语言的句法）、"
        "是否被截断（句首或句尾被切掉）、是否是片段或单词、上下文是否能独立理解。\n"
        "  10分=完整独立句；7-9=完整短句或子句；4-6=可解读但不完整；"
        "0-3=明显截断/碎片/单词。\n"
        "- tts_suitability: 是否适合作为 TTS 训练数据。考量长度合理、"
        "无背景音/音效标记、无敏感词、发音明确、上下文清晰。\n"
        "  10分=理想 TTS 样本；7-9=可用；4-6=勉强；0-3=不适合。\n\n"
        "【待评估文本】\n{text}\n\n"
        "请只输出 JSON，不要其他内容：\n"
        '{{"text_quality": <0-10>, "semantic_completeness": <0-10>, "tts_suitability": <0-10>}}'
    )

    def __init__(
        self,
        api_url: str,
        api_token: str,
        model_id: str,
        timeout: int = 30,
        concurrency: int = 16,
        max_retries: int = 2,
    ):
        self.api_url = api_url
        self.api_token = api_token
        self.model_id = model_id
        self.timeout = timeout
        self.concurrency = concurrency
        self.max_retries = max_retries
        self._cache: Dict[str, Dict[str, float]] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, float]]:
        with self._cache_lock:
            return self._cache.get(key)

    def _cache_set(self, key: str, value: Dict[str, float]):
        with self._cache_lock:
            if len(self._cache) > 50000:
                # naive bound: drop half
                items = list(self._cache.items())
                self._cache = dict(items[len(items) // 2:])
            self._cache[key] = value

    def _call_api_once(self, text: str) -> Dict[str, float]:
        """Single API call. Returns {"quality": x, "semantic": y, "tts": z}; -1 on failure."""
        import requests

        # increase max_tokens to fit 3-key JSON
        prompt = self._PROMPT_TEMPLATE.format(text=text[:2000])
        messages = [
            {"role": "system", "content": "你是一个专业的文本质量评估助手。"},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 192,
        }

        fail = {"quality": -1.0, "semantic": -1.0, "tts": -1.0}
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=_json.dumps(payload),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                if "choices" in result and result["choices"]:
                    out_text = result["choices"][0]["message"]["content"]
                elif "data" in result and "text" in result["data"]:
                    out_text = result["data"]["text"]
                else:
                    return fail

                out_text = out_text.replace("```json", "").replace("```", "").strip()
                m = _re.search(r"\{[^}]+\}", out_text, _re.DOTALL)
                if m:
                    parsed = _json.loads(m.group())
                    return {
                        "quality": float(parsed.get("text_quality", -1)),
                        "semantic": float(parsed.get("semantic_completeness", -1)),
                        "tts": float(parsed.get("tts_suitability", -1)),
                    }
                return fail
            except Exception as e:
                if attempt == self.max_retries:
                    logger.warning(f"LLM scoring failed after {attempt} retries: {e}")
                    return fail
        return fail

    def score(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {"quality": -1.0, "semantic": -1.0, "tts": -1.0}
        key = self._hash_text(text)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        result = self._call_api_once(text)
        self._cache_set(key, result)
        return result

    def score_batch(self, items: List[Dict]) -> Dict[int, Dict[str, float]]:
        """Concurrent batch scoring.

        Args:
            items: [{"idx": int, "text": str}, ...]
        Returns:
            {idx: {"quality": x, "tts": y}}
        """
        if not items:
            return {}

        results: Dict[int, Dict[str, float]] = {}
        # populate cache hits up front
        pending = []
        for it in items:
            text = it.get("text", "")
            if not text or not text.strip():
                results[it["idx"]] = {"quality": -1.0, "semantic": -1.0, "tts": -1.0}
                continue
            key = self._hash_text(text)
            cached = self._cache_get(key)
            if cached is not None:
                results[it["idx"]] = cached
            else:
                pending.append((it["idx"], text, key))

        if not pending:
            return results

        def _worker(idx: int, text: str, key: str):
            r = self._call_api_once(text)
            self._cache_set(key, r)
            return idx, r

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(_worker, idx, text, key): idx for idx, text, key in pending}
            for future in as_completed(futures):
                try:
                    idx, r = future.result()
                    results[idx] = r
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"LLM batch worker failed for idx={idx}: {e}")
                    results[idx] = {"quality": -1.0, "semantic": -1.0, "tts": -1.0}

        return results
