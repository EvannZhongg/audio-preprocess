import tempfile

import numpy as np
import scipy.io.wavfile as wavfile
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from pyannote.audio import Model


class ComputeScore:

    def __init__(self, model: str, cache_dir: str, token, device: str = 'cpu'):
        super().__init__()
        self._load_model(model, cache_dir, token, device)

    def _load_model(self, model, cache_dir, token, device):

        self.model = Model.from_pretrained(
            model, cache_dir=cache_dir, strict=False, device=device, use_auth_token=token
        )
        self.pipeline = RegressiveActivityDetectionPipeline(self.model)

    def __call__(self, samples: str, sample_rate=16000):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                wavfile.write(temp_file.name, sample_rate, samples)
                results = self.pipeline(temp_file.name)
                c50, snr = np.mean(results["c50"]), np.mean(results["snr"])
                return float(c50), float(snr)
        except Exception:
            return -420.69, -420.69

    def explain_score(self):
        return super().explain_score(self.name)
