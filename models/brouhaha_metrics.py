import logging
import os
import tempfile

import numpy as np
import scipy.io.wavfile as wavfile
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from pyannote.audio import Model

logger = logging.getLogger(__name__)

class ComputeScore:

    def __init__(self, model: str, token, device: str = 'cpu'):
        self._load_model(model, token, device)

    def _load_model(self, model, token, device):
        self.model = Model.from_pretrained(
            model, strict=False, device=device, use_auth_token=token
        )
        self.pipeline = RegressiveActivityDetectionPipeline(self.model)

    def __call__(self, samples: str, sample_rate=16000):
        temp_path = None
        try:
            custom_temp_dir = os.environ.get("LARGE_TEMP_DIR", None)
            
            if custom_temp_dir:
                 os.makedirs(custom_temp_dir, exist_ok=True)

            with tempfile.NamedTemporaryFile(suffix='.wav', dir=custom_temp_dir, delete=False) as temp_file:
                temp_path = temp_file.name
                wavfile.write(temp_path, sample_rate, samples)
            
            results = self.pipeline(temp_path)
            c50, snr = np.mean(results["c50"]), np.mean(results["snr"])
            return float(c50), float(snr)

        except Exception as e:
            logger.warning(f"Brouhaha scoring failed: {e}")
            return -420.69, -420.69
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def explain_score(self):
        pass
