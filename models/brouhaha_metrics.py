import logging
import os
import tempfile
from unittest.mock import patch

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from pyannote.audio import Model

logger = logging.getLogger(__name__)

class ComputeScore:

    def __init__(self, model: str, token, device: str = 'cpu'):
        self._load_model(model, token, device)

    def _load_model(self, model, token, device):
        # PyTorch 2.6+ changed torch.load(weights_only) to True by default,
        # while this trusted pyannote Lightning checkpoint contains ordinary
        # metadata objects (e.g. TorchVersion). pyannote.audio 3.3 does not
        # pass weights_only=False itself, so scope the compatibility override
        # to this model load instead of changing global process behavior.
        torch_load = torch.load

        def load_checkpoint(*args, **kwargs):
            if kwargs.get("weights_only") is None:
                kwargs["weights_only"] = False
            return torch_load(*args, **kwargs)

        with patch("torch.load", load_checkpoint):
            self.model = Model.from_pretrained(
                model, strict=False, device=device, use_auth_token=token
            )
        if self.model is None:
            raise RuntimeError(f"failed to load Brouhaha model: {model}")
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
