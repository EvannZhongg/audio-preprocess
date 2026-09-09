
import torch
import torch.nn as nn
import torchaudio
from pyannote.audio.core.model import Model as PyannoteModel
from pyannote.audio.core.task import Problem, Resolution, Specifications

# We need to make sure this import works relative to the Emilia project root
from .eres2net.ERes2NetV2 import ERes2NetV2


class FBank(nn.Module):
    """
    Computes Kaldi-compliant FBank features.
    This is adapted from the feature extractor used in the original ERes2NetV2 script.
    """

    def __init__(self, sample_rate=16000, n_mels=80, frame_shift_ms=10, frame_length_ms=25):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav (torch.Tensor): Waveform tensor (batch, samples).
        Returns:
            torch.Tensor: Feature tensor (batch, frames, n_mels).
        """
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)

        # torchaudio.compliance.kaldi.fbank expects the input tensor to be on CPU for some versions
        feat = torchaudio.compliance.kaldi.fbank(
            wav.cpu(),
            num_mel_bins=self.n_mels,
            sample_frequency=self.sample_rate,
            frame_shift=self.frame_shift_ms,
            frame_length=self.frame_length_ms
        ).to(wav.device)

        # Per-channel mean normalization
        feat = feat - feat.mean(1, keepdim=True)
        return feat


class ERes2NetV2Embedding(PyannoteModel):
    """
    A wrapper class for the ERes2NetV2 model to make it compatible
    with the pyannote.audio framework.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path (str): Path to the pretrained ERes2NetV2 checkpoint.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        # The base pyannote.audio.core.model.Model expects sample_rate and duration
        # We set them to values appropriate for ERes2NetV2.
        super().__init__(sample_rate=16000, num_channels=1)

        self.feature_extractor = FBank(sample_rate=16000, n_mels=80)

        # ERes2NetV2 model configuration from the original script
        eres2net_config = {
            'feat_dim': 80,
            'embedding_size': 192,
            'baseWidth': 24,
            'scale': 4,
            'expansion': 4,
        }
        self.embedding_model = ERes2NetV2(**eres2net_config)

        # Load the pretrained model weights
        pretrained_state = torch.load(model_path, map_location=device)
        self.embedding_model.load_state_dict(pretrained_state)

        # The `device` attribute of an nn.Module is a read-only property.
        # We must call .to() to move the module and its parameters to the desired device.
        device_obj = torch.device(device)
        self.to(device_obj)
        self.embedding_model.eval()

        # --- pyannote.audio specifications ---
        # Initialize the specifications for a representation (embedding) task.
        # This informs pyannote about the model's purpose and characteristics.
        self._specifications = Specifications(
            problem=Problem.REPRESENTATION,
            resolution=Resolution.CHUNK,
            duration=2.0,
        )
        # Add custom attributes required by pyannote pipelines
        self.specifications.metric = "cosine"
        self.specifications.dimension = eres2net_config['embedding_size']

    def forward(self, waveform: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Extracts speaker embeddings from a batch of audio waveforms.
        pyannote.audio will handle chunking and batching.

        Args:
            waveform (torch.Tensor): A batch of waveforms (batch, 1, samples).
            weights (torch.Tensor, optional): Not used by this model, but part of the
                                              pyannote.audio API. Defaults to None.
        
        Returns:
            torch.Tensor: A batch of embeddings (batch, dimension).
        """
        # Ensure waveform is mono and on the correct device
        if waveform.shape[1] > 1:
            waveform = waveform[:, 0, :]  # Select first channel
        else:
            waveform = waveform.squeeze(1) # (batch, samples)

        # Extract FBank features
        features = self.feature_extractor(waveform)  # (batch, frames, n_mels)

        # The ERes2NetV2 model expects a 3D tensor (B, T, F).
        # It seems that in some cases, features are computed for a single
        # sample, resulting in a 2D tensor (T, F). We add the batch
        # dimension if it's missing to prevent errors in the model.
        if features.dim() == 2:
            features = features.unsqueeze(0)

        # Extract embeddings
        with torch.no_grad():
            embeddings = self.embedding_model(features) # (batch, dimension)

        return embeddings 
