import torch
import torchaudio

class FBank(object):
    """
    This class implements the Kaldi-compliant FBank feature extraction,
    matching the logic validated in the main script. It is designed to be
    a standalone, reusable component for the ERes2NetV2 model.
    """
    def __init__(self, sample_rate=16000, n_mels=80):
        # sample_rate and n_mels are kept for interface consistency,
        # but the core logic истины from the validated implementation.
        pass

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav (torch.Tensor): Waveform tensor.
        Returns:
            torch.Tensor: Feature tensor.
        """
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # The original logic handles cases where a batch is passed.
        # It processes only the first item.
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
            
        # Kaldi fbank requires a 2D tensor of shape (channel, n_samples)
        # but the validated logic processes single channel audio.
        feat = torchaudio.compliance.kaldi.fbank(
            wav, 
            num_mel_bins=80, 
            sample_frequency=16000
        )
        # Mean normalization
        feat = feat - feat.mean(0, keepdim=True)
        return feat 