import torch
import pandas as pd

from utils.logger import time_logger
from pipeline.global_var import PipelineParam

logger = PipelineParam.logger


# Step 2: Speaker Diarization
@time_logger
def speaker_diarization(dia_pipeline, audio, provider="pyannote"):
    """
    Perform speaker diarization on the given audio.

    Args:
        dia_pipeline: The loaded diarization pipeline object.
        audio (dict): A dictionary containing the audio waveform and sample rate.
        provider (str): The name of the provider ('pyannote').

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A dataframe containing segments with speaker labels.
            - dict: A dictionary mapping speaker labels to their embedding centroids.
    """
    logger.debug(f"Start speaker diarization with provider: {provider}")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    speaker_centroids = {}

    if provider == "pyannote":
        waveform = torch.tensor(audio["waveform"]).to(dia_pipeline.device)
        waveform = torch.unsqueeze(waveform, 0)
        # Pass return_embeddings=True to get speaker centroids
        segments, embeddings = dia_pipeline(
            {"waveform": waveform, "sample_rate": audio["sample_rate"]},
            return_embeddings=True,
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        # Create a mapping from speaker labels to their centroid embeddings
        for i, speaker in enumerate(segments.labels()):
            speaker_centroids[speaker] = embeddings[i]

    else:
        raise ValueError(f"Unsupported diarization provider: {provider}")

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df, speaker_centroids