import librosa

from models import smru_separate
from pipeline.global_var import PipelineParam
from utils.logger import Logger, time_logger

logger = Logger.get_logger(__name__)


@time_logger
def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A dictionary containing the separated vocals and updated audio waveform.
    """

    mix, rate = None, None

    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        # resample to 44100
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)

    # The new smru model expects a different input format and provides a different output format.
    # We will check the type of the predictor to handle this.
    if isinstance(predictor, smru_separate.Predictor):
        # SMRU model handles resampling internally and returns a different tuple.
        vocals, no_vocals = predictor.predict(mix)
    else:
        # Original UVR model
        vocals, no_vocals = predictor.predict(mix)


    # convert vocals back to previous sample rate
    logger.debug(f"vocals shape before resample: {vocals.shape}")
    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    logger.debug(f"vocals shape after resample: {vocals.shape}")
    audio["waveform"] = vocals[:, 0]  # vocals is stereo, only use one channel

    return audio
