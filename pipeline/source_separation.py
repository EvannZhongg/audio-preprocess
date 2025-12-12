import librosa
import numpy as np
import traceback
from models import smru_separate
from utils.logger import time_logger


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
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger

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
        try:
            vocals, no_vocals = predictor.predict(mix)
            
        except RuntimeError as e:
            err_msg = str(e)
            if "istft" in err_msg or "window overlap" in err_msg or "CUDA" in err_msg:
                logger.warning(f"SMRU ISTFT Error detected: {err_msg}")
                logger.warning("Fallback: Skipping separation, using original audio as vocals.")
                
                if mix.ndim == 2:
                    vocals = mix.T  # (C, T) -> (T, C)
                else:
                    # 如果是单声道 (T,) -> (T, 1)
                    vocals = mix[:, np.newaxis]
            else:
                raise e
        
        except Exception as e:
            logger.error(f"SMRU Unknown Error: {traceback.format_exc()}")
            logger.warning("Fallback: Using original audio.")
            if mix.ndim == 2:
                vocals = mix.T
            else:
                vocals = mix[:, np.newaxis]
    else:
        # Original UVR model
        vocals, no_vocals = predictor.predict(mix)

    # Safety check for dimensions
    if vocals.ndim == 1:
        vocals = vocals[:, np.newaxis]

    logger.debug(f"vocals shape before resample: {vocals.shape}")
    
    try:
        # librosa.resample expects (Channels, Time), so we transpose .T
        vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    except Exception as e:
        logger.error(f"Resampling failed: {e}. Fallback to original.")
        # Final fallback if resampling fails
        return audio

    logger.debug(f"vocals shape after resample: {vocals.shape}")
    
    # Update audio with the first channel of vocals
    audio["waveform"] = vocals[:, 0]

    return audio
