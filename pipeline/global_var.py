import multiprocessing
import os
import uuid

import torch
import yaml
from pyannote.audio import Pipeline

from models import (brouhaha_metrics, dnsmos, funasr_asr, separate_fast,
                    smru_separate, vad, whisper_asr)
from models.eres2net.ERes2NetV2 import ERes2NetV2
from models.eres2net.features import FBank
from utils.logger import Logger
from utils.tool import detect_gpu

try:
    import models.gemini_asr as gemini_asr  # 假设路径是这样，根据实际情况修改
except ImportError:
    pass


class PipelineParam:
    g_args = None
    batch_size = 8
    device = None

    cfg = None
    logger = None

    # speaker-diarization
    dia_pipeline = None

    # asr
    asr_model = None
    validation_asr_model = None

    supported_languages = None
    multilingual_flag = None

    # vad
    vad_model = None

    # smru
    separate_predictor1 = None

   # metrics
    dnsmos_compute_score = None
    brouhaha_metric = None

    # refinement
    refinement_model = None
    refinement_feature_extractor = None



def load_asr_model(cfg, asr_provider, device_name, cli_args):
    if asr_provider == "gemini":
        if "gemini" not in cfg:
            raise ValueError("Gemini configuration not found in config.json")
        asr_model = gemini_asr.load_asr_model(
            api_key=cfg["gemini"].get("api_key"),
            model_name=cfg["gemini"].get("model", "genimi-2.5-flash"),
        )
    elif asr_provider == "funasr":
        if "funasr" not in cfg:
            raise ValueError("FunASR configuration not found in config.json")

        funasr_model = cfg["funasr"].get("model", "iic/SenseVoiceSmall")
        funasr_vad_model = cfg["funasr"].get("vad_model", "fsmn-vad")
        funasr_model_dir_cache = cfg["funasr"].get("model_dir_cache", "iic/SenseVoiceSmall")
        funasr_vad_model_dir_cache = cfg["funasr"].get("vad_model_dir_cache", "fsmn-vad")

        if funasr_model_dir_cache and os.path.exists(funasr_model_dir_cache):
            funasr_model_dir = funasr_model_dir_cache
        else:
            funasr_model_dir = funasr_model

        if funasr_vad_model_dir_cache and os.path.exists(funasr_vad_model_dir_cache):
            funasr_vad_model_dir = funasr_vad_model_dir_cache
        else:
            funasr_vad_model_dir = funasr_vad_model

        asr_model = funasr_asr.load_asr_model(asr_model="SenseVoice", model_dir=funasr_model_dir, vad_model_dir=funasr_vad_model_dir, device=device_name)

    elif asr_provider == "paraformer":
        if "paraformer" not in cfg:
            raise ValueError("Paraformer configuration not found in config.json")
        paraformer_model = cfg["paraformer"].get("model", "paraformer-zh")
        vad_model = cfg["paraformer"].get("vad_model", "fsmn-vad")
        punc_model = cfg["paraformer"].get("punc_model", "ct-punc-c")
        paraformer_model_dir_cache = cfg["paraformer"].get("model_dir_cache", "iic/paraformer-zh")
        vad_model_dir_cache = cfg["paraformer"].get("vad_model_dir_cache", "iic/fsmn-vad")
        punc_model_dir_cache = cfg["paraformer"].get("punc_model_dir_cache", "iic/ct-punc-c")
        batch_size = cfg["paraformer"].get("batch_size", 64)

        if paraformer_model_dir_cache and os.path.exists(paraformer_model_dir_cache):
            paraformer_model_dir = paraformer_model_dir_cache
        else:
            paraformer_model_dir = paraformer_model

        if vad_model_dir_cache and os.path.exists(vad_model_dir_cache):
            vad_model_dir = vad_model_dir_cache
        else:
            vad_model_dir = vad_model

        if punc_model_dir_cache and os.path.exists(punc_model_dir_cache):
            punc_model_dir = punc_model_dir_cache
        else:
            punc_model_dir = punc_model

        asr_model = funasr_asr.load_asr_model(asr_model="ParaFormer", model_dir=paraformer_model_dir, vad_model_dir=vad_model_dir,  punc_model_dir=punc_model_dir, batch_size=batch_size, device=device_name)

    elif asr_provider == "whisper":
        if "whisper" not in cfg:
            raise ValueError("whisper configuration not found in config.json")
        model_path = cfg["whisper"].get("model", "Systran/faster-distil-whisper-large-v3")
        model_dir_cache = cfg["whisper"].get("model_dir_cache", "/root/.cache/huggingface/hub/models--Systran--faster-distil-whisper-large-v3/snapshots/c3058b475261292e64a0412df1d2681c06260fab")
        if model_dir_cache and os.path.exists(model_dir_cache):
            model_path = model_dir_cache
        else:
            model_path = model_path
        whisper_compute_type = cfg["whisper"].get("compute_type", "float16")
        if device_name == "cpu" and whisper_compute_type != "float32":
            whisper_compute_type = "float32"
        whisper_batch_size = cfg["whisper"].get("batch_size", 8)
        PipelineParam.batch_size = whisper_batch_size
        asr_model = whisper_asr.load_asr_model(
            model_path = model_path,  device=device_name, threads=cli_args.threads,
            compute_type=whisper_compute_type,
            asr_options={"initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."}
        )
    else:
        raise ValueError(f"Unknown ASR provider: {asr_provider}")
    return asr_model


def init_pipeline_global(config, cli_args):
    """
    Initializes a worker process.
    - Sets up logger and device (GPU).
    - Loads all models into global variables for this process.
    """
    process_name = multiprocessing.current_process().name
    try:
        worker_id_int = int(process_name.split('-')[-1])
    except ValueError:
        # If unable to extract (e.g., in main process debugging), default to 1
        worker_id_int = 1
    worker_id = f"worker_{worker_id_int}"

    # 1. Setup globals
    g_args = cli_args
    cfg = config
    batch_size = g_args.batch_size
    logger = Logger.get_logger(f"worker_{worker_id}")

    PipelineParam.g_args = g_args
    PipelineParam.cfg = cfg
    PipelineParam.batch_size = batch_size
    PipelineParam.logger = logger

    # 2. Setup device
    if detect_gpu() and torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        
        # Check for disabled_gpu_ids attribute for multi-GPU compatibility
        if hasattr(g_args, 'disabled_gpu_ids') and g_args.disabled_gpu_ids:
            disabled_ids = [int(i.strip()) for i in g_args.disabled_gpu_ids.split(',') if i]
            available_gpus = [i for i in range(total_gpus) if i not in disabled_ids]
            if not available_gpus:
                raise RuntimeError("All available GPUs are disabled.")
        else:
            # Default behavior for main.py or when no GPUs are disabled
            available_gpus = list(range(total_gpus))

        if not available_gpus:
             raise RuntimeError("No GPUs available for processing.")

        gpu_index = (worker_id_int - 1) % len(available_gpus)
        target_gpu_id = available_gpus[gpu_index]
        
        device_name = f"cuda:{target_gpu_id}"
        device = torch.device(device_name)
        
        torch.cuda.set_device(device)
        
        logger.info(f"Initialized {worker_id} on GPU {target_gpu_id} (Mapped from index {gpu_index})")
    else:
        logger.info(f"Worker {worker_id} using CPU, threads: {g_args.threads}")
        device_name = "cpu"
        device = torch.device(device_name)
        # whisperX expects compute type: int8 on CPU
        logger.info(f"Worker {worker_id} overriding compute type to int8 for CPU.")

    PipelineParam.device = device
    
    logger.debug(f"Worker {worker_id} loading models...")

    # 3. Load all models
    # Diarization Provider Loading
    logger.debug(" * Loading Speaker Diarization Model (pyannote)")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError("huggingface_token must start with 'hf', check the config file.")

    pyannote_model = cfg["pyannote"].get("model", "pyannote/speaker-diarization-3.1")
    pyannote_model_dir_cache = cfg["pyannote"].get("model_dir_cache", "/root/.cache/torch/pyannote/models--pyannote--speaker-diarization-3.1/snapshots/84fd25912480287da0247647c3d2b4853cb3ee5d/config.yaml")
    
    if pyannote_model_dir_cache and os.path.exists(pyannote_model_dir_cache):
        with open(pyannote_model_dir_cache, "r") as fp:
            pyannote_config = yaml.load(fp, Loader=yaml.SafeLoader)
        segmentation_model = pyannote_config["pipeline"]["params"]["segmentation"]
        embedding_model = pyannote_config["pipeline"]["params"]["embedding"]
        if os.path.exists(segmentation_model) and os.path.exists(embedding_model):
            pyannote_model_dir = pyannote_model_dir_cache
        else:
            pyannote_model_dir = pyannote_model
    else:
        pyannote_model_dir = pyannote_model
    dia_pipeline = Pipeline.from_pretrained(
        pyannote_model_dir,
        use_auth_token=cfg["huggingface_token"],
    )
    dia_pipeline.to(device)
    PipelineParam.dia_pipeline = dia_pipeline

    # ASR Model Loading
    logger.debug(" * Loading ASR Model")
    asr_provider = cfg.get("asr_provider", "whisper")
    PipelineParam.asr_model = load_asr_model(cfg, asr_provider, device_name, g_args)

    # Validation ASR Model Loading
    logger.debug(" * Loading Validation ASR Model")
    validation_asr_provider = cfg.get("validation_asr_provider", "whisper")
    assert asr_provider != validation_asr_provider, "asr_provider and validation_asr_provider must be different."
    asr_validation_enabled = cfg.get("asr_validation", {}).get("enable", False)

    if asr_validation_enabled:
        logger.info("Loading another ASR model for cross-validation.")
        PipelineParam.validation_asr_model = load_asr_model(cfg, validation_asr_provider, device_name, g_args)

    # VAD
    logger.debug(" * Loading VAD Model")
    vad_model = vad.SileroVAD(device=device)
    PipelineParam.vad_model = vad_model

    # Background Noise Separation
    logger.debug(" * Loading Background Noise Model")
    separation_provider = cfg["separate"].get("provider", "uvr") # default to uvr
    
    if separation_provider == "smru":
        smru_cfg = cfg["separate"]["smru"].copy()
        if os.environ.get("USE_E128_SMRU") == "true":
            smru_cfg["conf"] = "ckpts/denoise_derev_48k_SFI_E128.yaml"
            logger.info("Using SMRU E128 model for faster GPU processing.")
        else:
            logger.info("Using SMRU standard model.")
        separate_predictor1 = smru_separate.Predictor(args=smru_cfg, device=device_name)
    else: # Default to uvr
        logger.info("Using UVR for source separation.")
        separate_predictor1 = separate_fast.Predictor(args=cfg["separate"]["uvr"], device=device_name)
    PipelineParam.separate_predictor1 = separate_predictor1

    # DNSMOS Scoring
    logger.debug(" * Loading DNSMOS Model")
    primary_model_path = cfg["mos_model"]["primary_model_path"]
    dnsmos_compute_score = dnsmos.ComputeScore(primary_model_path, device_name)
    PipelineParam.dnsmos_compute_score = dnsmos_compute_score

    if cfg["metrics"].get("use_brouhaha", False):
        brouhaha_model = cfg["metrics"].get('brouhaha', {}).get("model", "pyannote/brouhaha")
        brouhaha_model_dir_cache = cfg["metrics"].get('brouhaha', {}).get("model_dir_cache", "/root/.cache/torch/pyannote/models--pyannote--brouhaha/snapshots/c93c9b537732dd50c28c0366c73f560c3a7aeb02/pytorch_model.bin")
        if brouhaha_model_dir_cache and os.path.exists(brouhaha_model_dir_cache):
            brouhaha_model = brouhaha_model_dir_cache
        PipelineParam.brouhaha_metric = brouhaha_metrics.ComputeScore(brouhaha_model, token=cfg["huggingface_token"], device=device_name)

    # Refinement Model
    refinement_cfg = cfg.get("embedding_refinement", {})
    if refinement_cfg.get("enable", True):
        logger.debug(" * Loading ERes2Net Model for Refinement")
        eres2net_path = refinement_cfg.get("eres2net_model_path")
        if eres2net_path and os.path.exists(eres2net_path):
            refinement_model = ERes2NetV2(feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2)
            pretrained_state = torch.load(eres2net_path, map_location=device)
            refinement_model.load_state_dict(pretrained_state)
            refinement_model.to(device)
            refinement_model.eval()
            refinement_feature_extractor = FBank()

            PipelineParam.refinement_model = refinement_model 
            PipelineParam.refinement_feature_extractor = refinement_feature_extractor
        else:
            logger.warning("ERes2Net model path not found or specified, skipping refinement.")
            refinement_model = None

    # Language flags
    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    PipelineParam.supported_languages = supported_languages
    PipelineParam.multilingual_flag = multilingual_flag
    
    torch.set_num_threads(g_args.threads)
    logger.debug(f"Worker {worker_id} finished loading models.")
