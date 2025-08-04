import os
import torch
from pyannote.audio import Pipeline

from utils.logger import Logger
from utils.tool import detect_gpu
from models.eres2net.features import FBank
from models.eres2net.ERes2NetV2 import ERes2NetV2
from models import dnsmos, funasr_asr, separate_fast, vad, smru_separate

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
    whisper_asr_model = None
    funasr_asr_model = None

    supported_languages = None
    multilingual_flag = None

    # vad
    vad_model = None

    # smru
    separate_predictor1 = None

    # mos
    dnsmos_compute_score = None

    # refinement
    refinement_model = None
    refinement_feature_extractor = None


def init_pipeline_global(config, cli_args):
    """
    Initializes a worker process.
    - Sets up logger and device (GPU).
    - Loads all models into global variables for this process.
    """
    from multiprocessing.process import current_process
    worker_id_str = current_process().name
    if worker_id_str == "MainProcess":
        worker_id = 0
    else:
        worker_id = int(worker_id_str.split('-')[-1]) - 1

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

        # Assign worker to an available GPU
        gpu_id = available_gpus[worker_id % len(available_gpus)]
        
        logger.info(f"Worker {worker_id} using GPU {gpu_id} (from available list: {available_gpus})")
        device_name = f"cuda:{gpu_id}"
        device = torch.device(device_name)
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
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=cfg["huggingface_token"],
    )
    dia_pipeline.to(device)
    PipelineParam.dia_pipeline = dia_pipeline

    # ASR Model Loading
    logger.debug(" * Loading ASR Model(s)")
    asr_provider = cfg.get("asr_provider", "whisper")
    asr_validation_enabled = cfg.get("asr_validation", {}).get("enable", False)

    if asr_validation_enabled:
        logger.info("Loading both Whisper and FunASR for cross-validation.")
        whisper_asr_model = whisper_asr.load_asr_model(g_args.whisper_arch, device_name, compute_type=g_args.compute_type)
        funasr_cfg = cfg.get("funasr", {})
        funasr_asr_model = funasr_asr.load_asr_model(
            model_dir=funasr_cfg.get("model_dir", "iic/SenseVoiceSmall"), device=device_name
        )
        PipelineParam.whisper_asr_model = whisper_asr_model
        PipelineParam.funasr_asr_model = funasr_asr_model
    else:
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
            asr_model = funasr_asr.load_asr_model(
                model_dir=cfg["funasr"].get("model_dir", "iic/SenseVoiceSmall"), device=device_name
            )
        elif asr_provider == "paraformer":
            if "paraformer" not in cfg:
                raise ValueError("Paraformer configuration not found in config.json")
            paraformer_cfg = cfg["paraformer"]
            asr_model = funasr_asr.load_asr_model(
                model_dir=paraformer_cfg.get("model_dir", "paraformer-zh"), device=device_name
            )
        else: # Default to Whisper
            asr_model = whisper_asr.load_asr_model(
                g_args.whisper_arch, device_name, compute_type=g_args.compute_type, threads=g_args.threads,
                asr_options={"initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."}
            )
        PipelineParam.asr_model = asr_model

    # VAD
    logger.debug(" * Loading VAD Model")
    vad_model = vad.SileroVAD(device=device)
    PipelineParam.vad_model = vad_model

    # Background Noise Separation
    logger.debug(" * Loading Background Noise Model")
    separation_provider = cfg["separate"].get("provider", "uvr") # default to uvr
    
    if separation_provider == "smru":
        logger.info("Using SMRU for source separation.")
        separate_predictor1 = smru_separate.Predictor(args=cfg["separate"]["smru"], device=device_name)
    else: # Default to uvr
        logger.info("Using UVR for source separation.")
        separate_predictor1 = separate_fast.Predictor(args=cfg["separate"]["uvr"], device=device_name)
    PipelineParam.separate_predictor1 = separate_predictor1

    # DNSMOS Scoring
    logger.debug(" * Loading DNSMOS Model")
    primary_model_path = cfg["mos_model"]["primary_model_path"]
    dnsmos_compute_score = dnsmos.ComputeScore(primary_model_path, device_name)
    PipelineParam.dnsmos_compute_score = dnsmos_compute_score

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