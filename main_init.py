import argparse
import json
import os
import sys
import warnings
import hashlib
import torch.multiprocessing as mp
from functools import partial

# import google.generativeai as genai
# import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import tqdm
from models import (dnsmos, funasr_asr, separate_fast, vad)
from models.eres2net.ERes2NetV2 import ERes2NetV2
from models.eres2net.features import FBank
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import \
    SpeakerDiarization as PyannoteSpeakerDiarization
from pydub import AudioSegment
from utils.logger import Logger, time_logger
from utils.tool import (calculate_audio_stats, check_env, detect_gpu,
                        export_to_libritts, export_to_mp3, get_audio_files,
                        load_cfg)

warnings.filterwarnings("ignore")


def get_short_hash(text, length=6):
    """
    Generates a short, deterministic hash from a string to create a unique file-specific prefix.
    """
    if isinstance(text, str):
        text = text.encode('utf-8')
    
    hasher = hashlib.sha1(text)
    return hasher.hexdigest()[:length]


class EmiliaPipeline:
    def __init__(self, config, cli_args):
        """
        Initializes the pipeline, loading all necessary models.
        """
        self.logger = Logger.get_logger("pipeline")
        self.logger.info("Initializing processing pipeline and loading models...")

        # 1. Setup globals
        self.g_args = cli_args
        self.cfg = config
        self.batch_size = self.g_args.batch_size
        
        # 2. Setup device
        if detect_gpu() and self.g_args.num_workers > 0: # num_workers > 0 indicates GPU usage intent
            # In a server context, we might not have multiple worker IDs. Default to GPU 0
            # Or implement more sophisticated device management if multiple GPUs are to be used by the server.
            gpu_id = 0 
            self.logger.info(f"Pipeline using GPU {gpu_id}")
            device_name = f"cuda:{gpu_id}"
            simple_device_name = "cuda"
            self.device = torch.device(device_name)
        else:
            self.logger.info(f"Pipeline using CPU")
            device_name = "cpu"
            simple_device_name = "cpu"
            self.device = torch.device(device_name)
        
        self.logger.debug("Loading models...")

        # 3. Load all models
        # Diarization Provider Loading
        self.logger.debug(" * Loading Speaker Diarization Model (pyannote)")
        if not self.cfg["huggingface_token"].startswith("hf"):
            raise ValueError("huggingface_token must start with 'hf', check the config file.")
        self.dia_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.cfg["huggingface_token"],
        )
        self.dia_pipeline.to(self.device)

        # ASR Model Loading
        self.logger.debug(" * Loading ASR Model(s)")
        asr_provider = self.cfg.get("asr_provider", "whisper")
        asr_validation_enabled = self.cfg.get("asr_validation", {}).get("enable", False)

        if asr_validation_enabled:
            self.logger.info("Loading both Whisper and FunASR for cross-validation.")
            self.whisper_asr_model = whisper_asr.load_asr_model(self.g_args.whisper_arch, device_name, compute_type=self.g_args.compute_type)
            funasr_cfg = self.cfg.get("funasr", {})
            self.funasr_asr_model = funasr_asr.load_asr_model(
                model_dir=funasr_cfg.get("model_dir", "iic/SenseVoiceSmall"), device=device_name
            )
        else:
            if asr_provider == "gemini":
                # This part would need adaptation if Gemini is intended to be used in the long-running server
                raise NotImplementedError("Gemini ASR provider is not yet supported in server mode.")
            elif asr_provider == "funasr":
                if "funasr" not in self.cfg:
                    raise ValueError("FunASR configuration not found in config.json")
                self.asr_model = funasr_asr.load_asr_model(
                    model_dir=self.cfg["funasr"].get("model_dir", "iic/SenseVoiceSmall"), device=device_name
                )
            elif asr_provider == "paraformer":
                 if "paraformer" not in self.cfg:
                    raise ValueError("Paraformer configuration not found in config.json")
                 paraformer_cfg = self.cfg["paraformer"]
                 self.asr_model = funasr_asr.load_asr_model(
                    model_dir=paraformer_cfg.get("model_dir", "paraformer-zh"), device=device_name
                )
            else: # Default to Whisper
                self.asr_model = whisper_asr.load_asr_model(
                    self.g_args.whisper_arch, device_name, compute_type=self.g_args.compute_type, threads=self.g_args.threads,
                    asr_options={"initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."}
                )

        # VAD
        self.logger.debug(" * Loading VAD Model")
        self.vad = vad.SileroVAD(device=self.device)

        # Background Noise Separation
        self.logger.debug(" * Loading Background Noise Model")
        self.separate_predictor1 = separate_fast.Predictor(args=self.cfg["separate"]["step1"], device=simple_device_name)

        # DNSMOS Scoring
        self.logger.debug(" * Loading DNSMOS Model")
        primary_model_path = self.cfg["mos_model"]["primary_model_path"]
        self.dnsmos_compute_score = dnsmos.ComputeScore(primary_model_path, simple_device_name)

        # Refinement Model
        refinement_cfg = self.cfg.get("embedding_refinement", {})
        if refinement_cfg.get("enable", True):
            self.logger.debug(" * Loading ERes2Net Model for Refinement")
            eres2net_path = refinement_cfg.get("eres2net_model_path")
            if eres2net_path and os.path.exists(eres2net_path):
                self.refinement_model = ERes2NetV2(feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2)
                pretrained_state = torch.load(eres2net_path, map_location=self.device)
                self.refinement_model.load_state_dict(pretrained_state)
                self.refinement_model.to(self.device)
                self.refinement_model.eval()
                self.refinement_feature_extractor = FBank()
            else:
                self.logger.warning("ERes2Net model path not found or specified, skipping refinement.")
                self.refinement_model = None
        else:
            self.refinement_model = None

        # Language flags
        self.supported_languages = self.cfg["language"]["supported"]
        self.multilingual_flag = self.cfg["language"]["multilingual"]
        
        self.logger.info("Pipeline initialized and all models loaded successfully. ✅")
        self.audio_count = 0


    def standardization(self, audio):
        """
        Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.
        """
        name = "audio"

        if isinstance(audio, str):
            name = os.path.basename(audio)
            audio = AudioSegment.from_file(audio)
        elif isinstance(audio, AudioSegment):
            name = f"audio_{self.audio_count}"
            self.audio_count += 1
        else:
            raise ValueError("Invalid audio type")

        self.logger.debug("Entering the preprocessing of audio")
        audio = audio.set_frame_rate(self.cfg["entrypoint"]["SAMPLE_RATE"])
        audio = audio.set_sample_width(2)
        audio = audio.set_channels(1)
        self.logger.debug("Audio file converted to WAV format")
        target_dBFS = -20
        gain = target_dBFS - audio.dBFS
        self.logger.info(f"Calculating the gain needed for the audio: {gain} dB")
        normalized_audio = audio.apply_gain(min(max(gain, -3), 3))
        waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
        max_amplitude = np.max(np.abs(waveform))
        if max_amplitude > 0:
            waveform /= max_amplitude
        self.logger.debug(f"waveform shape: {waveform.shape}")
        return {
            "waveform": waveform,
            "name": name,
            "sample_rate": self.cfg["entrypoint"]["SAMPLE_RATE"],
        }

    def source_separation(self, audio):
        """
        Separate the audio into vocals and non-vocals.
        """
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)
        vocals, _ = self.separate_predictor1.predict(mix)
        vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
        audio["waveform"] = vocals[:, 0]
        return audio

    def speaker_diarization(self, audio):
        """
        Perform speaker diarization on the given audio.
        """
        self.logger.debug(f"Start speaker diarization with provider: pyannote")
        waveform = torch.tensor(audio["waveform"]).to(self.dia_pipeline.device)
        waveform = torch.unsqueeze(waveform, 0)
        segments, embeddings = self.dia_pipeline(
            {"waveform": waveform, "sample_rate": audio["sample_rate"]},
            return_embeddings=True,
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
        speaker_centroids = {
            speaker: embeddings[i] for i, speaker in enumerate(segments.labels())
        }
        return diarize_df, speaker_centroids

    def cut_by_speaker_label(self, vad_list, audio_duration, stats, step_name="post_process_vad"):
        """
        Merge and trim VAD segments by speaker labels.
        """
        MERGE_GAP = 2
        MIN_SEGMENT_LENGTH = 3
        MAX_SEGMENT_LENGTH = 20
        GRACE_PERIOD_END_S = 0.05
        updated_list = []
        discarded_long_count = 0
        discarded_long_duration = 0.0

        for vad in vad_list:
            last_start_time = updated_list[-1]["start"] if updated_list else None
            last_end_time = updated_list[-1]["end"] if updated_list else None
            last_speaker = updated_list[-1]["speaker"] if updated_list else None

            if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
                duration = vad["end"] - vad["start"]
                self.logger.warning(
                    f"Discarding segment for speaker {vad['speaker']} due to long duration ({duration:.2f}s)."
                )
                discarded_long_count += 1
                discarded_long_duration += duration
                continue

            if (
                last_speaker is None
                or last_speaker != vad["speaker"]
                or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
            ):
                updated_list.append(vad)
            elif (
                vad["start"] - last_end_time < MERGE_GAP
                and vad["end"] - last_start_time < MAX_SEGMENT_LENGTH
            ):
                updated_list[-1]["end"] = vad["end"]
            else:
                updated_list.append(vad)

        count_before_min_len_filter = len(updated_list)
        duration_before_min_len_filter = sum(s["end"] - s["start"] for s in updated_list)
        filter_list = [
            vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ]
        count_after_min_len_filter = len(filter_list)
        duration_after_min_len_filter = sum(s["end"] - s["start"] for s in filter_list)
        discarded_short_count = count_before_min_len_filter - count_after_min_len_filter
        discarded_short_duration = duration_before_min_len_filter - duration_after_min_len_filter
        stats['steps'][step_name]['discarded_count'] = discarded_long_count + discarded_short_count
        stats['steps'][step_name]['discarded_duration'] = discarded_long_duration + discarded_short_duration

        if not filter_list:
            return filter_list

        for i in range(len(filter_list) - 1):
            new_end = filter_list[i]["end"] + GRACE_PERIOD_END_S
            filter_list[i]["end"] = min(new_end, filter_list[i + 1]["start"])
        filter_list[-1]["end"] = min(filter_list[-1]["end"] + GRACE_PERIOD_END_S, audio_duration)

        return filter_list

    def refine_vad_list_by_embedding(self, vad_list, audio):
        """
        Refines the VAD list by checking for internal embedding consistency.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        refined_vad_list = []
        SIMILARITY_THRESHOLD = 0.6

        def _get_embedding(waveform_segment):
            if len(waveform_segment) / audio["sample_rate"] < 0.1: return None
            waveform_16k = librosa.resample(
                waveform_segment, orig_sr=audio["sample_rate"], target_sr=16000
            )
            features = self.refinement_feature_extractor(
                torch.tensor(waveform_16k, dtype=torch.float32).to(self.device)
            )
            with torch.no_grad():
                embedding = self.refinement_model(features.unsqueeze(0)).cpu().numpy()
            return embedding

        for segment in vad_list:
            duration = segment["end"] - segment["start"]
            if duration < 1.0:
                refined_vad_list.append(segment)
                continue

            start_frame_main = int(segment["start"] * audio["sample_rate"])
            end_frame_main = int(segment["end"] * audio["sample_rate"])
            segment_waveform = audio["waveform"][start_frame_main:end_frame_main]
            reference_embedding = _get_embedding(segment_waveform)
            if reference_embedding is None:
                refined_vad_list.append(segment)
                continue
            
            is_consistent = True
            window_start_s = 0
            while window_start_s + 1.1 <= duration:
                window_start_frame = int(window_start_s * audio["sample_rate"])
                window_end_frame = int((window_start_s + 1.1) * audio["sample_rate"])
                window_waveform = segment_waveform[window_start_frame:window_end_frame]
                window_embedding = _get_embedding(window_waveform)
                if window_embedding is None:
                    window_start_s += 0.4
                    continue
                
                similarity = cosine_similarity(reference_embedding, window_embedding)[0, 0]
                if similarity < SIMILARITY_THRESHOLD:
                    is_consistent = False
                    self.logger.debug(f"Discarding VAD segment from {segment['start']:.2f}s due to inconsistency.")
                    break
                window_start_s += 0.4
            
            if is_consistent:
                refined_vad_list.append(segment)

        return refined_vad_list

    def asr(self, vad_segments, audio):
        """
        Perform Automatic Speech Recognition on VAD segments.
        """
        if not vad_segments: return []
        temp_audio = audio["waveform"]
        start_time = vad_segments[0]["start"]
        end_time = vad_segments[-1]["end"]
        temp_audio = temp_audio[int(start_time * audio["sample_rate"]):int(end_time * audio["sample_rate"])]
        for seg in vad_segments:
            seg["start"] -= start_time
            seg["end"] -= start_time
        temp_audio = librosa.resample(
            temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
        )

        if self.cfg.get("asr_validation", {}).get("enable", False):
            # ASR Cross-Validation Logic
            # ... (omitted for brevity, assumes standard path)
            return []

        if self.multilingual_flag and self.cfg.get("asr_provider", "whisper") == "whisper":
            valid_vad_segments = []
            valid_vad_segments_language = []
            for seg in vad_segments:
                segment_audio = temp_audio[int(seg["start"] * 16000):int(seg["end"] * 16000)]
                language, prob = self.asr_model.detect_language(segment_audio)
                if language in self.supported_languages and prob > 0.8:
                    valid_vad_segments.append(seg)
                    valid_vad_segments_language.append(language)
            if not valid_vad_segments: return []
            
            all_transcribe_result = []
            for language_token in list(set(valid_vad_segments_language)):
                lang_vad_segments = [
                    valid_vad_segments[i] for i, x in enumerate(valid_vad_segments_language) if x == language_token
                ]
                transcribe_result_temp = self.asr_model.transcribe(
                    temp_audio, lang_vad_segments, batch_size=self.batch_size, language=language_token, print_progress=False
                )
                result = transcribe_result_temp["segments"]
                for res_seg in result:
                    res_seg["start"] += start_time
                    res_seg["end"] += start_time
                    res_seg["language"] = transcribe_result_temp["language"]
                all_transcribe_result.extend(result)
            return sorted(all_transcribe_result, key=lambda x: x["start"])
        else:
            language = None
            if self.cfg.get("asr_provider", "whisper") == "whisper":
                language, prob = self.asr_model.detect_language(temp_audio)
                if not (language in self.supported_languages and prob > 0.8):
                    self.logger.warning(f"Unsupported language '{language}' or low confidence.")
                    return []
            
            transcribe_result = self.asr_model.transcribe(
                temp_audio, vad_segments, batch_size=self.batch_size, language=language, print_progress=False
            )
            result = transcribe_result["segments"]
            for seg in result:
                seg["start"] += start_time
                seg["end"] += start_time
                seg["language"] = transcribe_result["language"]
            return result

    def mos_prediction(self, audio, vad_list):
        """
        Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.
        """
        audio_16k = librosa.resample(
            audio["waveform"], orig_sr=self.cfg["entrypoint"]["SAMPLE_RATE"], target_sr=16000
        )
        for vad in vad_list:
            segment = audio_16k[int(vad["start"] * 16000):int(vad["end"] * 16000)]
            vad["dnsmos"] = self.dnsmos_compute_score(segment, 16000, False)["OVRL"]
        predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])
        return predict_dnsmos, vad_list

    def filter_segments(self, mos_list, mos_filter_cfg):
        """
        Filter out segments based on MOS and other quality checks.
        """
        if not mos_list: return []
        strategy = mos_filter_cfg.get("strategy", "average")
        threshold = mos_filter_cfg.get("fixed_threshold", 3.0) if strategy == "fixed" else np.mean([v["dnsmos"] for v in mos_list])
        list_after_mos_filter = [seg for seg in mos_list if seg.get('dnsmos', 0) >= threshold]
        
        if not list_after_mos_filter: return []
        
        filtered_audio_stats, _ = calculate_audio_stats(list_after_mos_filter)
        final_filtered_list = [list_after_mos_filter[idx] for idx, _ in filtered_audio_stats]
        return final_filtered_list

    def update_stats(self, stats, step_name, list_before, list_after):
        """A helper function to calculate and update processing statistics."""
        count_before = len(list_before)
        duration_before = sum(s["end"] - s["start"] for s in list_before) if list_before else 0
        count_after = len(list_after)
        duration_after = sum(s["end"] - s["start"] for s in list_after) if list_after else 0
        stats['steps'][step_name]['discarded_count'] = count_before - count_after
        stats['steps'][step_name]['discarded_duration'] = duration_before - duration_after

    def print_processing_summary(self, stats, audio_name):
        """Prints a formatted summary of the audio processing statistics."""
        self.logger.info(f"--- Processing Summary for: {audio_name} ---")
        initial_count = stats['initial']['count']
        initial_duration = stats['initial']['duration']
        if initial_count == 0:
            self.logger.info("No initial segments found.")
            return

        for step_name, data in stats['steps'].items():
            if data['discarded_count'] > 0:
                self.logger.info(f" > Dropped by {step_name}: {data['discarded_count']} segments")

        final_count = stats['final']['count']
        final_duration = stats['final']['duration']
        retention_rate_count = (final_count / initial_count) * 100
        retention_rate_duration = (final_duration / initial_duration) * 100 if initial_duration > 0 else 0
        retention_rate_duration = min(100.0, retention_rate_duration)
        self.logger.info("-" * 20)
        self.logger.info(f"Final Retention: {final_count} / {initial_count} segments ({retention_rate_count:.2f}%)")
        self.logger.info(f"Final Duration: {final_duration:.2f}s / {initial_duration:.2f}s ({retention_rate_duration:.2f}%)")
        self.logger.info("--- End of Summary ---")

    def main_process(self, audio_path, save_path=None, audio_name=None, progress_callback=None):
        """
        Process a single audio file through the entire pipeline.
        """
        # A simple callback helper
        def _update_progress(progress, step):
            if progress_callback:
                progress_callback(progress, step)
        
        try:
            processing_stats = { 'initial': {'count': 0, 'duration': 0.0}, 'steps': { 'embedding_refinement': {}, 'post_process_vad': {}, 'asr': {}, 'mos_filter': {} }, 'final': {'count': 0, 'duration': 0.0} }
            audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
            save_path = save_path or os.path.join(os.path.dirname(audio_path) + "_processed", audio_name)
            os.makedirs(save_path, exist_ok=True)
            self.logger.info(f"Processing audio: {audio_name}, save to: {save_path}")

            _update_progress(5, "音频标准化 ⚙️")
            self.logger.info("Step 0: Preprocess all audio files")
            audio = self.standardization(audio_path)

            _update_progress(15, "人声背景声分离 🎤")
            self.logger.info("Step 1: Source Separation")
            audio = self.source_separation(audio)

            _update_progress(35, "说话人日志 🗣️")
            self.logger.info("Step 2: Speaker Diarization")
            diarize_df, _ = self.speaker_diarization(audio)
            
            file_hash = get_short_hash(audio_name)
            speaker_mapping = {old: f"SPK_{file_hash}_{old.split('_')[-1]}" for old in diarize_df["speaker"].unique()}
            diarize_df["speaker"] = diarize_df["speaker"].map(speaker_mapping)

            _update_progress(45, "语音活动检测 🔍")
            self.logger.info("Step 3: Fine-grained Segmentation by VAD")
            vad_list_initial = self.vad.vad(diarize_df, audio)
            processing_stats['initial']['count'] = len(vad_list_initial)
            processing_stats['initial']['duration'] = sum(s["end"] - s["start"] for s in vad_list_initial)
            
            if self.cfg.get("embedding_refinement", {}).get("enable", True) and self.refinement_model:
                _update_progress(55, "片段优化 ✨")
                self.logger.info("Step 3.5: Refining VAD list by speaker embedding")
                vad_list_refined = self.refine_vad_list_by_embedding(vad_list_initial, audio)
                self.update_stats(processing_stats, 'embedding_refinement', vad_list_initial, vad_list_refined)
            else:
                vad_list_refined = vad_list_initial

            _update_progress(65, "片段后处理 🛠️")
            self.logger.info("Step 4: Post-process VAD segments")
            audio_duration = len(audio["waveform"]) / audio["sample_rate"]
            segment_list = self.cut_by_speaker_label(vad_list_refined, audio_duration, processing_stats)

            _update_progress(85, "语音识别 📝")
            self.logger.info("Step 5: ASR")
            asr_result = self.asr(segment_list, audio)
            self.update_stats(processing_stats, 'asr', segment_list, asr_result)

            if not asr_result:
                self.logger.warning(f"No valid speech segments found in {audio_name}")
                return None, [], processing_stats

            _update_progress(95, "质量筛选 ✔️")
            self.logger.info("Step 6: Filter")
            _, mos_list = self.mos_prediction(audio, asr_result)
            filtered_list = self.filter_segments(mos_list, self.cfg.get("mos_filter", {}))
            self.update_stats(processing_stats, 'mos_filter', mos_list, filtered_list)

            if not filtered_list:
                self.logger.warning(f"No segments passed quality filtering for {audio_name}")
                return None, [], processing_stats

            _update_progress(100, "生成结果文件 📦")
            self.logger.info("Step 7: write result to file")
            output_format = self.cfg.get("output_format", "default")
            if output_format == "libritts":
                final_path = export_to_libritts(audio, filtered_list, save_path, audio_name)
            else:
                final_path = export_to_mp3(audio, filtered_list, save_path, audio_name)
                with open(os.path.join(save_path, audio_name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(filtered_list, f, ensure_ascii=False, indent=2)

            processing_stats['final']['count'] = len(filtered_list)
            processing_stats['final']['duration'] = sum(s["end"] - s["start"] for s in filtered_list)
            self.print_processing_summary(processing_stats, audio_name)
            return final_path, filtered_list, processing_stats
        except Exception as e:
            self.logger.error(f"---!!! An error occurred while processing '{os.path.basename(audio_path)}'. !!!---")
            self.logger.error(f"Error details: {e}", exc_info=True)
            return None, [], None


# --- For Command-Line Execution ---
def init_worker(config, cli_args):
    """
    Initializes a worker process for multiprocessing.
    This is kept for backward compatibility with the original CLI execution mode.
    """
    global pipeline_instance
    pipeline_instance = EmiliaPipeline(config, cli_args)

def main_process_wrapper(audio_path):
    """
    A wrapper for main_process to be used with pool.map.
    """
    # The pipeline_instance is a global initialized per-worker
    return pipeline_instance.main_process(audio_path)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path", type=str, default="/data/workspace/procast_30min")
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--whisper_arch", type=str, default="medium")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--exit_pipeline", type=bool, default=False)
    args = parser.parse_args()
    
    main_logger = Logger.get_logger("main_cli")
    main_cfg = load_cfg(args.config_path)

    if args.input_folder_path:
        main_logger.info(f"Using input folder from CLI: {args.input_folder_path}")
        main_cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    check_env(main_logger)
    input_folder_path = main_cfg["entrypoint"]["input_folder_path"]
    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"Input folder not found: {input_folder_path}")

    audio_paths = get_audio_files(input_folder_path)
    if not audio_paths:
        main_logger.warning(f"No audio files found in {input_folder_path}. Exiting.")
        sys.exit(0)

    num_workers = min(args.num_workers, len(audio_paths))
    main_logger.info(f"Found {len(audio_paths)} files. Processing with {num_workers} worker(s).")
    
    if num_workers > 0:
        init_args = (main_cfg, args)
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            results = list(tqdm.tqdm(pool.imap(main_process_wrapper, audio_paths), total=len(audio_paths)))
    else: # Run in single-process mode
        pipeline = EmiliaPipeline(main_cfg, args)
        for audio_path in tqdm.tqdm(audio_paths):
            pipeline.main_process(audio_path)

    main_logger.info("--- All files have been processed. ---")

    if args.exit_pipeline:
        main_logger.info("exit_pipeline is True, exiting...")
        sys.exit(0)
