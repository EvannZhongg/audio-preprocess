import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from collections import OrderedDict
import yaml
import os

from .SMRUse_SFI import SMRUse_SFI

class test_feat_cfg():
    def __init__(self, config):
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
        self.segment = config['segment']
        self.sr = config['sr']
        self.stft_window = config['stft_window']
        
        
class test_model_cfg():
    def __init__(self, config):
        self.ppFlag = config['ppFlag']
        self.num_repeat = config['num_repeat']
        self.input_feat_dim = config['input_feat_dim']
        self.feat_emb_dim = config['feat_emb_dim']

class Predictor:
    def __init__(self, args, device='cpu'):
        self.device = device
        
        # The 'conf' path from config.json is expected to be relative to the project's execution directory.
        cfg_path = args.get("conf")
        if not cfg_path:
            raise ValueError("SMRU 'conf' path not specified in config.json")
        
        with open(cfg_path, 'r') as file:
            config = yaml.safe_load(file)
            feat_cfg = test_feat_cfg(config['feat_cfg'])
            model_cfg = test_model_cfg(config['model_cfg'])
            ckpt_path = config['ckpt_path']
            
        # If ckpt_path is not absolute, resolve it relative to the directory of the YAML config file.
        if not os.path.isabs(ckpt_path):
            # os.path.dirname(cfg_path) gives the directory of the yaml file.
            # e.g. if cfg_path is 'ckpts/conf.yaml', dirname is 'ckpts'
            # and ckpt_path is 'model.pt', the final path becomes 'ckpts/model.pt'
            ckpt_path = os.path.join(os.path.dirname(cfg_path), ckpt_path)
            
        self.model = SMRUse_SFI(feat_cfg, model_cfg).to(self.device)
        self.load_model(ckpt_path)
        
        self.chunk_size = args.get('chunk_size', 12)  #
        self.valid_size = args.get('valid_size', 8)    # 
        self.overlap = args.get('overlap', 1)        # 
        self.batch_size = args.get('batch_size', 16) # Batch size for inference

    def load_model(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict['denoise'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, inputs):
        """
        Takes a mix waveform (numpy array) and returns separated vocals and background
        """
        # Ensure input is stereo
        if inputs.ndim == 1:
            inputs = np.stack([inputs, inputs])
            
        # SMRU expects mono, so we take the mean of the channels.
        # This might need adjustment depending on how you want to handle stereo.
        mono_input = np.mean(inputs, axis=0)
        
        sr = 44100  # The input to source_separation is resampled to 44100
        
        # Resample for the model
        if sr == 22050:
            resampled_input = librosa.resample(mono_input, orig_sr=sr, target_sr=24000)
            sr_label = 24000
        elif sr == 44100:
            resampled_input = librosa.resample(mono_input, orig_sr=sr, target_sr=48000)
            sr_label = 48000
        else:
            resampled_input = mono_input
            sr_label = sr

        chunk_samples = int(self.chunk_size * sr_label)
        valid_samples = int(self.valid_size * sr_label)
        overlap_samples = int(self.overlap * sr_label)
        step_samples = valid_samples - overlap_samples

        if len(resampled_input) <= valid_samples:
            num_chunks = 1
        else:
            num_chunks = (len(resampled_input) - valid_samples + step_samples - 1) // step_samples + 1

        output = np.zeros(len(resampled_input), dtype=np.float32)
        window = np.hanning(2 * overlap_samples).astype(np.float32)

        input_chunks = []
        chunk_info = []

        for i in range(num_chunks):
            global_start = i * step_samples
            context_samples = int((self.chunk_size - self.valid_size) / 2 * sr_label)

            chunk_start = global_start - context_samples
            chunk_end = chunk_start + chunk_samples

            start = max(0, chunk_start)
            end = min(len(resampled_input), chunk_end)
            input_chunk = resampled_input[start:end]

            pad_left = max(0, -chunk_start)
            pad_right = max(0, chunk_end - len(resampled_input))

            input_chunk = np.pad(input_chunk, (pad_left, pad_right), mode='constant')

            if len(input_chunk) == chunk_samples:
                input_chunks.append(input_chunk)
                chunk_info.append({'global_start': global_start, 'index': i})

        # Process chunks in batches
        for i in tqdm(range(0, len(input_chunks), self.batch_size), desc="SMRU Denoising (Batched)"):
            batch_chunks = input_chunks[i:i + self.batch_size]
            batch_info = chunk_info[i:i + self.batch_size]

            input_tensor = torch.tensor(np.array(batch_chunks), dtype=torch.float32).to(self.device)
            output_batch, _, _ = self.model(input_tensor, sr_label)
            output_batch = output_batch.cpu().numpy()
            
            # Process each chunk in the batch result
            for j, chunk_result in enumerate(output_batch):
                info = batch_info[j]
                global_start = info['global_start']
                chunk_index = info['index']

                context_samples = int((self.chunk_size - self.valid_size) / 2 * sr_label)
                valid_start_in_chunk = context_samples
                valid_output = chunk_result[valid_start_in_chunk:valid_start_in_chunk + valid_samples]

                # Overlap-add logic
                if chunk_index != 0:
                    valid_output[:overlap_samples] *= window[:overlap_samples]
                if chunk_index != num_chunks - 1:
                    valid_output[-overlap_samples:] *= window[overlap_samples:]

                output_start = global_start
                output_end = output_start + len(valid_output)
                
                if output_end > len(output):
                    output_end = len(output)
                    valid_output = valid_output[:output_end - output_start]

                output[output_start:output_end] += valid_output


        # Resample back to original sr
        if sr == 22050:
            output = librosa.resample(output, orig_sr=24000, target_sr=sr)
        elif sr == 44100:
            output = librosa.resample(output, orig_sr=48000, target_sr=sr)
            
        # Normalize and ensure length matches input
        if len(output) > len(mono_input):
            output = output[:len(mono_input)]
        elif len(output) < len(mono_input):
            output = np.pad(output, (0, len(mono_input) - len(output)), 'constant')

        output = output / (np.max(np.abs(output)) + 1e-8) * 0.9
        
        # The source_separation function expects (vocals, no_vocals)
        # SMRU is a denoiser, so 'vocals' is the denoised signal.
        # 'no_vocals' would be the noise, which we can get by subtraction.
        vocals = output
        no_vocals = mono_input - output

        # Return as stereo
        vocals_stereo = np.stack([vocals, vocals], axis=0)
        no_vocals_stereo = np.stack([no_vocals, no_vocals], axis=0)

        # The original `separate_fast` returns vocals as (channels, samples)
        # and its output is transposed later. Let's return in the expected format (samples, channels)
        return vocals_stereo.T, no_vocals_stereo.T 