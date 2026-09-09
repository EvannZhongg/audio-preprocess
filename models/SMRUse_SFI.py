import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import torch.nn as nn

class STFT(nn.Module):
    '''
    3D stft for data with [B,C,T]
    '''

    def __init__(self, n_fft, hop_length, win_length=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft if win_length == None else win_length

    def forward(self, input, win):
        # input :torch.float32 shape: [B, T] or [B, C, T]
        win = win.to(input.device)
        
        if input.dim() == 3:
            B, C, T = input.shape
            x = input.view(B * C, T)  # [B, C, T] -> [B*C, T]
            output = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=win,
                return_complex=True)  # [B*C, T] -> [B*C, frequecies, frames]
            _, fre, time = output.shape
            output = output.view(
                B, C, fre, time)  # [B*C, T] -> [B, C, frequecies, frames]

        elif input.dim() == 2:
            output = torch.stft(input,
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                win_length=self.win_length,
                                window=win,
                                return_complex=True)

        return output


class iSTFT(nn.Module):
    '''
    3D istft for data with [B,F,T]
    '''

    def __init__(self, n_fft, hop_length, win_length=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft if win_length == None else win_length

    def forward(self, input, seq_len, win):
        # input :torch.complex64 shape:[B, fre, time]
        win = win.to(input.device)
        output = torch.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=win,
            length=seq_len)  # [B ,frequecies, frames] -> [B, T]
        return output


class SMRUse_SFI(nn.Module):
    def __init__(self, feat_cfg, model_cfg):
        super().__init__()
        self.epsi           = 1e-8
        self.n_fft          = feat_cfg.n_fft
        self.hop_length     = feat_cfg.hop_length
        self.seq_len        = int(feat_cfg.segment * feat_cfg.sr)
        self.stft_window    = feat_cfg.stft_window
        
        self.register_buffer("window_8k", torch.sqrt(torch.hann_window(self.n_fft//6, periodic=False)), False)
        self.stft_8k = STFT(n_fft=self.n_fft//6, hop_length=self.hop_length//6, win_length=len(self.window_8k))
        self.istft_8k = iSTFT(n_fft=self.n_fft//6, hop_length=self.hop_length//6, win_length=len(self.window_8k))

        self.register_buffer("window_16k", torch.sqrt(torch.hann_window(self.n_fft//3, periodic=False)), False)
        self.stft_16k = STFT(n_fft=self.n_fft//3, hop_length=self.hop_length//3, win_length=len(self.window_16k))
        self.istft_16k = iSTFT(n_fft=self.n_fft//3, hop_length=self.hop_length//3, win_length=len(self.window_16k))

        self.register_buffer("window_24k", torch.sqrt(torch.hann_window(self.n_fft//2, periodic=False)), False)
        self.stft_24k = STFT(n_fft=self.n_fft//2, hop_length=self.hop_length//2, win_length=len(self.window_24k))
        self.istft_24k = iSTFT(n_fft=self.n_fft//2, hop_length=self.hop_length//2, win_length=len(self.window_24k))

        self.register_buffer("window_32k", torch.sqrt(torch.hann_window(self.n_fft//3*2, periodic=False)), False)
        self.stft_32k = STFT(n_fft=self.n_fft//3*2, hop_length=self.hop_length//3*2, win_length=len(self.window_32k))
        self.istft_32k = iSTFT(n_fft=self.n_fft//3*2, hop_length=self.hop_length//3*2, win_length=len(self.window_32k))

        self.register_buffer("window_48k", torch.sqrt(torch.hann_window(self.n_fft, periodic=False)), False)
        self.stft_48k = STFT(n_fft=self.n_fft, hop_length=self.hop_length, win_length=len(self.window_48k))
        self.istft_48k = iSTFT(n_fft=self.n_fft, hop_length=self.hop_length, win_length=len(self.window_48k))

        self.window_selector = {8000: self.window_8k, 16000: self.window_16k, 24000: self.window_24k, 
                           32000: self.window_32k, 48000: self.window_48k}
        self.stft_selector = {8000: (self.stft_8k, self.istft_8k), 16000: (self.stft_16k, self.istft_16k),
                         24000: (self.stft_24k, self.istft_24k), 32000: (self.stft_32k, self.istft_32k),
                         48000: (self.stft_48k, self.istft_48k)}
        self.special_sr_map = {22050: 24000, 44100: 48000}
                      
        self.ppFlag         = model_cfg.ppFlag
        if self.ppFlag == 1:
            from .SMRU_SFI import SMRUNet
            self.ppModel = SMRUNet(model_cfg)
        elif self.ppFlag == 2:
            from SMRU_16k_causal_mapping_v2 import SMRUNet
            self.ppModel = SMRUNet(model_cfg)
        elif self.ppFlag == 3:
            from SMRU_SFI_addBlocks import SMRUNet
            self.ppModel = SMRUNet(model_cfg)
        elif self.ppFlag == 4:
            from SMRU_SFI_16Blocks import SMRUNet
            self.ppModel = SMRUNet(model_cfg)
        elif self.ppFlag == 5:
            from SMRU_SFI_22Blocks import SMRUNet
            self.ppModel = SMRUNet(model_cfg)
        else:
            raise NotImplementedError("???")

    def forward(self, inputs, orig_sr):
        # inputs B T
        # import pdb; breakpoint()

        wav_len = inputs.shape[-1]
        process_sr = orig_sr
        if process_sr in self.special_sr_map:
            process_sr = self.special_sr_map[process_sr]

        this_window = self.window_selector[process_sr].to(inputs.device)
        mix_stft = self.stft_selector[process_sr][0](inputs, this_window)# mic_stft  [B, F, T]
        # import pdb; breakpoint()
        est_stft = self.ppModel(mix_stft, process_sr)
        est = self.stft_selector[process_sr][1](est_stft, wav_len, this_window)   # [B,T]

        return est, process_sr, orig_sr
    

class test_feat_cfg():
    def __init__(self):
        self.n_fft = 1536
        self.hop_length = 384
        self.segment = 1
        self.sr = 48000
        self.stft_window = 'hann'

class test_model_cfg():

    def __init__(self):
        self.ppFlag = 1
        self.num_repeat = 6
        self.input_feat_dim = 2
        self.feat_emb_dim = 192

def prepare_input(input_size):
    mix = torch.randn(input_size).to("cpu")
    return {'inputs': mix, 'orig_sr':48000}

def print_time_paramter_complexity(net, input_size):
    from ptflops import get_model_complexity_info
    print(input_size)
    macs, params = get_model_complexity_info(
        net, input_size, 
        as_strings=True, 
        print_per_layer_stat=True, 
        verbose=True,
        input_constructor=prepare_input)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def net_backward():
    """
    The whole pipeline give a latency of aec_shift + net win = 48ms
    """
    feat_cfg = test_feat_cfg()
    model_cfg = test_model_cfg()
    net_causal = SMRUse_SFI(feat_cfg, model_cfg)

    d = torch.device('cpu')
    noisy_wavs = torch.randn([1,16000])
    net_causal = net_causal.to(d)
    noisy_wavs =  noisy_wavs.to(d)
    net_causal(noisy_wavs).squeeze().mean().backward()
    print("check backward successfully")


def run_onehundred_times():
    from tqdm import tqdm
    import time
    torch.set_num_threads(1)
    feat_cfg = test_feat_cfg()
    model_cfg = test_model_cfg()
    net_causal = SMRUse_SFI(feat_cfg, model_cfg).to("cpu")
    net_causal = net_causal.eval()
    noisy_inp = torch.randn([1,48000], dtype=torch.float32).to("cpu")
    with torch.no_grad():
        start = time.time()
        for i in tqdm(range(30)):
            out1 = net_causal(noisy_inp, 48000)
    end = time.time()
    print("RTF:{}".format((end-start)/30.))


if __name__ == "__main__":
    # local test
    import torch
    feat_cfg = test_feat_cfg()
    model_cfg = test_model_cfg()
    model = SMRUse_SFI(feat_cfg, model_cfg).to("cpu")
    print_time_paramter_complexity(model, (1, 48000))
    # net_backward()
    run_onehundred_times() 