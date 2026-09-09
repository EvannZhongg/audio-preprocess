import torch
import torch.nn as nn
import numpy as np


class ResRNN(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 residual: bool = True,
                 causal: bool = False,
                 is_cln = False
                 ):
        super(ResRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.residual = residual

        self.eps = torch.finfo(torch.float32).eps
        self.is_cln = is_cln
        if is_cln:
            self.norm = CumulativeLayerNorm1d(input_size)
        else:
            self.norm = nn.LayerNorm(input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)
        # self.rnn = nn.GRU(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)
        self.proj = nn.Linear(hidden_size * (int(not causal) + 1), input_size)

    def forward(self, input):
        # input: B_, C, T
        # import pdb; breakpoint()
        if self.is_cln:
            rnn_output, _ = self.rnn(self.norm(input).transpose(-2, -1))
        else:
            rnn_output, _ = self.rnn(self.norm(input.transpose(-2, -1)))
        rnn_output = self.proj(rnn_output).transpose(-2, -1).contiguous()

        if self.residual:
            return input + rnn_output
        else:
            return rnn_output
        
class ResMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.norm = nn.GroupNorm(1, dim, torch.finfo(torch.float32).eps)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, kernel_size=1)

    def forward(self, x):
        """
        x: (B*nB, C, T)
        return: (B*nB, C, T)
        """
        B_, C, T = x.shape
        residual = x

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = residual + x
        
        return x

class VariableRateBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 factor: int = 2,
                 causal: bool = False,
                 num_layer: int = 1,
                 is_cmix = True
                 ):
        super(VariableRateBlock, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.factor = factor
        self.causal = causal
        self.is_cmix = is_cmix

        self.band_rnn = []
        for _ in range(num_layer):
            self.band_rnn.append(ResRNN(input_channels, hidden_channels, residual=True, causal=causal))
        self.band_rnn = nn.Sequential(*self.band_rnn)
        if is_cmix:
            self.channelmix = ResMLP(input_channels, input_channels*2)

        self.band_comm = ResRNN(input_channels, hidden_channels//2, residual=True, causal=False, is_cln=False)
    
    def forward(self, input):
        """
        input: (B, nband, C, T)
        return: (B*nband, C, T)
        """
        B, nB, C, T = input.shape
        input = input.view(-1,C,T)
        band_output = self.band_rnn(input)

        # channel_mixing
        if self.is_cmix:
            band_output = self.channelmix(band_output)

        # band_shuffle
        band_output = band_output.view(B, nB, C, -1).transpose(1, 3).contiguous()
        x = self.band_comm(band_output.view(-1, C, nB)).view(-1, T, C, nB).transpose(1, 3).contiguous()
        return x


class SMRUForLoopSM(nn.Module):
    def __init__(self, isdown, input_feat_dim, num_repeat, feat_emb_dim,
                    sr=48000, win=1536, stride=768
                 ):
        super(SMRUForLoopSM, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.isdown = isdown
        self.feat_emb_dim = feat_emb_dim
        self.input_feat_dim = input_feat_dim
        self.num_repeat = num_repeat
        # self.eps = torch.finfo(torch.float32).eps
        self.enc_dim = self.win // 2 + 1

        # 0-1k (200 hop), 1k-4k (500 hop), 4k-8k (1k hop)
        bandwidth_125 = int(np.floor(125 / (sr / 2.) * self.enc_dim)) #1k 8band
        bandwidth_250 = int(np.floor(250 / (sr / 2.) * self.enc_dim)) #4k 20band
        bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim)) #8k 28band
        bandwidth_1000 = int(np.floor(1000 / (sr / 2.) * self.enc_dim)) #16k 36band
        bandwidth_2000 = int(np.floor(2000 / (sr / 2.) * self.enc_dim)) #24k 40band
        self.band_width = [bandwidth_125] * 8
        self.band_width += [bandwidth_250] * 12
        self.band_width += [bandwidth_500] * 8
        self.band_width += [bandwidth_1000] * 8
        self.band_width += [bandwidth_2000] * 3
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        self.band_width[0] = self.band_width[0]+1
        self.band_width[-1] = self.band_width[-1]-1
        print(self.band_width)

        self.band_selector = {}
        self.band_selector[8000] = self.band_width[:20]
        self.band_selector[16000] = self.band_width[:28]
        self.band_selector[24000] = self.band_width[:32]
        self.band_selector[32000] = self.band_width[:36]
        self.band_selector[48000] = self.band_width[:40]
        # import pdb; breakpoint()
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(
                                         nn.GroupNorm(1, self.band_width[i] * self.input_feat_dim, torch.finfo(torch.float32).eps),
                                         nn.Conv1d(self.band_width[i] * self.input_feat_dim, self.feat_emb_dim, 1)
                                         )
                           )

        self.en0 = VariableRateBlock(feat_emb_dim, feat_emb_dim)
        self.en1 = VariableRateBlock(feat_emb_dim, feat_emb_dim)
        self.en2 = VariableRateBlock(feat_emb_dim, feat_emb_dim)
        self.de2 = VariableRateBlock(feat_emb_dim, feat_emb_dim)
        self.de1 = VariableRateBlock(feat_emb_dim, feat_emb_dim)
        self.de0 = VariableRateBlock(feat_emb_dim, feat_emb_dim)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(nn.Sequential(
                                           nn.GroupNorm(1, self.feat_emb_dim, torch.finfo(torch.float32).eps),
                                           nn.Conv1d(self.feat_emb_dim, self.feat_emb_dim * 2, 1),
                                           nn.PReLU(),
                                           nn.Conv1d(self.feat_emb_dim * 2, self.feat_emb_dim * 2, 1),
                                           nn.PReLU(),
                                           nn.Conv1d(self.feat_emb_dim * 2, self.band_width[i] * self.input_feat_dim * 2, 1)
                                           )
                             )

    def forward(self, input_feat, input_sr):
        # input_feat shape: (B, F, T, 2*4)
        this_band_width = self.band_selector[input_sr]
        this_nband = len(this_band_width)

        batch_size, F, T, input_dim = input_feat.shape
        muti_dim = input_dim
        input_feat = input_feat.permute(0, 3, 1, 2).contiguous()
        input_muti = input_feat.clone()

        subband_spec_RI = []
        subband_spec_muti = []
        subband_feature = []
        band_idx = 0
        for i in range(len(this_band_width)):
            subband_spec_RI.append(input_feat[:, :, band_idx:band_idx + this_band_width[i]].contiguous())
            subband_spec_muti.append(input_muti[:, :, band_idx:band_idx + this_band_width[i]].contiguous())
            subband_feature.append(self.BN[i](subband_spec_RI[i].view(batch_size, this_band_width[i] * input_dim, -1)))
            band_idx += this_band_width[i]
        subband_feature = torch.stack(subband_feature, 1)


        e_sep_out0 = self.en0(subband_feature)
        e_sep_out1 = self.en1(e_sep_out0)
        e_sep_out2 = self.en2(e_sep_out1)
        d_sep_out2 = self.de2(e_sep_out2)
        d_sep_out1 = self.de1(d_sep_out2)
        d_sep_out0 = self.de0(d_sep_out1)

        sep_output = d_sep_out0.view(batch_size, this_nband, self.feat_emb_dim, -1)

        sep_subband_spec = []
        for i in range(len(this_band_width)):
            this_output = self.mask[i](sep_output[:, i])
            this_output = this_output.view(batch_size, 2, muti_dim, this_band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # B, muti_dim, BW, T
            this_mask = this_mask.permute(0,2,3,1).contiguous() # B,F,T,C
            this_mask = this_mask.reshape([batch_size, this_band_width[i], T, muti_dim // 2, 2])
            this_mask = torch.view_as_complex(this_mask)          # B, bw, T, 4

            sep_subband_spec.append(this_mask.squeeze(-1))
        est_spec = torch.cat(sep_subband_spec, 1)  # B, F, T

        return est_spec

class SMRUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.isdown = False
        self.input_feat_dim = cfg.input_feat_dim
        self.num_repeat = cfg.num_repeat
        self.feat_emb_dim = cfg.feat_emb_dim

        self.smru = SMRUForLoopSM(self.isdown, self.input_feat_dim, self.num_repeat, self.feat_emb_dim)


    def forward(self, nsy_stft, input_sr):
        nsy_stft = nsy_stft.permute(0,2,1)      # [B, T, F]
        nsy_spec_decoupled    = torch.view_as_real(nsy_stft).permute(0,2,1,3)          # [B,F,T,2]
        spec_clean          = self.smru(nsy_spec_decoupled, input_sr) 
        
        return spec_clean
    

class test_model_cfg():

    def __init__(self):
        self.ppFlag = 1     
        self.num_repeat = 6
        self.input_feat_dim = 2
        self.feat_emb_dim = 192
        self.isdown = False



def prepare_input(input_size):
    """
        input_size: including batch_size.
        For threeD, input_size = [(2, 3, 80, 192, 160), (2, 1, 80, 192, 160)]
    """
    x1 = torch.ones(input_size, dtype=torch.cfloat)

    return {'nsy_stft': x1, 'input_sr':48000}

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


def checkcausal_net():
    model_cfg = test_model_cfg()
    net_causal = SMRUNet(model_cfg)
    net_causal = net_causal.eval()
    noisy_inp = torch.randn([1,769,100], dtype=torch.complex64)
    with torch.no_grad():
        out1 = net_causal(noisy_inp[[0],:,:], 48000)
        for i in range(0,10):
            noisy_inp2 = noisy_inp.clone()
            noisy_inp2[:,1,i:] = 1000 + torch.rand_like(noisy_inp2[:,1,i:])
            out2 = net_causal(noisy_inp2[[0],:,:], 48000)
            print(out2.shape)
            print((((out1-out2).abs()).sum(0).sum(0)>1e-3).float().argmax())
            print(((out1-out2).abs()).sum(0).sum(0))

def net_backward():
    model_cfg = test_model_cfg()
    net_causal = SMRUNet(model_cfg)
    d = torch.device('cuda')
    noisy_wavs = torch.randn([1,161,10], dtype=torch.complex64)
    net_causal = net_causal.to(d)
    noisy_wavs =  noisy_wavs.to(d)
    net_causal(noisy_wavs, noisy_wavs, noisy_wavs, noisy_wavs).squeeze().mean().backward()
    print("check backward successfully")

if __name__ == '__main__':
    model_cfg = test_model_cfg()
    model = SMRUNet(model_cfg)
    model = model.eval()
    print_time_paramter_complexity(model, (1,769,63))
    # checkcausal_net()
    # net_backward() 