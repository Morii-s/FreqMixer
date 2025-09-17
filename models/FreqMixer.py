import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import pywt
from einops import rearrange
from utils.RevIN import RevIN
from math import sqrt


class WaveletCrossMLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, wavelet_levels, d_model, patch_num, dropout=0.1, d_factor=2, p_factor=2):
        super(WaveletCrossMLPModel, self).__init__()
        
        assert input_dim == output_dim
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_ff = input_dim * d_factor
        self.wavelet_levels = wavelet_levels
        self.patch_num = patch_num
        
        self.cross_attentions = nn.ModuleList([
            CrossAttention(d_model=d_model, num_heads=8, dropout=dropout)
            for _ in range(wavelet_levels)
        ])
        
        self.channel_mixing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=self.d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_features=self.d_ff, out_features=input_dim),
            )
        ])

        self.dropout = nn.Dropout(dropout)

        self.wavelet_level_enc = nn.Parameter(torch.zeros(1, wavelet_levels, 1, d_model))
        nn.init.xavier_uniform_(self.wavelet_level_enc)
        
    def forward(self, x):
        # [B*M, wavelet_levels, num_patch, d_model]
        batch_size, wavelet_levels, num_patch, d_model = x.shape

        wavelet_features = []
        for i in range(wavelet_levels):
            wavelet_features.append(x[:, i])  # [B*M, num_patch, d_model]
        
        for i in range(wavelet_levels):
            wavelet_features[i] = wavelet_features[i] + self.wavelet_level_enc[:, i]
        
        attended_features = []
        for i in range(wavelet_levels):
            query = wavelet_features[i]  # [B*M, num_patch, d_model]
            
            other_levels = []
            for j in range(wavelet_levels):
                if j != i:
                    other_levels.append(wavelet_features[j])
            
            if other_levels:
                key_value = torch.cat(other_levels, dim=1)
                
                attn_output = self.cross_attentions[i](query, key_value)
                
                attended_feature = wavelet_features[i] + self.dropout(attn_output)
            else:
                attended_feature = wavelet_features[i]
                
            attended_features.append(attended_feature)
        
        x_combined = torch.stack(attended_features, dim=1)  # [B*M, wavelet_levels, num_patch, d_model]
        
        x_flat = rearrange(x_combined, 'b w n d -> b n (w d)')  # [B*M, num_patch, wavelet_levels*d_model]
        
        for channel_mixer in self.channel_mixing:
            x_flat = x_flat + self.dropout(channel_mixer(x_flat))

        x_output = rearrange(x_flat, 'b n (w d) -> b w n d', w=wavelet_levels, d=d_model)
        
        return x_output  # [B*M, wavelet_levels, num_patch, d_model]


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.m = configs.m
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.revin = RevIN(self.enc_in, eps=1e-5, affine = True, subtract_last = False)

        self.wavelet_embedding = WaveletEmbedding(
            d_channel=configs.enc_in,
            swt=True,
            requires_grad=False,
            wv=configs.wv if hasattr(configs, 'wv') else 'db1',
            m=configs.m if hasattr(configs, 'm') else 3
        )

        self.inverse_wavelet = WaveletEmbedding(
            d_channel=configs.enc_in,
            swt=False,
            requires_grad=False,
            wv=configs.wv if hasattr(configs, 'wv') else 'db1',
            m=configs.m if hasattr(configs, 'm') else 3
        )

        self.p_ff = self.patch_num * getattr(configs, 'p_factor', 2)
        self.patch_mixing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.patch_num, self.p_ff),  
                nn.GELU(),
                nn.Dropout(getattr(configs, 'dropout', 0.1)),
                nn.Linear(self.p_ff, self.patch_num)  
            ) 
        ])


        self.mlp_model = WaveletCrossMLPModel(
            input_dim=self.d_model * (self.m + 1),  
            output_dim=self.d_model * (self.m + 1),
            wavelet_levels=self.m + 1, 
            d_model=self.d_model,     
            dropout=getattr(configs, 'dropout', 0.1),
            d_factor = getattr(configs, 'd_factor', 2),
            patch_num = self.patch_num
        )

        if self.task_name == 'long_term_forecast':
            self.in_layer = nn.Linear(configs.patch_size, self.d_model)
            self.out_layer = nn.Linear(int(self.d_model * (self.patch_num)), configs.pred_len)

            for layer in (self.mlp_model, self.in_layer, self.out_layer):
                layer.cuda()
                layer.train()

        self.freq_adaptive_patch = FrequencyAdaptivePatchEmbedding(
            seq_len=configs.seq_len,
            wavelet_levels=configs.m + 1,
            d_model=configs.d_model,
            base_patch_size=configs.patch_size,
            stride=configs.stride,
            num_heads=getattr(configs, 'num_heads', 8),
            dropout=getattr(configs, 'dropout', 0.1),
            patch_configs=getattr(configs, 'patch_configs', None)
        )
        
        self.patch_num = self.freq_adaptive_patch.target_patch_num

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :] # [B, L, D]
        return None

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):

        B, L, M = x_enc.shape
        x_enc = self.revin(x_enc, 'norm')

        x_enc = x_enc.permute(0, 2, 1)  # [B, M, L]
        x = self.wavelet_embedding(x_enc)
        B, M, wavelet_levels, L = x.shape
        x = rearrange(x, 'b m c l -> (b m) c l')


        freq_enhanced = self.freq_adaptive_patch(x.float())
        # freq_enhanced: [B*M, wavelet_levels, target_patch_num, d_model]
        
        patch_mlp_in = freq_enhanced.permute(0, 1, 3, 2)
        patch_mlp_out = patch_mlp_in
        for module in self.patch_mixing:
            patch_mlp_out = module(patch_mlp_out)
        
        enhanced_embedding = freq_enhanced + patch_mlp_out.permute(0, 1, 3, 2)
        

        mlp_output = self.mlp_model(enhanced_embedding)
        
        last_embedding = rearrange(mlp_output, 'b m n c -> b m (n c)')
        outputs = self.out_layer(last_embedding)
        
        outputs = rearrange(outputs, '(b m) c h -> b m c h', b=B, m=M)

        outputs = self.inverse_wavelet(outputs)
        outputs = rearrange(outputs, 'b m l -> b l m')
        outputs = self.revin(outputs, 'denorm')

        return outputs


class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients

        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)

            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)

    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]

        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")

            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2

        return approx_coeff
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value):
        batch_size = query.shape[0]
        
        q = self.query(query)  # [B, N, D]
        k = self.key(key_value)  # [B, M, D]
        v = self.value(key_value)  # [B, M, D]

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D/H]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D/H]
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D/H]
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [B, N, D]
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class FrequencyAdaptivePatchEmbedding(nn.Module):
    def __init__(self, seq_len, wavelet_levels, d_model, base_patch_size=16, stride=8, num_heads=8, dropout=0.1, patch_configs=None):
        super().__init__()
        self.seq_len = seq_len
        self.wavelet_levels = wavelet_levels
        self.d_model = d_model
        self.base_patch_size = base_patch_size
        self.stride = stride
        
        if patch_configs is not None:
            self.level_configs = self._parse_patch_configs(patch_configs)
        else:
            self.level_configs = self._design_patch_configs()
        
        self.patch_embeddings = nn.ModuleList()
        self.padding_layers = nn.ModuleList()
        
        for config in self.level_configs:
            self.patch_embeddings.append(
                nn.Linear(config['patch_size'], d_model)
            )
            padding_size = max(0, config['patch_size'] - config['stride'])
            self.padding_layers.append(
                nn.ReplicationPad1d((0, padding_size))
            )
        
        self.cross_attentions = nn.ModuleList([
            CrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(wavelet_levels)
        ])
        
        self.target_patch_num = min([config['patch_num'] for config in self.level_configs])
        self.patch_aligners = nn.ModuleList()
        
        for i, config in enumerate(self.level_configs):
            current_patch_num = config['patch_num']
            if current_patch_num != self.target_patch_num:
                self.patch_aligners.append(
                    HybridAligner(d_model, self.target_patch_num)
                )
                print(f"Level {i}: Alignment {current_patch_num} -> {self.target_patch_num}")
            else:
                self.patch_aligners.append(nn.Identity())
                print(f"Level {i}: No alignment required,patch_num={current_patch_num}")
        
        self.dropout = nn.Dropout(dropout)

        self.wavelet_level_enc = nn.Parameter(torch.zeros(1, wavelet_levels, 1, d_model))
        nn.init.xavier_uniform_(self.wavelet_level_enc)
    
    def _parse_patch_configs(self, patch_configs_str):
        configs = []
        config_pairs = patch_configs_str.split(';')
        
        for level, config_pair in enumerate(config_pairs):
            if level >= self.wavelet_levels:
                break
                
            patch_size, stride = map(int, config_pair.split(','))
            
            padding_size = max(0, patch_size - stride)
            
            effective_seq_len = self.seq_len + padding_size
            patch_num = (effective_seq_len - patch_size) // stride + 1
            
            configs.append({
                'level': level,
                'patch_size': patch_size,
                'patch_num': patch_num,
                'stride': stride,
                'padding_size': padding_size
            })
        
        while len(configs) < self.wavelet_levels:
            level = len(configs)
            patch_size = 8
            stride = 4
            
            padding_size = max(0, patch_size - stride)
            effective_seq_len = self.seq_len + padding_size
            patch_num = (effective_seq_len - patch_size) // stride + 1
            
            configs.append({
                'level': level,
                'patch_size': patch_size,
                'patch_num': patch_num,
                'stride': stride,
                'padding_size': padding_size
            })
        
        return configs
    
    def _design_patch_configs(self):
        configs = []   
        for level in range(self.wavelet_levels):
            if level == 0:
                patch_size = 16
                stride = 8  
            elif level == 1:
                patch_size = 16
                stride = 8   
            elif level == 2:
                patch_size = 12
                stride = 6  
            elif level == 3:
                patch_size = 12
                stride = 6  
            elif level == 4:
                patch_size = 12
                stride = 6  
            else:
                patch_size = 8
                stride = 4   

            padding_size = max(0, patch_size - stride)
            

            effective_seq_len = self.seq_len + padding_size
            patch_num = (effective_seq_len - patch_size) // stride + 1
            
            configs.append({
                'level': level,
                'patch_size': patch_size,
                'patch_num': patch_num,
                'stride': stride,
                'padding_size': padding_size
            })
        
        return configs
    
    def forward(self, x):
        """
        input: x [B*M, wavelet_levels, seq_len]
        output: [B*M, wavelet_levels, target_patch_num, d_model]
        """
        B_M, W, L = x.shape
        
        level_patches = []
        for level in range(W):
            level_data = x[:, level]  # [B*M, seq_len]
            config = self.level_configs[level]
            
            padded_data = self.padding_layers[level](level_data.unsqueeze(1)).squeeze(1)
            

            patches = padded_data.unfold(
                dimension=-1, 
                size=config['patch_size'], 
                step=config['stride']
            )  # [B*M, patch_num, patch_size]
            
            embedded = self.patch_embeddings[level](patches)  # [B*M, patch_num, d_model]
            
            embedded = embedded + self.wavelet_level_enc[:, level]
            
            level_patches.append(embedded)
        
        attended_patches = []
        for i in range(W):
            query = level_patches[i]  # [B*M, patch_num_i, d_model]
            
            other_levels = []
            for j in range(W):
                if j != i:
                    other_levels.append(level_patches[j])
            
            if other_levels:
                key_value = torch.cat(other_levels, dim=1)  # [B*M, sum(other_patch_nums), d_model]
                
                attn_output = self.cross_attentions[i](query, key_value)
                
                attended_patch = query + self.dropout(attn_output)
            else:
                attended_patch = query
            
            attended_patches.append(attended_patch)
        
        aligned_patches = []
        for i, patches in enumerate(attended_patches):
            aligned = self.patch_aligners[i](patches)
            aligned_patches.append(aligned)
        
        return torch.stack(aligned_patches, dim=1)  # [B*M, wavelet_levels, target_patch_num, d_model]

class HybridAligner(nn.Module):
    def __init__(self, d_model, target_num):
        super().__init__()
        self.target_num = target_num
        self.d_model = d_model
        
        self.feature_adjust = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self.position_adjust = nn.Linear(d_model, d_model)
        
    def forward(self, patches):
        # patches: [B*M, patch_num, d_model]
        B_M, patch_num, d_model = patches.shape
        
        if patch_num == self.target_num:
            return patches
        
        patches_t = patches.transpose(1, 2)  # [B*M, d_model, patch_num]
        aligned_t = F.interpolate(
            patches_t, 
            size=self.target_num, 
            mode='linear', 
            align_corners=True
        )  # [B*M, d_model, target_num]
        aligned = aligned_t.transpose(1, 2)  # [B*M, target_num, d_model]
        
        adjusted = self.feature_adjust(aligned)
        
        position_adjusted = self.position_adjust(aligned)
         
        final_aligned = aligned + self.residual_weight * (adjusted + position_adjusted)
        
        return final_aligned