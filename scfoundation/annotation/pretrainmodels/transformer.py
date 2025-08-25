# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited


import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_layer import TransformerEncoderLayer

class pytorchTransformerModule(nn.Module):
    def __init__(self,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 ff_mult=4,
                 norm_first=False,
                 use_prompt= False,
                 num_tokens=64,
                 prompt_type= 'encoder-prompt',
                 n_layers_conf =None,
                 mlp_adapter_conf=None,
                 space_adapter_conf=None
                 ):
        super(pytorchTransformerModule, self).__init__()

        self.max_seq_len = max_seq_len
        self.depth = depth

        self.use_prompt = use_prompt
        self.prompt_type = prompt_type
        self.num_tokens = num_tokens
        self.n_layers_conf = n_layers_conf
        self.mlp_adapter_conf = mlp_adapter_conf
        self.space_adapter_conf = space_adapter_conf

        layers = []
        for i in range(depth):
            # layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
            #                                          dim_feedforward=dim * ff_mult,
            #                                          batch_first=True,
            #                                          norm_first=norm_first,
            #                                          #activation="gelu",
            #                                          ))

            layers.append(TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                     dim_feedforward=dim * ff_mult,
                                                     batch_first=True,
                                                     norm_first=norm_first,
                                                     # activation="gelu",
                                                     dropout = 0,
                                                      n_layers_conf=self.n_layers_conf,
                                                      mlp_adapter_conf=self.mlp_adapter_conf,
                                                      space_adapter_conf=self.space_adapter_conf,
                                                      use_prompt=self.use_prompt,
                                                      prompt_type=self.prompt_type,
                                                      num_tokens=self.num_tokens,
                                                      block_number=i,
                                                     ))

        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask):
        b, n, _, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # x get encodings [B, N, D] , batch_first is True
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask) # , src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # x = self.transformer_encoder(x)
        x = self.norm(x)

        return x
