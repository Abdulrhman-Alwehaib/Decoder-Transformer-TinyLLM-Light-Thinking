from utility import *
import constants
import os
import pandas as pd
import numpy as np
import yaml
import re
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import math



class Blocks(nn.Module):
    def __init__(self,d_model,n_head,dropoutRate):
        super().__init__()
        headsize = d_model // n_head
        self.sa_head = nn.MultiheadAttention(embed_dim=d_model,num_heads=n_head,batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model,4 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 4,d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropoutRate)

    def forward(self,x,mask=None,padding=None):
       
        att_out, _ = self.sa_head(self.ln1(x),self.ln1(x),self.ln1(x),
                             attn_mask=mask,
                             key_padding_mask=padding)
        x = x + self.dropout(att_out)

        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x
    
    class decoderNextWordPrediction(nn.Module):
        def __init__(self,vocab_size,d_model,block_size,n_heads,n_layers,dropRate):
            super().__init__()
            self.d_model = d_model
            self.toke_embedding_table = nn.Embedding(vocab_size,d_model)
            pe = torch.zeros(block_size,d_model)
            position = torch.arange(0,block_size,dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer("pe",pe)

            self.dropout = nn.Dropout(dropRate)

            self.blocks = nn.ModuleList([Blocks(d_model,n_heads,dropoutRate=dropRate) for _ in range(n_layers)])

            self.ln_final = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model,vocab_size)

            self.register_buffer("mask",torch.triu(torch.ones(block_size,block_size),diagonal=1).bool())

        def forward(self,idx,padding=None):
            B,T = idx.shape

            x = self.toke_embedding_table(idx) * math.sqrt(self.d_model)

            x = x + self.pe[:T,:]

            x = self.dropout(x)

            current_mask = self.mask[:T,:T]

            for block in self.blocks:
                x = block(x,mask=current_mask,padding=padding)

            x = self.ln_final(x)
            logits = self.lm_head(x)


            return logits
    