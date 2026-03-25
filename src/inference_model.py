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
import Transfromer_Decoder_Arch


class inference_model():
    def __init__(self,constants_class: constants):
        self.constants = constants_class()
        self.config = readYaml(self.constants.CONFIG_FILE_PATH)
        self.modelInferenceConfig = self.config["inference"]
        self.params = readYaml(self.constants.PARAMS_FILE_PATH)
        self.architecture = self.params["architecture"]
        self.hyperParameters = self.params["hyperParameters"]
        self.architecture = Transfromer_Decoder_Arch()
        self.model = self.architecture.decoderNextWordPrediction(
            self.architecture["vocab_size"],self.architecture["d_model"],
            self.architecture["block_size"],
            self.architecture["heads"],
            self.architecture["layers"],
            self.architecture["dropRate"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.modelInferenceConfig["tokenizer"])
        self.state_dict = torch.load("artifacts/model_trainer/model.pth", map_location=self.hyperParameters["device"])
        self.model = self.model.load_state_dict(self.state_dict)

    
    

    def inference(self,arabicInputText,inputTextLength):
        self.model.eval() 
        context_length = self.architecture["block_size"]
        idx = torch.tensor(self.tokenizer.encode(arabicInputText)).unsqueeze(0).to(self.hyperParameters["device"])

        with torch.no_grad():
            for _ in range(inputTextLength): 
            
                idx_cond = idx[:, -context_length:] 
                
                
                logits = self.model(idx_cond)
                
                
                last_token_logits = logits[:, -1, :] 
                
            
                next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)
                
            
                idx = torch.cat((idx, next_token), dim=1)

        print(self.tokenizer.decode(idx[0].tolist()))