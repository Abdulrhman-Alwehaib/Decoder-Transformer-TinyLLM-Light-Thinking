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





class modelTrainer():
    def __init__(self,constants_class:constants):
        self.constants = constants_class()
        self.config = readYaml(self.constants.CONFIG_FILE_PATH)
        self.modelTrainerConfig = self.config["model_trainer"]
        self.params = readYaml(self.constants.PARAMS_FILE_PATH)
        self.architecture = self.params["architecture"]
        self.hyperParameters = self.params["hyperParameters"]
        self.proccessedData = torch.load(self.modelTrainerConfig["inputDataProccessed"])
        self.proccessedDataMASK = torch.load(self.modelTrainerConfig["inputDataMASKS"])
        transfromerDecoder = Transfromer_Decoder_Arch()

        self.model = transfromerDecoder.decoderNextWordPrediction(self.architecture["vocab_size"],self.architecture["d_model"],
                                  self.architecture["block_size"],
                                  self.architecture["heads"],
                                  self.architecture["layers"],
                                  self.architecture["dropRate"]
                                  )
        self.model = self.model.to(self.hyperParameters["device"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(),self.hyperParameters["learning_rate"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelTrainerConfig["tokenizer_path"])
        self.lossFN = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        split_idx = int(0.95 * len(self.proccessedData))
        train_ds = TensorDataset(self.proccessedData[:split_idx], self.proccessedDataMASK[:split_idx])
        val_ds = TensorDataset(self.proccessedData[split_idx:], self.proccessedDataMASK[split_idx:])

        self.train_loader = DataLoader(train_ds, batch_size=self.hyperParameters["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.hyperParameters["batch_size"])

    def trainer(self):
        scaler = torch.amp.GradScaler("cuda")
        for epoch in range(self.hyperParameters["epochs"]):
            self.model.train()
            
            for batchINDEX, (batch,masks) in enumerate(self.train_loader):
                batch = batch.to(self.hyperParameters["device"])
                
                masks = masks.to(self.hyperParameters["device"])
                
                inputs = batch[:,:-1]
                target = batch[:,1:]

                input_masks = (masks[:, :-1] == 0)

                with torch.amp.autocast("cuda"):
                    
                    logits = self.model(inputs,input_masks)
                    B, T, C = logits.shape
                    logits = logits.reshape(B * T, C)
                    target = target.reshape(B * T)
                    loss = self.lossFN(logits, target)
                    

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                
                
                
                if batchINDEX % 100 == 0:
                    print(f"Epoch {epoch+1} | Batch {batchINDEX} | Loss: {loss.item():.4f}")
            self.model.eval() 
            total_val_loss = 0
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for v_batch, v_masks in self.val_loader:
                    v_batch, v_masks = v_batch.to(self.hyperParameters["device"]), v_masks.to(self.hyperParameters["device"])
                    v_logits = self.model(v_batch[:,:-1], (v_masks[:,:-1] == 0))
                    v_loss = self.lossFN(v_logits.reshape(-1, v_logits.size(-1)), v_batch[:,1:].reshape(-1))
                    total_val_loss += v_loss.item()

            print(f"Epoch {epoch+1} , Val Loss: {total_val_loss/len(self.val_loader):.4f}")



        
        torch.save(self.model.state_dict(), self.modelTrainerConfig["trainedModel"])

    def execute(self):
        self.trainer()
        
        

     