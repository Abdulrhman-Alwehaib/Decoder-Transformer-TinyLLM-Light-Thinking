from utility import *
import constants
import os
import pandas as pd
import numpy as np
import yaml
import re
from transformers import AutoTokenizer
import torch


class dataTransformation():
    def __init__(self,constants_class: constants):
        self.constants = constants_class()
        self.config = readYaml(self.constants.CONFIG_FILE_PATH)
        self.dataTransfromationConfig = self.config["data_transfromation"]
        self.dataFrame = pd.read_csv(self.dataTransfromationConfig["inputData"])
        

    def filter_arabic_only(self,text: str):
        if not isinstance(text, str):
            return ""
        return re.sub(r'[^\u0600-\u06FF\s0-9]', '', text)
    
    def clean_anomalies(self):
        self.dataFrame = self.dataFrame.dropna()
        self.dataFrame = self.dataFrame.drop_duplicates()
        self.dataFrame = self.dataFrame.astype(str)
        self.dataFrame[self.dataTransfromationConfig["targetColumns"]] = self.dataFrame[self.dataTransfromationConfig["targetColumns"]].apply(self.filter_arabic_only)
    
    def is_valid_line(self,text: str, min_words=20):
        if len(text.split()) < min_words:
            return False
        return True
    
    def normalizationForTinyLLM(self,text: str):
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'ـ', '', text)
        text = re.sub(r'[\u064B-\u065F]', '', text)
        text = re.sub(r'[أإآ]', 'ا', text)
        numbersMapping = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
        text = text.translate(numbersMapping)
        text = text.replace(',', '،').replace('?', '؟')
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def tokenizationObject(self,seriesObj: pd.Series):
        tokenizerGPT = AutoTokenizer.from_pretrained("gpt2")

        customTokenizer = tokenizerGPT.train_new_from_iterator(seriesObj, vocab_size=16000,new_special_tokens=["<QUESTION>","<ANSWER>","<|endoftext|>","[PAD]"],
                                                            initial_alphabet=[])

        customTokenizer.add_special_tokens({'pad_token': '[PAD]'})

        customTokenizer.save_pretrained(self.dataTransfromationConfig["tokenizer"])

        return customTokenizer

    
    def transformation(self):
        self.clean_anomalies()
        cleanedText = []
        for i in range(len(self.dataFrame)):
            if self.is_valid_line(self.dataFrame[self.dataTransfromationConfig["targetColumns"]].iloc[i]):
                text = self.normalizationForTinyLLM(self.dataFrame[self.dataTransfromationConfig["targetColumns"]].iloc[i]) + " <|endoftext|> "
                cleanedText.append(text)
        newDF = pd.DataFrame({self.dataTransfromationConfig["targetColumns"]: cleanedText})

        
        tokenizer = self.tokenizationObject(newDF[self.dataTransfromationConfig["targetColumns"]])

        full_tokens = tokenizer(newDF[self.dataTransfromationConfig["targetColumns"]], truncation=False, padding=False)["input_ids"]

        flattened_ids = [token for sequence in full_tokens for token in sequence]
        block_size = 512
        total_len = (len(flattened_ids) // block_size) * block_size 
        chunks = [flattened_ids[i : i + block_size] for i in range(0, total_len, block_size)]

        IDs = torch.tensor(chunks)
        masks = torch.ones_like(IDs)

        torch.save(IDs,self.dataTransfromationConfig["inputIDS"])
        torch.save(masks,self.dataTransfromationConfig["InputMasks"])

    def execute(self):
        self.transformation()




        

        
    
