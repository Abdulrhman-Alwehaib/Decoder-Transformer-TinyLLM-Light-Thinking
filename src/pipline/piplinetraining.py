from components.data_ingestion import dataIngestion
from components.data_transfromation import dataTransformation
from components.data_validation import dataValidation
from components.model_trainer import modelTrainer
from components import Transfromer_Decoder_Arch
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

class runPipline():
    def __init__(self,constants_class: constants): 
        self.constants = constants_class()
        self.dataIngestionObj = dataIngestion(self.constants)
        self.datavalidationObj = dataValidation(self.constants)
        self.dataTransfromationObj = dataTransformation()
        self.modelTrainer = modelTrainer(self.constants)
        self.Transformer_arch_decoderOBJ = Transfromer_Decoder_Arch()
    
    def dataIngestionExecute(self):
        self.dataIngestionObj.execute()
    
    def dataValidationExecute(self):
        self.datavalidationObj.execute()
    
    def dataTransformationExecute(self):
        self.dataTransfromationObj.execute()
    
    def modelTrainerExecute(self):
        self.modelTrainer.execute()
    

    


    