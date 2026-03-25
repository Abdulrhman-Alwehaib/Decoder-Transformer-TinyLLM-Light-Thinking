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
from pipline.piplinetraining import runPipline


if __name__ == "__main__":
    constantsOBJ = constants()
    pipline = runPipline(constantsOBJ)

    pipline.dataIngestionExecute()
    pipline.dataValidationExecute()
    pipline.dataTransformationExecute()
    pipline.modelTrainerExecute()

    


