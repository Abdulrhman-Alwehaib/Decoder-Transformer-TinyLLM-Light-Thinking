from utility import *
import constants
import os
import pandas as pd
import numpy as np
import yaml


class dataValidation:
    def __init__(self,constants_class:constants):
        self.constants = constants_class()
        self.config = readYaml(self.constants.CONFIG_FILE_PATH)
        self.dataValidationConfig = self.config["data_validation"]
        self.dataFrame = pd.read_csv(self.dataValidationConfig["inputData"])

    def createYamlFile(self,confitmationFile,confirmation):
        with open(confitmationFile,"w") as f:
            yaml.dump(confirmation,f)

    def validating(self):
        validation = True
        dataFrameColumns = self.dataFrame.columns
        requiredColumns = self.dataValidationConfig["required_couloumns"]
        for column in requiredColumns:
            if column not in dataFrameColumns:
                validation = False
                print(f"Missing Required Column {column}")
        
        confirmation = {
            "confirmed_validation": validation
        }
        validationFile = self.dataValidationConfig["confirmation_file"]
        self.createYamlFile(validationFile,confirmation)

    def execute(self):
        self.validating()
    

