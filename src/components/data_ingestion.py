from utility import *
import constants
import os
from datasets import Dataset, load_dataset

class dataIngestion:
    def __init__(self,constants_class: constants):
        self.constants = constants_class()
        self.config = readYaml(self.constants.CONFIG_FILE_PATH)
        self.dataIngestionConfig = self.config["data_ingestion"]
    def getData(self):
        os.makedirs(self.dataIngestionConfig["root_dir"],exist_ok=True)
        datasetsCount = 1
        datasetsLINKS = [self.dataIngestionConfig["source_hugging_face"]]
        datasetLINKSFeature = [self.dataIngestionConfig["specficType"]]
        files = self.dataIngestionConfig["fileDomain"]
        for i in range(datasetsCount):
            dataset = load_dataset(datasetsLINKS[i],
                                data_files=files,
                                verification_mode="no_checks",
                                )
            trainingSpecfic = dataset['train']
            datasetTEMP = trainingSpecfic.select_columns([datasetLINKSFeature[i]])
            df = datasetTEMP.to_pandas()
            df.to_csv(self.dataIngestionConfig["resulted_data_folder"],index=False,escapechar='\\')
    
    def execute(self):
        self.getData()


