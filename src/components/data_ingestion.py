import sys 
import os 
import pandas as pd 
##project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
##sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts','train.csv')
    test_data_path :str = os.path.join('artifacts','test.csv')
    raw_data_path :str = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enterd data ingestion component")
        try:
            df = pd.read_csv('D:\\DataScience\\Projects\\MathScore-Predictor\\notebook\\data\\stud.csv')
            logging.info('Read the dataset as dataframe')


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
        
            logging.info("Train test split initiated")

            train_df,test_df = train_test_split(df,test_size=0.2,random_state=47)

            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data is completed")
            return (
                 self.ingestion_config.train_data_path,
                 self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    