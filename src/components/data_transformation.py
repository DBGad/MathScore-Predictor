import sys 
import os 


import pandas as pd 
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    Preprocessor_obj_path = os.path.join('artifacts','Preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.Data_Transformation_Config = DataTransformationConfig()


    def Get_Transformer_Object(self):
        logging.info("Started Get Transformer Function")
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipline = Pipeline(
                steps=[('Impoute',SimpleImputer(strategy='median'))
                       ,("StandardScaler",StandardScaler())]
            ) 
            categorical_pipline = Pipeline(
                steps=[('Impoute',SimpleImputer(strategy='most_frequent')),
                       ('OneHotEncoder',OneHotEncoder()),
                       ("StandardScaler",StandardScaler(with_mean=False))]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([("numerical_pipline",numerical_pipline,numerical_columns),
                                              ('categorical_pipline',categorical_pipline,categorical_columns)])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        logging.info("Started data transformation")
        try:
            train_df =pd.read_csv( train_path)
            test_df = pd.read_csv(test_path)

            Preprocessor_obj = self.Get_Transformer_Object()
            target_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_name],axis=1)
            target_feature_train_df = train_df[target_name]

            input_feature_test_df = test_df.drop(columns=[target_name],axis=1)
            target_feature_test_df = test_df[target_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = Preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = Preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
        
            save_object(
                file_path=self.Data_Transformation_Config.Preprocessor_obj_path,
                obj=Preprocessor_obj)

            logging.info(f"Saved preprocessing object.")

            return (train_arr,test_arr,self.Data_Transformation_Config.Preprocessor_obj_path) 

        except Exception as e:
            raise CustomException(e,sys)