import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    Model_Trainer_Config = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.train_model_file_path = ModelTrainerConfig()

    def initiate_model_trainer (self,train_arr,test_arr) :
        logging.info("Start Model Training")
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info("Split training and test input data")

            models ={"Linear Regression":LinearRegression(),
                     "KNN":KNeighborsRegressor(),
                     "Decision Tree":DecisionTreeRegressor(),
                     "SVM":SVR(),
                     "Random Forest":RandomForestRegressor(),
                      "Ada Boost":AdaBoostRegressor() }
            

            models_report:dict = evaluate_model (X_train,y_train,X_test,y_test,models)

            best_score = max(sorted (models_report.values()))

            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_score)
            ]

            best_model = models[best_model_name]

            if best_score < 0.6:
                raise CustomException("cannot find best model")
            

            logging.info(f"Best found model on both training and testing dataset")

            save_object(file_path= self.train_model_file_path.Model_Trainer_Config,
                        obj=best_model)
            
            return best_score
        
        except Exception as e:
            raise CustomException(e,sys)