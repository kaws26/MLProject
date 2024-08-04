import logging.config
import sys
from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
from src.MLProject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.MLProject.components.data_transformation import DataTransformation,DataTransformationConfig
from src.MLProject.components.model_training import ModelTrainerConfig,ModelTrainer

if __name__=="__main__":
    logging.info("The execution has started")
    try:
        #Data ingestion
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        
        #Data Transformation
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        
        #Model training
        model_trainer=ModelTrainer()
        r2sqr=model_trainer.initiate_model_trainer(train_arr,test_arr)
        print(r2sqr)
        
    except Exception as e:
        raise CustomException(e,sys)