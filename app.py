import logging.config
import sys
from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject.components.data_transformation import DataTransformation

if __name__=="__main__":
    logging.info("The execution has started")
    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        
        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    except Exception as e:
        raise CustomException(e,sys)