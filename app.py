import logging.config
import sys
from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
from src.MLProject.components.data_ingestion import DataIngestion

if __name__=="__main__":
    logging.info("The execution has started")
    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise CustomException(e,sys)