import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging

import os
from src.MLProject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            
            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])
            
            logging.info(f"Numerical columns:{numerical_columns}")
            logging.info(f"Categorical columns:{categorical_columns}")
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                    
                    
                ]
            )
            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
         try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
        
            logging.info("Reading the train and test file")
            
            preprocessing_obj=self.get_data_transformer_obj()
            
            target_column_name="math_score"
            numerical_columns=['writing_score','reading_score']
            
            #Dividing the train data into Independent and dependent variable
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df=train_df[target_column_name]
            
            #Dividing the test data into Independent and dependent variable
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df=test_df[target_column_name]
            
            logging.info("Applying preprocessing to train and test dataset")
            
            input_train_feature_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_test_feature_arr=preprocessing_obj.transform(input_features_test_df)
            
            
            train_arr=np.c_[input_train_feature_arr,np.array(target_features_train_df)]
            test_arr=np.c_[input_test_feature_arr,np.array(target_features_test_df)]
            
            logging.info(f"Saved Preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
            
            
         except Exception as e:
            raise CustomException(e,sys)