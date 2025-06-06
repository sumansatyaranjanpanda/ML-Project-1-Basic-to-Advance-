import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
import pandas as pd

### ✅ This class defines where your files will be saved:'artifacts' folder

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

##✅ This class will use the above paths and do all the work.
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\DATA\StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    # Step 1: Ingest data
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Transform data
    data_transformation_obj = DataTransformation()
    train_array, test_array, _ = data_transformation_obj.initiate_data_transformation(train_data, test_data)

    # Step 3: Train model
    model_trainer = ModelTrainer()
    result = model_trainer.initiate_model_trainer(train_array, test_array)  ## result  like={"best_model_name": RandomForest,"best_model_score": 0.99,"model_path":"\\\\\"}

    # Step 4: Print Results
    print("✅ Model Training Completed")
    print("Best Model:", result["best_model_name"])
    print("R2 Score:", result["best_model_score"])
    print("Model saved at:", result["model_path"])

        
