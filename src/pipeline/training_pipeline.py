from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer






if __name__ == "__main__":
    dataingestion = DataIngestion()
    train_data_path,test_data_path = dataingestion.initiate_data_ingestion()
    datatansformation = DataTransformation()
    train_data,test_data,_ = datatansformation.initiate_data_transformation(train_data_path,test_data_path)
    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_training(train_data,test_data)