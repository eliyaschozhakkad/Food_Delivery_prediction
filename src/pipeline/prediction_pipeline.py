import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from geopy.distance import geodesic

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):

        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
    

class CustomData():
    
    def __init__(self,Delivery_person_Age:float, Delivery_person_Ratings:float, Vehicle_condition:int,multiple_deliveries:float,
                 Restaurant_latitude:float,Restaurant_longitude:float,Delivery_location_latitude:float,Delivery_location_longitude:float,
                 Weather_conditions:str,Road_traffic_density:str,Type_of_vehicle:str,Festival:str,City:str):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.Restaurant_latitude = Restaurant_latitude
        self.Restaurant_longitude = Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City

    def get_data_as_dataframe(self):

        try:
            # Coordinates of two points
            point1 = (self.Restaurant_latitude, self.Restaurant_longitude)
            point2 = (self.Delivery_location_latitude, self.Delivery_location_longitude)

            # Calculate the distance between the points
            distance = geodesic(point1, point2).miles

            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City],
                'Distance':[distance]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
            
        except Exception as e:
            logging.info('Exception occured in get_data_as_dataframe')
            raise CustomException(e,sys)





