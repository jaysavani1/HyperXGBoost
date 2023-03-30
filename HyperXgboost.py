import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


class HyperXGBoost:
    
    def __init__(self, model_params: dict = {}):
        
        self.model_params = model_params
    
    def fit(self,):
        pass
    
    def predict(self,):
        pass
    
    def save(self,file_path):
        output_file = open(f"{file_path}xgboost_model.pickle", "w")
        pickle.dump(self,output_file)
    
    @classmethod
    def load(cls, file_path):
        input_file = open(f"{file_path}xgboost_model.pickle", "rb")
        return pickle.load(input_file)