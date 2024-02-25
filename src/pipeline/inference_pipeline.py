import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object, preprocess_data

class InferencePipeline:
    def __init__(self):
        pass
    
    def inference(self, df_features):
        try:
            model_path=os.path.join("data","model.pkl")
            scaler_path=os.path.join("data", "scaler.pkl")
            
            model=load_object(model_path)
            scaler=load_object(scaler_path)
            
            data_scaled=preprocess_data(df_features, scaler)
            print(data_scaled)
            prediction=model.predict(data_scaled)
            return prediction
                
        except Exception as e:
            raise CustomException(e, sys)

# if __name__=="__main__":
#     data={
#         "co": [3524.78],
#         "no": [223.52],
#         "no2": [68.55],
#         "o3": [0.00],
#         "so2": [44.82],
#         "pm2_5": [217.14],
#         "pm10": [236.65],
#         "nh3": [24.57]
#     }
#     df_feature=pd.DataFrame(data)
    
#     inf=InferencePipeline()
#     response=inf.inference(df_features=df_feature)
#     print(response)