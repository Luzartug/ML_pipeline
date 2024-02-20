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
            model=load_object(model_path)
            
            data_scaled=preprocess_data(df_features)
            prediction=model.predict(data_scaled)
            return str(prediction)
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    data={
        "co": [420.57],
        "no": [1.93],
        "no2": [25.36],
        "o3": [20.56],
        "so2": [5.19],
        "pm2_5": [14.79],
        "pm10": [19.17],
        "nh3": [2.22]
    }
    df_feature=pd.DataFrame(data)
    
    inf=InferencePipeline()
    response=inf.inference(df_features=df_feature)
    print(type(response))
    
    