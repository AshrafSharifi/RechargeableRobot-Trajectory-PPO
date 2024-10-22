import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau


def time_to_reach_POI():
    
    reach_time = dict()
    S1 = {'1_2': 15, '1_3': 90, '1_4': 30, '1_5': 75, '1_6': 45, '1_7': 60}
    reach_time['sensor1']= S1;
    
    S2 = {'2_1': 90, '2_3': 75, '2_4': 15 ,'2_5': 60, '2_6': 30, '2_7': 45}
    reach_time['sensor2']= S2;
     
    S3 = {'3_1': 15, '3_2': 30, '3_4':45  ,'3_5':90 , '3_6':60 , '3_7':75 }
    reach_time['sensor3']= S3;
    
    S4 = {'4_1':75 , '4_2':90 , '4_3':60  ,'4_5':45 , '4_6':15 , '4_7':30 }
    reach_time['sensor4']= S4;
    
    S5 = {'5_1':30 , '5_2':45 , '5_3':15  ,'5_4':60 , '5_6':75 , '5_7':90 }
    reach_time['sensor5']= S5;
    
    S6 = {'6_1':60 ,'6_2':75 , '6_3':45 , '6_4':90  ,'6_5':30 , '6_7':15 }
    reach_time['sensor6']= S6
    
    S7 = {'7_1': 45, '7_2':60 , '7_3':30  ,'7_4': 75, '7_5':15 , '7_6':90 }
    reach_time['sensor7']= S7;
   
    return reach_time;

charging_stations = [5,7,3]
initialChargingLevel = 300
reach_time = time_to_reach_POI()

data_header = [
    'sensor', 'complete_timestamp(YYYY_M_DD_HH_M)',
    'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
    'hour'
]
DB_file = "data/ICCAD/train_DB.csv"

df = pd.read_csv(DB_file)
df = df[data_header].values
df = pd.DataFrame(df)
current_batteryLevel = initialChargingLevel
counter = 0
all_counter = 0
# Iterate through the DataFrame with access to current and next row
for i, (index, row) in enumerate(df.iterrows()):
    if i < len(df) - 1:  # Check if next row exists
        next_row = df.iloc[i + 1]
        
        # for current measurement
        current_batteryLevel -= 15 
        
        # for travelling
        temp_reachtime = reach_time['sensor'+ str(row[0])][str(row[0])+'_'+str(next_row[0])]
        current_batteryLevel -= temp_reachtime
        
        if current_batteryLevel <= 15:
            current_batteryLevel = initialChargingLevel
            all_counter += 1
            if next_row[0] not in charging_stations:
                counter += 1
        
print(counter)
print(all_counter)
    
    
