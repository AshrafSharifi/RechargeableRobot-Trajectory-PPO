from functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import csv
import pickle
import PPO_Agent as ppo
import tyro

        
    

def create_states_dict():
    pd.options.mode.chained_assignment = None 
    
    sensorList = list(range(1,9))
    data_size = 0
    states_dict = dict()
    for sensor in sensorList:
        sensor_data = pd.read_csv("data/sensor_t" + str(sensor) + ".csv")
        
        # -------remove data of days which has small size
        grouped = sensor_data.groupby(['year','month','day_of_month'])
        group_sizes = grouped.size().reset_index(name='GroupSize')
        valid_groups = grouped.filter(lambda x: len(x) == 96)
        indices_to_keep = valid_groups.index
        sensor_data = sensor_data.loc[indices_to_keep]
        # ----------------------------------------------
        
        sensor_dict = dict()
        
        sensor_dict["years"] = (sensor_data['year'].unique())
        num_year = sensor_dict["years"].shape[0]
        
        for year in sensor_dict["years"]:
            # Filter data based on the specified date
            
            idx_y = (sensor_data['year'] == year)
            filtered_data_step1 = sensor_data.loc[idx_y,:]

            sensor_dict["months"] = (filtered_data_step1['month'].unique())
            num_month = sensor_dict["months"].shape[0]
            
            for month in sensor_dict["months"]:
                
                idx_m = (filtered_data_step1['month'] == month) 
                filtered_data_step2 = filtered_data_step1.loc[idx_m,:]
                
                sensor_dict["days"] = (filtered_data_step2['day_of_month'].unique())
                num_day = (filtered_data_step2['day_of_month'].unique()).shape[0]
                
                for day in sensor_dict["days"]:
                    
                    idx_d = (filtered_data_step2['day_of_month'] == day)
                    filtered_data = filtered_data_step2.loc[idx_d,:]
                    
                    
        data_size += (num_year * num_month * num_day)
        states_dict[sensor] = sensor_dict 
        with open('data/states_dict', 'wb') as fout:
            pickle.dump(aggregated_data, fout)
        fout.close()

if __name__ == '__main__':
    #create_states_dict()
    fin_name = 'data/states_dict'
    with open(fin_name, 'rb') as fin:
        states_dict = pickle.load(fin)
    fin.close() 
    key = list(states_dict.keys())[0]
    [y,m] = key.split('_')
    d = (states_dict[key])[0]
    initial_state = [1 ,int(y), int(m), d, 0, 0]
    ppo.train(None,None,states_dict,True)
                
    
    
    
    
    
    
    
    