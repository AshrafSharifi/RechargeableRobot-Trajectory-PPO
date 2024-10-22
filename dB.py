import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from functions import *
from tabulate import tabulate
from keras.layers import BatchNormalization
from variables import variables

class dB(object):
    def __init__(self,varobj):
        
        self.POIs = [1,2,4,6,7,5,3]
        
        self.func= functions()
        
        
        # with open('data/states_dict', 'rb') as fin:
        #     self.states_dict = pickle.load(fin)
        # fin.close()
        
        data = self.func.get_all_data()
        self.sensorsdata= data
        
        self.var_obj = varobj
        
        self.climatic_data = dict()
        

        sensorList = list(range(1,9))
        output = dict()
        for sensor in sensorList:
            fin_name = 'data/Sensor_data/sensor_t'+str(sensor)+'.csv' 
            self.climatic_data[sensor] = pd.read_csv(fin_name)
        
   


    def create_train_db(self,complete_paths):

        df_train =  pd.DataFrame(columns=list(self.climatic_data[1].columns) + ['sensor'])
        df_test =  pd.DataFrame(columns=list(self.climatic_data[1].columns) + ['sensor'])        
        item = next(iter(complete_paths.values()))
        from_sensor = item[0][0]
        from_date = item[1][0]
        indx = self.localize_row(self.climatic_data[from_sensor],from_date)
        df_train.loc[len(df_train)] = list(self.climatic_data[from_sensor].iloc[indx]) + [str(from_sensor)]
        df_test = self.add_test_equavalentrecord(from_sensor,from_date,df_test)
        
        for item in complete_paths.values():
            from_sensor = item[0][0]
            from_date = item[1][0]
            to_sensor = item[0][-1]
            to_date = item[2][2]
            to_battery = item[2][0]
            
            if to_battery == 300:
                # decrease time offset 30 which is measurement time + charging time
                reach_date = self.func.change_minutes(to_date,-30)
            else:
                # decrease time offset 15 which is measurement time
                reach_date = self.func.change_minutes(to_date,-15)
                
            indx = self.localize_row(self.climatic_data[to_sensor],reach_date)
            print(indx)
            if(indx==-1):
                print('found')
            df_train.loc[len(df_train)] = list(self.climatic_data[to_sensor].iloc[indx]) + [str(to_sensor)]
            
            # copy middle nodes of path into df_test
            middle_POIs = item[0][1:-1]
            df_test = self.add_test_middle_POI(middle_POIs,from_sensor,from_date,df_test)
        df_train.to_csv("data/DB/train_DB.csv", index=False, encoding='utf-8')
        df_test.to_csv("data/DB/test_DB.csv", index=False, encoding='utf-8')

            
            
            
    def localize_row(self,df,datetime_str):
        datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
        # Extract year, month, day, hour, and minute
        year = datetime_obj.year
        month = datetime_obj.month
        day = datetime_obj.day
        hour = datetime_obj.hour
        minute = datetime_obj.minute
     
        all_conditions = (df['hour'] == hour) & (df['month'] == month) & (df['day_of_month'] == day)
         # Check the value of A and select the corresponding row
        if minute == 0:
            condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '0')
        elif minute == 15:
            condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '1')
        elif minute == 30:
            condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '2')
        elif minute == 45:
            condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '3')
     
        indx = (df[all_conditions & condition].index)
        if indx.size == 1:
            return indx[0]
        else:
            return -1
        
    def add_test_equavalentrecord(self,from_sensor,from_date,df_test):
        sensor_range = [x for x in range(1,8) if x != from_sensor]
            
        for sensor in sensor_range:
            indx = self.localize_row(self.climatic_data[sensor],from_date)
            df_test.loc[len(df_test)] = list(self.climatic_data[sensor].iloc[indx]) + [str(sensor)]        
        return df_test
    
    def add_test_middle_POI(self,middle_POIs,from_sensor,from_date,df_test):
        for sensor in middle_POIs:
            dist = self.var_obj.shortest_paths_data[str(from_sensor)][str(sensor)]['distance']
            reach_time = self.func.change_minutes(from_date, dist)
            for oth_sensor in range(1,8):
                indx = self.localize_row(self.climatic_data[oth_sensor],reach_time)
                df_test.loc[len(df_test)] = list(self.climatic_data[oth_sensor].iloc[indx]) + [str(oth_sensor)]
        return df_test
                
        