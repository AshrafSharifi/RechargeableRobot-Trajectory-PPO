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

class Environment:
    def __init__(self , env_id,reward_temperature_weight,reward_time_weight,reward_charging_weight,var_obj):
        # [3,1,1]
        # [5,.3,.3]
        self.env_id : int  = 0
        self.temperature_weight = reward_temperature_weight # Weight for maximizing temperature change
        self.time_weight = reward_time_weight  # Weight for minimizing time
        self.charging_weight = reward_charging_weight
        self.POIs = [1,2,4,6,7,5,3]
        travel_times = [15,30,45,60,75,90]
        self.func= functions()
        # self.reach_time = self.func.time_to_reach_POI()
        self.process_time = 15
        self.state_dim = 8
        self.action_dim = 1  
        self.min_temp = 0
        self.max_temp = 0
        self.min_time = 15
        self.max_time = 90
        self.mean_time = np.mean(travel_times)
        self.std_time = np.std(travel_times)
        
        # self.process_penalty = 0.5*(self.process_time / 60)
        self.process_penalty = 0.5*(self.process_time / 60)
        # self.process_penalty = self.process_time
        
        with open('data/states_dict', 'rb') as fin:
            self.states_dict = pickle.load(fin)
        fin.close()
        
        data = self.func.get_all_data()
        self.sensorsdata= data
        self.reach_time_minutes = 0
        self.var_obj = var_obj
   

    
    def get_random_state(self,state_dict):
        next_obs = [0] * 8
        next_obs[0] =random.choice(range(1,8))
        keys = list(state_dict.keys())
        random_key = random.choice(keys)
        [y,m] = random_key.split('_')
        next_obs[1]=int(y)
        next_obs[2]=int(m)
        next_obs[3] = random.choice(state_dict[random_key])
        random_hour = random.randint(0, 23)
        random_min = random.choice([0, 15, 30, 45])
        next_obs[4] = random_hour
        next_obs[5] = random_min
        next_obs[6] = self.var_obj.initialChargingLevel
        next_obs[7] = self.var_obj.shortest_paths_data[str(next_obs[0])]['dis_to_charging_station']
        return next_obs
    
    def reset(self):
 
        key = list(self.states_dict.keys())[0]
        [y,m] = key.split('_')
        d = (self.states_dict[key])[0]
        initial_state = [1 ,int(y), int(m), d, 0, 0,self.var_obj.initialChargingLevel,self.var_obj.shortest_paths_data['1']['dis_to_charging_station']]
        return initial_state
        
    def get_min_max_temp(self,state=None):        
        fin_name = 'data/temp_info' 
        with open(fin_name, 'rb') as fin:
            temp_info = pickle.load(fin)
        fin.close() 
        
        self.min_temp = temp_info['min_temp']
        self.max_temp = temp_info['max_temp']
        # self.mean_temp = temp_info['mean_temp']
        # self.std_temp = temp_info['std_temp']
        

    
               
    
    def compute_reach_time(self,current_sensor, next_sensor):
        reach_time_data = self.reach_time["sensor"+str(current_sensor)]
        reach_time = reach_time_data[str(current_sensor)+'_'+str(next_sensor)]
        return reach_time
        

    
    def step(self,state,action,his_trajectory=None,is_visited=0,time_offset=0):
        state = state.cpu().numpy().astype(int)[0]
        current_sensor = state[0]
        primary_action = action
        # print(action)
        # res = []
        # for act in range(1,8):
        #     if act != state[0]:
        #         action = act
        #         current_hour = state[4] 
        #         current_minute = state[5]
                
        #         Flag = False
        #         if current_sensor == action:
        #             new_state = state
        #         else:
        #             reach_to_next_time = self.var_obj.shortest_paths_data[str(current_sensor)][str(action)]['distance']+time_offset
        #             self.reach_time_minutes = reach_to_next_time-time_offset
        #             base_time = str(current_hour)+':'+str(current_minute)+':00'
        #             next_hour, next_min, Flag = self.func.add_minutes(base_time, reach_to_next_time+(is_visited*self.process_time))
        #             next_charge_level = state[6]-reach_to_next_time-self.process_time
        #             dis_to_charging_station = self.var_obj.shortest_paths_data[str(action)]['dis_to_charging_station']
        #             new_state = [action,state[1],state[2],state[3],next_hour,next_min,next_charge_level,dis_to_charging_station]
        #         reward,temperature_difference,reach_time_minutes,_ = self.calculate_reward(state, new_state,his_trajectory)  # Calculate the reward
        #         res.append([act,reward,temperature_difference,reach_time_minutes,next_charge_level,dis_to_charging_station])
            
        
        
        action = int(primary_action) 
        # print(max(res, key=lambda x: x[1]))
        # print(primary_action)

        current_hour = state[4] 
        current_minute = state[5]
        Flag = False
        if current_sensor == action:
           new_state = state
        else:
            # reach_to_next_time = reach_time[str(current_sensor)+'_'+str(action)]+time_offset
            reach_to_next_time = self.var_obj.shortest_paths_data[str(current_sensor)][str(action)]['distance']
            self.reach_time_minutes = reach_to_next_time
            base_time = str(current_hour)+':'+str(current_minute)+':00'
            next_hour, next_min, Flag = self.func.add_minutes(base_time, reach_to_next_time)
            next_charge_level = state[6]-reach_to_next_time-self.process_time
            dis_to_charging_station = self.var_obj.shortest_paths_data[str(action)]['dis_to_charging_station']
            new_state = [action,state[1],state[2],state[3],next_hour,next_min,next_charge_level,dis_to_charging_station]
        reward,temperature_difference,reach_time_minutes,his_trajectory = self.calculate_reward(state, new_state,his_trajectory)  # Calculate the reward
    
        his_trajectory,_ = self.update_his_trajectory(state, his_trajectory)
        
        time_offset = self.process_time
        if new_state[6] == self.var_obj.initialChargingLevel:
            time_offset *= 2
        new_state = self.add_process_time(new_state,time_offset)
        return new_state, Flag,reward,temperature_difference,reach_time_minutes,his_trajectory
    
    def add_process_time(self,state,added_min):
        current_hour = state[4] 
        current_minute = state[5]
        base_time = str(current_hour)+':'+str(current_minute)+':00'
        next_hour, next_min, Flag = self.func.add_minutes(base_time, added_min)  
        new_state = [state[0],state[1],state[2],state[3],next_hour,next_min,state[6],state[7]]
        return new_state
    
    def get_current_temp(self,state):
            current_sensor = int(state[0])
            temp_data =  self.sensorsdata["sensor"+str(current_sensor)]
            temp_data = [value for value in temp_data.values()][0]
            state_data = [value for value in temp_data.values() if value['year'] == int(state[1]) and value['month'] == int(state[2]) and value['day'] == int(state[3])] 
            if len(state_data)==0:
                return None
            temp_var = state_data[0]['sensorData']
            current_sate = temp_var[temp_var['hour'] == state[4]]
            
            current_temperature = 0
            # Check the value of A and select the corresponding row
            if state[5] == 0:
                current_temperature = current_sate.iloc[0].temp_to_estimate
            elif state[5] == 15:
                current_temperature = current_sate.iloc[1].temp_to_estimate
            elif state[5] == 30:
                current_temperature = current_sate.iloc[2].temp_to_estimate
            elif state[5] == 45:
                current_temperature = current_sate.iloc[3].temp_to_estimate
            else:
                current_temperature = None
                
            return current_temperature
    
    # def calculate_reward_for_first_state(self,state, temperature_change):
        
       
       
    #         temperature_change =abs((temperature_change - self.min_temp) / (self.max_temp - self.min_temp))
    #         # Combine the two factors with the defined weights
    #         reward = (self.temperature_weight *abs( temperature_change)) - self.process_penalty
            
            
    #         return reward
    
    def calculate_reward_for_first_state(self,state, temperature_change):
        
            
       
            # temperature_change =abs((temperature_change - self.min_temp) / (self.max_temp - self.min_temp))
            # Combine the two factors with the defined weights
            reward = (self.temperature_weight *abs( temperature_change)) - self.process_penalty
            
            
            return reward
    
    def update_his_trajectory(self,state,his_trajectory):
        desired_item = None

        for key, value in his_trajectory.items():
            if value[0][0] == state[0]:
                desired_item = value
                break

       
        next_temperature = desired_item[1]
        his_trajectory[key] = [state,next_temperature]
        return his_trajectory,next_temperature
    

    
    def normalize_temperature(self,temperature_data):
        temperature_anomalies = temperature_data - self.min_temp
        normalized_temperature = temperature_anomalies / (self.max_temp-self.min_temp)
        return normalized_temperature
    
    def normalize_time(self,time):
        time_anomalies = time - self.mean_time
        normalized_time = time_anomalies / self.std_time
        return abs(normalized_time)
    
    
    
    def get_best_charging_station(self,obs_input,his_trajectory):
        
        obs = obs_input.cpu().numpy()
        current_sensor = obs[0][0]
        battery_level = obs[0][6]
        dis_to_chargingStation = obs[0][7]
        reward_set = dict()
        for station in self.var_obj.charging_stations:
            # if battery_level > self.var_obj.shortest_paths_data[current_sensor][station]:
            new_state, Flag,reward,temperature_difference,reach_time_minutes,_ = self.step(obs_input, station,his_trajectory)
            reward_set[station] = reward  
        
        
        max_index = max(reward_set, key=reward_set.get)    
        return max_index
        


    def calculate_reward(self,state, next_state,his_trajectory):
        reward= 0
        battery_level = next_state[6]
        dis_to_chargingStation = next_state[7]
        
        if next_state[0]== state[0]:
            return -1000,0,90,his_trajectory
        else:
            
  
            
            reward = 0
            
            if (dis_to_chargingStation > battery_level or battery_level<=0) and dis_to_chargingStation!=0:
                return -20,None,None,his_trajectory
            
            elif battery_level<=60 and dis_to_chargingStation!=0:
                reward = -100
            
            elif (dis_to_chargingStation > battery_level or battery_level<=60) and dis_to_chargingStation==0:
                reward = 50 
                next_state[6] = self.var_obj.initialChargingLevel
        

    
            his_trajectory,temperature = self.update_his_trajectory(next_state,his_trajectory)        
            # Calculate the temperature change from current to next state
            next_temperature = self.get_current_temp(next_state)
           
            temperature_change1 = abs(next_temperature - temperature)
            
            time_factor = self.normalize_time(self.reach_time_minutes)
            self.process_penalty = 15
        
            # temperature_change1 = abs(self.normalize_temperature(next_temperature) - self.normalize_temperature(temperature))
            # time_factor = abs((self.reach_time_minutes - self.min_time) / (self.max_time - self.min_time))   
            
            # check wether charging level in new position is enough to reach to nearest charging station or not
            if reward not in[50,-100,-20]:
                reward = (self.temperature_weight * temperature_change1) - (self.time_weight * time_factor) - (self.charging_weight * time_factor) #- self.process_penalty
            else:
                reward += (self.temperature_weight * temperature_change1)
                
            return reward,temperature_change1,self.reach_time_minutes,his_trajectory

    def extract_history_traj(self,current_state):
        current_sensor = int(current_state[0])
        next_sensor = 0
        
        history_path = dict()
        initial_sensor = int(current_state[0])
        charging_level = current_state[6]
        dis_to_charging_station = current_state[7]
        
        while initial_sensor != next_sensor:
            current_sensor = int(current_state[0])
            temperature = self.get_current_temp(current_state)
            history_path[str(current_sensor)]=[current_state, temperature]
            base_time = str(int(current_state[4])) + ':' + str(int(current_state[5]))+ ':00'
            next_hour, next_min, Flag = self.func.add_minutes(base_time, 15)
            next_sensor = self.find_next_item(current_sensor)
            current_state = np.array([int(next_sensor), int(current_state[1]), int(current_state[2]), int(current_state[3]), next_hour, next_min, self.var_obj.initialChargingLevel,self.var_obj.shortest_paths_data[str(next_sensor)]['dis_to_charging_station']])
        
        # Custom sorting key based on day, hour, and minutes
        def sorting_key(item):
            return tuple(item[0][2:5])
        

        # Sort the dictionary based on the custom key
        his_trajectory = dict(sorted(history_path.items(), key=lambda x: sorting_key(x[1])))
        last_item = list(his_trajectory.values())[-1]
        current_state = last_item[0]
        base_time = str(current_state[4]) + ':' + str(current_state[5]) + ':00'
        next_hour, next_min, Flag = self.func.add_minutes(base_time, 15)
        next_sensor = self.find_next_item(current_state[0])
        current_state = np.array([int(next_sensor), int(current_state[1]), int(current_state[2]), int(current_state[3]), next_hour, next_min,charging_level,dis_to_charging_station])

        return history_path,np.reshape(current_state, [1, self.state_dim])
    
    def find_next_item(self,num):
        # Find the index of the given number in the array
        try:
            index = self.POIs.index(num)
        except ValueError:
            print(f"The number {num} is not in the array.")
            return None
    
        # Calculate the index of the next number in the circular array
        next_index = (index + 1) % len(self.POIs)
    
        # Return the next number
        return self.POIs[next_index]  
    
    
    
    def has_shape_zero(self,value):
        return value[5].shape == (0, 0)
    
    def refine_time(self,table):
        
        visited_rows = [0]

        for row in table:
            visited_rows.append(1 if row[1] == 'Visited' else 0)

        cumulative_sum = np.cumsum(visited_rows)  
        
        count = 0
        for row in table:
            if row[1] == 'Passed':
                state = row[6]
                base_time = str(state[0][4])+':'+str(state[0][5])+':00'
                extra_time = self.process_time 
    
                state[0][4], state[0][5], Flag = self.func.add_minutes(base_time, extra_time)
                row[6]=state
                
            count += 1

        initial_day = table[0][6][0][3]   
        for row in table:
            state = row[6]
            if state[0][4] == 0 and state[0][3]==initial_day:  # Check if hour is 0 and minutes is 0
               state[0][3] += 1  # Increment the day by 1
        return table
    
    
    def add_prev_temp(self,table,his_trajectory):
        

        for row in table:
            if row[1] == 'Passed' and row[-1]!=None:
                temperature_difference = abs(his_trajectory[row[0]][1] - row[-1])
                row[4] = temperature_difference
                
            
        return table
    
    def create_train_db(self,complete_paths):
        
        climatic_data = dict()
        for sensor in range(1,8):
            temp_data =  self.sensorsdata["sensor"+str(sensor)]
            temp_data = [value for value in temp_data.values()][0]
            first_element = next(iter(temp_data.items()))
            climatic_data[sensor] = first_element[1]['sensorData']
        
        
        df_train =  pd.DataFrame(columns=list(climatic_data[1].columns) + ['sensor'])
        for item in complete_paths.values():
            
            
            
            
            indx = localize_row(climatic_data[item[0]],state)
            # df.loc[indx,'status'] = item[0]
            # df.to_csv(csv_file, index=False, encoding='utf-8')
            # if item[0]=='Visited':
                
            # df_train.loc[len(df_train)] = list(df.iloc[indx]) + [str(sensor)]
            
    def localize_row(df,state):
     
     all_conditions = (df['hour'] == state[4]) & (df['month'] == state[2]) & (df['day_of_month'] == state[3])
     # Check the value of A and select the corresponding row
     if state[5] == 0:
         condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '0')
     elif state[5] == 15:
         condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '1')
     elif state[5] == 30:
         condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '2')
     elif state[5] == 45:
         condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '3')
     
     indx = (df[all_conditions & condition].index)
     if indx.size == 1:
         return indx[0]
     else:
         return -1