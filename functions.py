import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import pickle
from datetime import datetime, timedelta
from tabulate import tabulate

class functions(object):

  def __init__(self, sensor_data = dict(), threshold=0,sensor=0, displayResult=False, hasOutput=False):
        self.df = sensor_data;
        self.threshold = threshold;
        if len(sensor_data)!=0:
            self.date = str(sensor_data['year'].iloc[0]) + '-' + str(sensor_data['month'].iloc[0]) + '-' + str(sensor_data['day_of_month'].iloc[0]);
        else:
            self.date = ""
        self.displayResult = displayResult
        self.hasOutput = hasOutput
        self.sensor = sensor
  def data_preperation(self):

    # Calculate temperature change as the difference between consecutive temperature values
    self.df['temperatureChange'] = self.df['temp_to_estimate'].diff()

    # The first row will have NaN in the 'TemperatureChange' column; you can replace it with 0 if needed
    self.df['temperatureChange'].fillna(0, inplace=True)
    
    self.df = self.df.sort_values(by='hour')
    self.df = self.df.to_numpy()
    return self.df;
  

  def first_derivative(self):
    
    data=self.df;
    temperature_data = data[:, 1]
    time_vector = data[:, 9]
    
    # Calculate the first derivative (rate of change)
    derivative = np.diff(temperature_data)
    
    # Find indices where the derivative exceeds the threshold
    sharp_change_indices = np.where(np.abs(derivative) > self.threshold)[0]
    # average_temperature = temperature_data.mean()

    # # Find indices of values exceeding the threshold in both directions
    # above_threshold_indices = np.where(temperature_data > average_temperature + self.threshold)[0]
    # below_threshold_indices = np.where(temperature_data < average_temperature - self.threshold)[0]
    
    # # Concatenate the indices into a single array
    # sharp_change_indices = np.concatenate((above_threshold_indices, below_threshold_indices))

    
    if self.displayResult:
        # Display the sharp change indices and values
        print('Sharp Change Indices:')
        print(time_vector[sharp_change_indices])
        print('Sharp Change Values:')
        print(derivative[sharp_change_indices])
        plt.close('all')
        # Plot the temperature data with detected sharp changes
        plt.figure()
        plt.plot(time_vector, temperature_data, 'b', linewidth=2)
        plt.scatter(time_vector[sharp_change_indices], temperature_data[sharp_change_indices], c='r', marker='o')
        plt.xlabel('Time Intervals')
        plt.ylabel('Temperature')
        # plt.title("Temperature Data with Sharp Change Detection (" + self.date + "_Sensor: "+ str(self.sensor) +" _Threshold: "+ str(round(self.threshold, 3) )+ " )")
        plt.title("Temperature Data with Sharp Change Detection (" + self.date + "_Sensor: "+ str(self.sensor) + " )")
        plt.legend(['Temperature Data', 'Sharp Changes'])
        plt.grid(True)
        plt.ylim(0, 35)
        plt.show()
    if self.hasOutput:
        out= {}
        out["SharpChangeValues"] = derivative[sharp_change_indices]
        out["SharpChangeIndices"] = sharp_change_indices
        out["SharpChangeTimes"] = time_vector[sharp_change_indices]
        return out

  def find_sharp_changes(self):
    
    data=self.df;
    temperature_data = data[:, 1]
    time_vector = data[:, 9]
    
    # Calculate the whole average temperature
    average_temperature = self.df['temp_to_estimate'].mean()
    
    # Calculate the standard deviation of the temperature
    temperature_standard_deviation = self.df['temp_to_estimate'].std()
    
    # Set a threshold for detecting sharp changes in temperature
    threshold = 2 * temperature_standard_deviation
    
    # Identify any temperature changes that fall outside of the typical range
    df_outliers = self.df[self.df['temp_to_estimate'] > average_temperature + threshold]
    df_outliers = df_outliers.append(self.df[self.df['temp_to_estimate'] < average_temperature - threshold])
    
    
    # if self.displayResult:
    #     # Display the sharp change indices and values
    #     print('Sharp Change Indices:')
    #     print(time_vector[sharp_change_indices])
    #     print('Sharp Change Values:')
    #     print(derivative[sharp_change_indices])
        
    #     # Plot the temperature data with detected sharp changes
    #     plt.figure()
    #     plt.plot(time_vector, temperature_data, 'b', linewidth=2)
    #     plt.scatter(time_vector[sharp_change_indices], temperature_data[sharp_change_indices], c='r', marker='o')
    #     plt.xlabel('Time Intervals')
    #     plt.ylabel('Temperature')
    #     plt.title("Temperature Data with Sharp Change Detection (" + self.date + ")")
    #     plt.legend(['Temperature Data', 'Sharp Changes'])
    #     plt.grid(True)
    #     plt.show()
    # if self.hasOutput:
    #     out= {}
    #     out["SharpChangeValues"] = derivative[sharp_change_indices]
    #     out["SharpChangeIndices"] = sharp_change_indices
    #     out["SharpChangeTimes"] = time_vector[sharp_change_indices]
    #     return out

  def corr_matrix(self, columns):
    columns =['temperatureChange','temp_to_estimate','temp_centr','hum','dewpoint__c','wetbulb_c','windspeed_km_h','thswindex_c','solar_rad_w_m_2']
    # df = self.df[['temperatureChange','temp_to_estimate','temp_centr','hum','dewpoint__c','wetbulb_c','windspeed_km_h','thswindex_c','rain_mm','solar_rad_w_m_2']]
    df = self.df[columns]
   
    # Drop non-numerical variables
    # _, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    corr_matrix = df.corr()
    return(corr_matrix)
    # display(corr_matrix)
    # sns.heatmap(corr_matrix,ax=axes);
    
  def find_threshold_based_on_variation(self):

    # Calculate the temperature variations (differences between consecutive values)
    temperature_variations = self.df['temp_to_estimate'].diff().fillna(0)
    
    # Calculate the mean and standard deviation of temperature variations
    mean_variations = temperature_variations.mean()
    std_variations = temperature_variations.std()
    
    # Define a threshold as a multiple of the standard deviation (e.g., 2 times the std)
    threshold = 2 * std_variations
    
    return(threshold)
    

    
  def find_threshold(self):

     # Calculate the temperature variations (differences between consecutive values)
    temperature_variations = self.df['temp_to_estimate']
    
    # Calculate the mean and standard deviation of temperature variations
    mean_variations = temperature_variations.mean()
    std_variations = temperature_variations.std()
    
    # Define a threshold as a multiple of the standard deviation (e.g., 2 times the std)
    threshold = 2 * std_variations
    return (threshold)
  
  def add_minute(self,base_time, minutes_to_add):
     # Parse the base time as a datetime object
     base_time = datetime.strptime(base_time, '%H:%M:%S')

     # Create a timedelta to represent the minutes to add
     time_delta = timedelta(minutes=minutes_to_add)

     # Add the timedelta to the base time
     result_time = base_time + time_delta

     # Format the result as a string in HH:MM:SS format
     result_time_str = result_time.strftime('%H:%M:%S')
     hours = result_time.hour
     minutes = result_time.minute

     return hours, minutes
  
  def add_minutes(self, base_time, minutes_to_add):
      
      flag= False
      # Parse the base time as a datetime object
      base_time = datetime.strptime(base_time, '%H:%M:%S')

      # Create a timedelta to represent the minutes to add
      time_delta = timedelta(minutes=int(minutes_to_add) )

      # Add the timedelta to the base time
      result_time = base_time + time_delta

      # Check if the resulting time is greater than or equal to 24 hours
      day_difference = (result_time.date() - base_time.date()).days

     # Check if dt2 is the next day of dt1
      if day_difference == 1:
          flag = True

      # Format the result as a string in HH:MM:SS format
      result_time_str = result_time.strftime('%H:%M:%S')
      hours = result_time.hour
      minutes = result_time.minute

      return hours, minutes, flag
    
    
  def time_to_reach_POI(self):
      
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
  
  def get_all_data(self):
        
        sensorList = list(range(1,9))
        output = dict()
        for sensor in sensorList:
            fin_name = 'data/aggregated_data' + str(sensor)
            with open(fin_name, 'rb') as fin:
                aggregated_data = pickle.load(fin)
                output["sensor" + str(sensor)] = aggregated_data
            fin.close() 
            
        return output
  # def prepare_data_for_train(self):
      
  #     pd.options.mode.chained_assignment = None 
  #     sensorList = list(range(1,9))
     
  #     for sensor in sensorList: 
  #         fin_name = 'data/aggregated_data' + str(sensor)
  #         with open(fin_name, 'rb') as fin:
  #             aggregated_data = pickle.load(fin)
  #         fin.close()
   
  def change_outlier(self,df):
      
        
        # Calculate the IQR (Interquartile Range) to identify outliers
        Q1 = df['temp_to_estimate'].quantile(0.25)
        Q3 = df['temp_to_estimate'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define a threshold to identify outliers (you can adjust this as needed)
        outlier_threshold = 1.5
        
        # Identify outliers
        outliers = df[(df['temp_to_estimate'] < Q1 - outlier_threshold * IQR) | (df['temp_to_estimate'] > Q3 + outlier_threshold * IQR)]
        
        # Replace outliers with their previous values
        for idx in outliers.index:
            if idx > 0:
                df.at[idx, 'temp_to_estimate'] = df.at[idx - 1, 'temp_to_estimate']
        
        return df       
      


     

    
 
