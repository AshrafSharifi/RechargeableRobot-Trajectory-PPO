import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
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
from datetime import datetime
from keras.layers import Conv1D, ReLU, MaxPooling1D, Flatten
from ModelClass import *
import operator

def compute_metrics(predicted_values,actual_values):
    

    
    # Calculate loss, MSE, MAE, and MAPE
    loss = np.sum((predicted_values - actual_values) ** 2)
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Print the results
    print(f"Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    
    
def analyze_result_per_POIs(resul,sensor):   
    predicted_values = resul['Predicted']
    actual_values = resul['Actual']
    print('---Results For All ')
    compute_metrics(np.array(predicted_values), np.array(actual_values))
    sensor = pd.Series(sensor)
    for POI in range(1,8):
        indices = sensor[sensor == POI].index
        pred = predicted_values[indices]
        actual = actual_values[indices]
        print('---Results For POI ',str(POI))
        compute_metrics(np.array(pred), np.array(actual))
        # plot_result_for_each_POI(actual_values,predicted_values,POI,result_items)
        
        
# Load your dataset (assuming it's in a DataFrame format)
# Replace 'data' with your actual DataFrame variable
dropout= 0.2
patience= 100
# Select relevant features (excluding 'temp_to_estimate' from the input features)
features = [
    'sensor', 'year', 'month', 'week', 'day_of_year', 
    'day_of_month', 'day_of_week', 'hour', 'dist_to_central_station', 
    'temp_centr','temp_to_estimate'
]
target = 'temp_to_estimate'
train = 0
if train==1:
    DB_file = "data/DB/train_DB.csv"
    all_data = pd.read_csv(DB_file)[features]
    for i in range(1,8):
        print(len(all_data[all_data['sensor']==i]))
    data =all_data[features[:-1]]
    
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(data)

    # Normalize the target variable
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(all_data[[target]])
    
    
    
    # Define the number of timesteps and reshape data
    timesteps = 3  # For example, 3 time steps
    X = []
    y = []
    
    for i in range(timesteps, len(scaled_target)):
        X.append(scaled_features[i-timesteps:i])
        y.append(scaled_target[i])
    
    X, y = np.array(X), np.array(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # # Build the LSTM model
    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1))  # Output layer to predict 'temp_to_estimate'
    
    
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    # custom_optimizer = Adam(learning_rate=learning_rate)
    # model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])

    model.compile(optimizer='adam', loss='mae', metrics=['mape', 'mae', 'mse'])
    
    
    
    # # Compile the model
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mape', 'mae', 'mse'])
    
    # # Train the model
    # history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    checkpointer = ModelCheckpoint(filepath = 'data/Simple_LStm/lstm_model.h5', verbose = 2, save_best_only = True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    # Train the model       
    # history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,verbose=1, validation_split=validation_split,callbacks = [checkpointer,early_stopping])
    
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=.2, callbacks=[checkpointer, early_stopping])
    
    
    # Evaluate the model on test data
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    # Save the trained model and scalers
    # model.save('data/Simple_LStm/lstm_model.h5')  # Save the model
    joblib.dump(feature_scaler, 'data/Simple_LStm/feature_scaler.pkl')  # Save feature scaler
    joblib.dump(target_scaler, 'data/Simple_LStm/target_scaler.pkl')    # Save target scaler
    # # Predict
    # y_pred = model.predict(X_test)
    
    # # Inverse scaling if necessary (to return to the original scale)
    # y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    # y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    
    # # Display actual and predicted values side by side in a DataFrame
    # results_df = pd.DataFrame({
    #     'Actual': y_test_rescaled,
    #     'Predicted': y_pred_rescaled
    # })
    # compute_metrics(y_pred_rescaled, y_test_rescaled)

else:
    # Load the saved model and scalers
    model = load_model('data/Simple_LStm/lstm_model.h5')  # Load the model
    feature_scaler = joblib.load('data/Simple_LStm/feature_scaler.pkl')  # Load feature scaler
    target_scaler = joblib.load('data/Simple_LStm/target_scaler.pkl')    # Load target scaler
    
    
    # Load your test data (assuming you have a test CSV file)
    test_data = pd.read_csv('data/DB/test_DB.csv')  # Replace with your test dataset path
    test_data = test_data.dropna()
    all_data = test_data[features]
    data =all_data[features[:-1]]
    
    
    
    # Select features and preprocess the test data (same process as training)
    scaled_test_features = feature_scaler.transform(data)  # Exclude the target column
    
    # Reshape test data for LSTM (using the same timestep as used in training)
    X_test = []
    timesteps = 3  # Make sure this matches the timestep used during training
    
    for i in range(timesteps, len(scaled_test_features)):
        X_test.append(scaled_test_features[i-timesteps:i])
    
    X_test = np.array(X_test)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Inverse transform the predicted values back to the original scale
    y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # (Optional) If you have actual values for the test set, inverse transform them as well
    # Assuming you have the actual target in the test data
    actual_test_values = test_data[target][timesteps:].values  # Exclude the initial timesteps
    sensors = test_data['sensor'][timesteps:].values  # Exclude the initial timesteps
    
    
    # Display the actual vs predicted results in a DataFrame
    results_df = pd.DataFrame({
        'Actual': actual_test_values,
        'Predicted': y_pred_rescaled
    })
    print(results_df)
    
    # Compute and display metrics (MSE, MAE, MAPE)
    compute_metrics(y_pred_rescaled, actual_test_values)
    analyze_result_per_POIs(results_df,sensors)
   
