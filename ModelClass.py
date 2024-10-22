import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Concatenate , Bidirectional, Conv1D,Flatten , ReLU, Input, Reshape, BatchNormalization, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow.keras.utils as utils


class ModelClass:
    def __init__(self, timesteps, feature_size,dropout,learning_rate):
        self.timesteps = timesteps
        self.feature_size = feature_size
        self.dropout = dropout
        self.learning_rate = learning_rate
      
    def create_LSTM(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(self.timesteps, self.feature_size))))
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=1))
        custom_optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
        model.build((None, self.timesteps, self.feature_size))  # Here, you specify the input shape
        utils.plot_model(model, to_file="data/models/lstm.pdf", show_shapes=True)
        return model
    
    def create_lstm_cnn_model(self,dropout_rate=0.2, l2_rate=0.00001):
        # Input layer
        input_layer = Input(shape=(self.timesteps, self.feature_size))
        # LSTM layers
        lstm1 = Bidirectional(LSTM(units=256, return_sequences=True))(input_layer)
        dropout1 = Dropout(dropout_rate)(lstm1)
        lstm2 = Bidirectional(LSTM(units=128, return_sequences=True))(dropout1)
        dropout2 = Dropout(dropout_rate)(lstm2)
        lstm3 = Bidirectional(LSTM(units=64, return_sequences=True))(dropout2)
        dropout3 = Dropout(dropout_rate)(lstm3)
        # Reshape the LSTM output to match the input requirements of Conv1D
        reshape = Reshape((self.timesteps, 128))(dropout3)
        # First Conv1D + ReLU
        conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l2_rate))(reshape)
        relu1 = ReLU()(conv1)
        conv2 = Conv1D(filters=10, kernel_size=3, strides=1, padding='same')(relu1)
        relu2 = ReLU()(conv2)
        # Flatten the output
        flatten = Reshape((self.timesteps * 10,))(relu2)
        # Output layer
        output = Dense(units=1, kernel_regularizer=l2(l2_rate))(flatten)
        # Create the model
        model = Model(inputs=input_layer, outputs=output)
        # Compile the model
        custom_optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
        # Print the model summary to check the shapes
        # model.summary()
        model.build((None, self.timesteps, self.feature_size))  # Here, you specify the input shape
        utils.plot_model(model, to_file="data/models/lstm_cnn.pdf", show_shapes=True)
        return model
    
    def create_cnn_lstm_model(self,dropout_rate=0.2, l2_rate=0.00001):
        # Input layer
        input_layer = Input(shape=(self.timesteps, self.feature_size))
        # First Conv1D + ReLU
        conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l2_rate))(input_layer)
        relu1 = ReLU()(conv1)
        # Third Conv1D + ReLU (output channels match input channels)
        conv2 = Conv1D(filters=self.feature_size, kernel_size=3, strides=1, padding='same')(relu1)
        relu2 = ReLU()(conv2)
        reshape = Reshape((self.feature_size,self.timesteps))(relu2)
        # Define the LSTM layers using the functional API
        lstm1 = Bidirectional(LSTM(units=256, return_sequences=True))(reshape)
        dropout1 = Dropout(self.dropout)(lstm1)
        lstm2 = Bidirectional(LSTM(units=128, return_sequences=True))(dropout1)
        dropout2 = Dropout(self.dropout)(lstm2)
        lstm3 = Bidirectional(LSTM(units=64, return_sequences=False))(dropout2)
        dropout3 = Dropout(self.dropout)(lstm3)
        # Output layer
        output = Dense(units=1, kernel_regularizer=l2(l2_rate))(dropout3)
        # Create the model
        model = Model(inputs=input_layer, outputs=output)
        custom_optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
        # Print the model summary to check the shapes
        # model.summary()
        model.build((None, self.timesteps, self.feature_size))  # Here, you specify the input shape
        utils.plot_model(model, to_file="data/models/cnn_lstm.pdf", show_shapes=True)
        return model
    
    def create_parallel_cnn_lstm_model(self,l2_rate=0.00001):
        # Input layer
        input_layer = Input(shape=(self.timesteps, self.feature_size))
        # LSTM branch
        lstm_branch = Bidirectional(LSTM(units=256, return_sequences=True))(input_layer)
        lstm_branch = Dropout(self.dropout)(lstm_branch)
        lstm_branch = Bidirectional(LSTM(units=128, return_sequences=True))(lstm_branch)
        lstm_branch = Dropout(self.dropout)(lstm_branch)
        lstm_branch = Bidirectional(LSTM(units=64, return_sequences=False))(lstm_branch)
        lstm_branch = Dropout(self.dropout)(lstm_branch)
        lstm_branch = Dense(64, activation='relu')(lstm_branch)
        # CNN branch
        cnn_branch = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l2_rate))(input_layer)
        cnn_branch = ReLU()(cnn_branch)
        cnn_branch = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l2_rate))(cnn_branch)
        cnn_branch = ReLU()(cnn_branch)
        cnn_branch = Flatten()(cnn_branch)
        # Concatenate CNN and LSTM branches
        concatenated = Concatenate()([cnn_branch, lstm_branch])        
        # Final Dense layer
        output = Dense(units=1, kernel_regularizer=l2(l2_rate))(concatenated)        
        # Create the model
        model = Model(inputs=input_layer, outputs=output)        
        # Compile the model
        custom_optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
        # Print the model summary to check the shapes
        # model.summary()
        model.build((None, self.timesteps, self.feature_size))  # Here, you specify the input shape
        utils.plot_model(model, to_file="data/models/parallel_cnn_lstm.pdf", show_shapes=True)
        return model
        
       
            
        
    

    