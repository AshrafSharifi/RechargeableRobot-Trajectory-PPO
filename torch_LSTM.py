import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
# Step 1: Data Preparation

# Sample Data (Replace this with your actual data loading method)
data_header = [
    'sensor', 
    'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
    'hour', 'complete_timestamp(YYYY_M_DD_HH_M)', 'temp_centr', 'temp_to_estimate'
]
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h_0, c_0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
train = 0
if train==1:
    DB_file = "data/DB/train_DB.csv"
    df = pd.read_csv(DB_file)
    df = pd.DataFrame(df)
    
    # Select relevant features and target
    features = df[data_header]
    target = df['temp_to_estimate']
    
    # Normalize the data
    scaler = MinMaxScaler()
    feature_scaler = scaler.fit_transform(features)
    target_scaler = scaler.fit_transform(target.values.reshape(-1, 1))
    


    # Save the scalers
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    
    # Create sequences of data for LSTM
    def create_sequences(data, target, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = target[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    SEQ_LENGTH = 10
    X, y = create_sequences(feature_scaler, target_scaler, SEQ_LENGTH)
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Step 2: LSTM Model Definition
    

    
    # Hyperparameters
    input_size = X.shape[2]  # Number of features
    hidden_size = 50
    num_layers = 2
    output_size = 1  # Predicting single value, temp_to_estimate
    
    # Instantiate the model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Step 3: Training Loop
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'lstm_temp_model.pth')
    
    # Step 4: Evaluation
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(batch_y.cpu().numpy())
    
    # Inverse transform to get actual values
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_values = scaler.inverse_transform(np.array(true_values).reshape(-1, 1))
    
    # Calculate MAPE, MAE, and MSE
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    
    print(f'MAPE: {mape:.2f}%')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    
  
    

else:   
    DB_file = "data/DB/test_DB.csv"
    df = pd.read_csv(DB_file)
    df = pd.DataFrame(df)
    
    # Select relevant features and target
    features = df[data_header]
    target = df['temp_to_estimate']
    
    # Normalize the data
    scaler = MinMaxScaler()
    feature_scaler = scaler.fit_transform(features)
    target_scaler = scaler.fit_transform(target.values.reshape(-1, 1))
    



    
    # Create sequences of data for LSTM
    def create_sequences(data, target, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = target[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    SEQ_LENGTH = 10
    X, y = create_sequences(feature_scaler, target_scaler, SEQ_LENGTH)
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Step 2: LSTM Model Definition
    
    
    
    
    
    # Define the model structure as used in training
    input_size = X.shape[2]  # Number of features
    hidden_size = 50
    num_layers = 2
    output_size = 1
    
    # Initialize and load the model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load('lstm_temp_model.pth'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(batch_y.cpu().numpy())
    
    # Inverse transform to get actual values
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_values = scaler.inverse_transform(np.array(true_values).reshape(-1, 1))
    
    
    
     # Calculate MAPE, MAE, and MSE
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    
    print(f'MAPE: {mape:.2f}%')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
 