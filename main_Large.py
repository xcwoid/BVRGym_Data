import sys, os, argparse
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
print(sys.executable)
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


col_names =  ['Dist', 'Time', 'HeadTarRel', 'Theta', 'Vel', 'Alt', 'Psi', 'AimVel0', 'AimAlt0', 'MD' ]

def load_data(pth):
    N = np.load(pth+'SSA_n10_n10.npy')
    NE = np.load(pth+'SSA_n075_n10.npy')
    E = np.load(pth+'SSA_n05_n10.npy')
    SE = np.load(pth+'SSA_n025_n10.npy')
    S = np.load(pth+'SSA_p00_n10.npy')
    SW = np.load(pth+'SSA_p025_n10.npy')
    W = np.load(pth+'SSA_p05_n10.npy')
    NW = np.load(pth+'SSA_p075_n10.npy')
    np_dict = {'N':N,
               'NE': NE,
               'E': E,
               'SE': SE ,
               'S':S,
               'SW': SW,
               'W': W,
               'NW': NW}
    return np_dict

def to_df(np_dict, overwrite = False):
    df0 = pd.DataFrame()
    for key in np_dict:
        df = pd.DataFrame(np_dict[key], columns=col_names)
        # forgot to record it, but it was constant in this case 
        if overwrite:
            df['AimVel0'] = 340.0
            df['AimAlt0'] = 10e3
        df['N'] = 0.0
        df['NE'] = 0.0
        df['E'] = 0.0
        df['SE'] = 0.0
        df['S'] = 0.0
        df['SW'] = 0.0
        df['W'] = 0.0
        df['NW'] = 0.0
        df[key] = 1.0
        df0 = pd.concat([df0, df], ignore_index=True)
    return df0 

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, hidden_size=64, num_heads=8):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  
        x = self.decoder(x[:, -1, :])  
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.decoder(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)        
        out, _ = self.rnn(x, h0)
        out = self.decoder(out[:, -1, :])
        return out
    

def train(X, y, model, save_to, send_to = 'cuda', batch_size = 32):

    writer = SummaryWriter('runs/'+save_to)
    # Split your data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert your data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(send_to)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(send_to)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float).to(send_to)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float).to(send_to)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Move model to GPU
    model = model.to(send_to)
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        writer.add_scalar("loss", running_loss / len(train_loader), epoch+1)
        #loss_train.append(running_loss / len(train_loader))
        #print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")

        torch.cuda.empty_cache()
        
        # Validation
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
            writer.add_scalar("test_loss", val_loss.item(), epoch+1)
            #loss_val.append(val_loss.item())
            #print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss.item()}")
        
        torch.cuda.empty_cache()
    
    # Saving losses
    #np.save(save_to + str(nr) + '_train.npy', np.array(loss_train))
    #np.save(save_to + str(nr) + '_val.npy', np.array(loss_val))


def get_data_list(pth, overwrite, pre_pad = True):

    np_dict = load_data(pth)
    df0 = to_df(np_dict, overwrite)

    #df0.drop(df0[df0['Time'] < 10].index, inplace=True)
    df0.drop(df0[df0['Time'] > 45].index, inplace=True)
    df0.drop(df0.index[df0.index % 2 == 0], inplace=True)
    num_rows = len(df0)
    rows_to_keep = num_rows // 10
    print('Data size:', rows_to_keep, 'out of ', num_rows)
    df0 = df0.head(rows_to_keep)

    for i in col_names:
        column_data = df0[[i]]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_column_data = scaler.fit_transform(column_data)
        df0[i] = scaled_column_data


    # Convert DataFrame to NumPy arrays
    features = df0.drop(columns=['MD']).values  
    targets = df0['MD'].values
    
    del df0
    del np_dict

    features_series = []
    targets_series = []
    tmp = []
    for idx, i in enumerate(features):
        if not pre_pad:
            tmp.append(i)
            features_series.append(np.array(tmp))
            targets_series.append(np.array(targets[idx]))
            tmp = []
        else:
            if tmp == []:
                tmp.append(i)
                features_series.append(np.array(tmp))
                targets_series.append(np.array(targets[idx]))
            else:
                if tmp[-1][1] < i[1]:
                    # add
                    tmp.append(i)
                    features_series.append(np.array(tmp))
                    targets_series.append(np.array(targets[idx]))
                else:
                    tmp = []


    # Assuming you have data_list with arrays of different shapes
    data_list = [torch.tensor(seq) for seq in features_series]  
    del features_series
    targets_data = torch.tensor(np.array(targets_series), dtype=torch.float32)
    del targets_series

    return data_list, targets_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type = str, help="tf, lstm, rnn", default=' ')
    parser.add_argument("-data", "--data", type = str, help="alt10, alt10c", default=' ')
    parser.add_argument("-to", "--to", type = str, help="cuda, cpu", default='cuda')
    parser.add_argument("-cluster", "--cluster", action='store_true', help="cluster path")
    parser.add_argument("-eps", "--eps", type = int, help="eps", default=1)
    parser.add_argument("-ta", "--trainAll", action='store_true', help="train all")
    args = vars(parser.parse_args())

    if args['trainAll']:
        for m in ['tf', 'lstm', 'rnn']:
            for d in ['alt10', 'alt10c']:
                print(m,d)
                for e in range(args['eps']):
                    if d == 'alt10':
                        pth = 'SSA_M_alt_10/'
                        overwrite = True
                    elif d == 'alt10c':
                        pth = 'SSA_M_alt_10_crank/'
                        overwrite = False
                    else:
                        print('Wrong path')
                        exit()

                    data_list, targets_data = get_data_list(pth, overwrite, pre_pad=False)
                    padded_data = pad_sequence(data_list, batch_first=True, padding_value=0)
                    X = padded_data
                    y = targets_data
                    
                    save_to = m + '_' + d + '_noPad'
                    
                    if m == 'tf':
                        model = TransformerModel(input_size=17, output_size=1)
                        train(X, y, model, save_to = save_to, batch_size = 32, send_to= args['to'])

                    elif m == 'lstm':
                        model = LSTMModel(input_size=17, output_size=1)
                        train(X, y, model, save_to = save_to, batch_size = 32, send_to= args['to'])
                    
                    elif m == 'rnn':
                        model = RNNModel(input_size=17, output_size=1)
                        train(X, y, model, save_to = save_to, batch_size = 32, send_to= args['to'])

    else:
        for i in range(args['eps']):
            if args['data'] == 'alt10':
                if args['cluster']:
                    pth = '/local_storage/users/scukins/SSA_M_alt_10/'
                else:
                    pth = 'SSA_M_alt_10/'
                    #pth = '../logs/SSA_M_alt_10/'
                overwrite = True
            elif args['data'] == 'alt10c':
                if args['cluster']:
                    pth = '/local_storage/users/scukins/SSA_M_alt_10_crank/'
                else:
                    #pth = '../logs/SSA_M_alt_10_crank/'
                    pth = 'SSA_M_alt_10_crank/'
                overwrite = False
            else:
                print('Wrong path')
                exit()

            data_list, targets_data = get_data_list(pth, overwrite)
            padded_data = pad_sequence(data_list, batch_first=True, padding_value=0)
            X = padded_data
            y = targets_data
            
            save_to = args['model'] + '_' + args['data']

            if args['model'] == 'tf':
                model = TransformerModel(input_size=17, output_size=1)
                train(X, y, model, save_to = save_to, batch_size = 32, send_to= args['to'])

            elif args['model'] == 'lstm':
                model = LSTMModel(input_size=17, output_size=1)
                train(X, y, model, save_to = save_to, batch_size = 32, send_to= args['to'])
            
            elif args['model'] == 'rnn':
                model = RNNModel(input_size=17, output_size=1)
                train(X, y, model, save_to = save_to, batch_size = 32, send_to= args['to'])



# python main_Large.py -model rnn -data alt10c -eps 30
# python main_Large.py -ta -eps 30














