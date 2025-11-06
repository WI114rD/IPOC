import torch
import torch.nn as nn
from torch.autograd import Variable
from pmdarima.arima import auto_arima
import statsmodels as sm
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF, DotProduct, Matern, RationalQuadratic
import sklearn
import numpy as np
import random
from tqdm import trange
import operator
from CP import ConformalPrediction, CPmodule
import time
import pandas as pd
from prophet import Prophet
import os
import logging
from data_preprocess import timeseries
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,ExtraTreesRegressor, AdaBoostRegressor,GradientBoostingRegressor
<<<<<<< HEAD
import lightgbm as lgb
=======

>>>>>>> d71b8ae5951d40bd6752d4bc0475319152b70f14
import warnings

logging.getLogger('prophet').setLevel(logging.WARNING) 





class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num = 1, dropout = 0.0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.cell = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim,
                            num_layers = self.layer_num, dropout = self.dropout,
                            batch_first = True) 
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 1, batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(self.layer_num * 1, batch_size, self.hidden_dim))
        rnn_output, hn = self.cell(x, (h0, c0))
        hn = hn[0].view(batch_size, self.hidden_dim)
        fc_output = self.fc(hn)
        return fc_output
    
class LSTM_test:
    def __init__(self, init_data=None, input_dim=1, hidden_dim=20, output_dim=1, input_length=12, lr=1e-3, train_epoch=40):
        self.model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.input_length = input_length
        self.lr = lr
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.train_epoch = train_epoch

        trainX, trainY = [], []
        train_size = len(init_data)
        for t in range(train_size - self.input_length):
            sampleX = init_data[t : self.input_length + t, :]
            sampleY = init_data[t + self.input_length, :]
            trainX.append(sampleX)
            trainY.append(sampleY)
        dataset = timeseries(trainX, trainY)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                                 num_workers=1)
        for x in range(self.train_epoch):
            # loss_epoch = 0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = Variable(x)
                self.last_pred = self.model.forward(x)
                self.BPprocess(y)

    def fit(self, sample):
        sample_np = np.array(sample[-1 * self.input_length:]).reshape(-1, self.input_length, 1)
        self.sample_t = torch.from_numpy(sample_np)
        self.sample_t = Variable(self.sample_t)

    def predict(self, horizon=1):
        if horizon == 1:
            self.last_pred = self.model.forward(self.sample_t)
            pred = np.squeeze(self.last_pred)
            # print(pred)
            return [pred.detach().numpy()]

    def BPprocess(self, y):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        y_tensor = Variable(y)
        self.optimizer.zero_grad()
        loss = self.criterion(self.last_pred, y_tensor)
        # loss_sum += loss.item()
        loss.backward()
        self.optimizer.step()
        return loss
    
class LSTM_test_CP:
    def __init__(self, init_data=None, input_dim=1, hidden_dim=20, output_dim=1, input_length=12, lr=1e-3, train_epoch=40,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        self.model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.input_length = input_length
        self.lr = lr
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.train_epoch = train_epoch
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        trainX, trainY = [], []
        train_size = len(init_data)
        for t in range(train_size - self.input_length):
            sampleX = init_data[t : self.input_length + t, :]
            sampleY = init_data[t + self.input_length, :]
            trainX.append(sampleX)
            trainY.append(sampleY)
        dataset = timeseries(trainX, trainY)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                                 num_workers=1)
        for x in range(self.train_epoch):
            # loss_epoch = 0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = Variable(x)
                self.last_pred = self.model.forward(x)
                self.BPprocess(y)
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)

    def fit(self, sample):
        sample_np = np.array(sample[-1 * self.input_length:]).reshape(-1, self.input_length, 1)
        self.sample_t = torch.from_numpy(sample_np)
        self.sample_t = Variable(self.sample_t)

    def predict(self, horizon=1):
        if horizon == 1:
            self.last_pred = self.model.forward(self.sample_t)
            pred = [np.squeeze(self.last_pred).detach().numpy()]   
            pred = [min(max(i, self.lower_threshold), self.upper_threshold) for i in pred]
            self.predictions = np.append(self.predictions, pred[0])
            if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
                y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
                return y_pred, y_upper_i, y_lower_i 
            else:
                return pred, None, None

    def BPprocess(self, y):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        y_tensor = Variable(y)
        self.optimizer.zero_grad()
        loss = self.criterion(self.last_pred, y_tensor)
        # loss_sum += loss.item()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.BPprocess(np.array([y_truth]))
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)



class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num = 1, dropout = 0.0):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.cell = nn.RNN(input_size = self.input_dim, hidden_size = self.hidden_dim,
                            num_layers = self.layer_num, dropout = self.dropout,nonlinearity="tanh",
                            batch_first = True) 
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    def forward(self, x):
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 1, batchSize , self.hidden_dim))
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hidden_dim)
        fcOutput = self.fc(hn)
        return fcOutput
    
class RNN_test_CP:
    def __init__(self, init_data=None, input_dim=1, hidden_dim=20, output_dim=1, input_length=12, lr=1e-3, train_epoch=40,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        self.model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.input_length = input_length
        self.lr = lr
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.train_epoch = train_epoch
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        trainX, trainY = [], []
        train_size = len(init_data)
        for t in range(train_size - self.input_length):
            sampleX = init_data[t : self.input_length + t, :]
            sampleY = init_data[t + self.input_length, :]
            trainX.append(sampleX)
            trainY.append(sampleY)
        dataset = timeseries(trainX, trainY)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                                 num_workers=1)
        for x in range(self.train_epoch):
            # loss_epoch = 0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = Variable(x)
                self.last_pred = self.model.forward(x)
                self.BPprocess(y)
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)

    def fit(self, sample):
        sample_np = np.array(sample[-1 * self.input_length:]).reshape(-1, self.input_length, 1)
        self.sample_t = torch.from_numpy(sample_np)
        self.sample_t = Variable(self.sample_t)

    def predict(self, horizon=1):
        if horizon == 1:
            self.last_pred = self.model.forward(self.sample_t)
            pred = [np.squeeze(self.last_pred).detach().numpy()]   
            pred = [min(max(i, self.lower_threshold), self.upper_threshold) for i in pred]
            self.predictions = np.append(self.predictions, pred[0])
            if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
                y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
                return y_pred, y_upper_i, y_lower_i 
            else:
                return pred, None, None

    def BPprocess(self, y):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        y_tensor = Variable(y)
        self.optimizer.zero_grad()
        loss = self.criterion(self.last_pred, y_tensor)
        # loss_sum += loss.item()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.BPprocess(np.array([y_truth]))
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num = 1, dropout = 0.0):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.dropout = dropout
        self.cell = nn.GRU(input_size = self.input_dim, hidden_size = self.hidden_dim,
                            num_layers = self.layer_num, dropout = self.dropout,
                            batch_first = True) 
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    def forward(self, x):
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 1, batchSize , self.hidden_dim))
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hidden_dim)
        fcOutput = self.fc(hn)
        return fcOutput

class GRU_test_CP:
    def __init__(self, init_data=None, input_dim=1, hidden_dim=20, output_dim=1, input_length=12, lr=1e-3, train_epoch=40,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        self.model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.input_length = input_length
        self.lr = lr
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = nn.MSELoss()
        self.train_epoch = train_epoch
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        trainX, trainY = [], []
        train_size = len(init_data)
        for t in range(train_size - self.input_length):
            sampleX = init_data[t : self.input_length + t, :]
            sampleY = init_data[t + self.input_length, :]
            trainX.append(sampleX)
            trainY.append(sampleY)
        dataset = timeseries(trainX, trainY)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                                 num_workers=1)
        for x in range(self.train_epoch):
            # loss_epoch = 0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = Variable(x)
                self.last_pred = self.model.forward(x)
                self.BPprocess(y)
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)

    def fit(self, sample):
        sample_np = np.array(sample[-1 * self.input_length:]).reshape(-1, self.input_length, 1)
        self.sample_t = torch.from_numpy(sample_np)
        self.sample_t = Variable(self.sample_t)

    def predict(self, horizon=1):
        if horizon == 1:
            self.last_pred = self.model.forward(self.sample_t)
            pred = [np.squeeze(self.last_pred).detach().numpy()]     
            pred = [min(max(i, self.lower_threshold), self.upper_threshold) for i in pred]
            self.predictions = np.append(self.predictions, pred[0])
            if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
                y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
                return y_pred, y_upper_i, y_lower_i 
            else:
                return pred, None, None

    def BPprocess(self, y):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        y_tensor = Variable(y)
        self.optimizer.zero_grad()
        loss = self.criterion(self.last_pred, y_tensor)
        # loss_sum += loss.item()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.BPprocess(np.array([y_truth]))
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    
class Naive:
    def __init__(self, seasonal_period=24):
        self.seasonal_period = seasonal_period
        self.last_sample = None
        self.predictions = np.empty(0)
    def fit(self, sample):
        self.last_sample = sample
    def predict(self, horizon):
        pred = self.last_sample[-1 * self.seasonal_period:-1 * self.seasonal_period + horizon]
        self.predictions = np.append(self.predictions, pred[0])
        return pred

class Naive_CP:
    def __init__(self,seasonal_period=24, warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.last_sample = None
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.last_sample = np.array(sample).reshape(-1,)
        
    def predict(self, horizon):
        pred = self.last_sample[-1 * self.seasonal_period:-1 * self.seasonal_period + horizon]
        pred = min(max(pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):

class NaiveDrift:
    def __init__():
        self.first, self.last, self.length = None, None, None
    def fit(sample):
        self.first = sample[0]
        self.last = sample[-1]
        self.length = len(sample)
    def predict(horizon):
        return [self.last + (i + 1) * (self.last - self.first) / (self.length - 1) for i in range(horizon)]
    
class NaiveDrift_CP:
    def __init__(self, warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        self.first, self.last, self.length = None, None, None
    
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
    
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
    
    def fit(self, sample):
        self.first = sample[0]
        self.last = sample[-1]
        self.length = len(sample)
    def predict(self, horizon=1):
        pred = [self.last + (i+1) * (self.last - self.first) / (self.length - 1) for i in range(horizon)]
        pred = [min(max(i, self.lower_threshold), self.upper_threshold) for i in pred]
        self.predictions = np.append(self.predictions, pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

class Prophet_fb:
    def __init__(self, start_time=pd.to_datetime("2022-05-31 00:00:00"), time_granularity="H",step_wise=True,init_data=None):
        self.start_time = start_time
        self.time_granularity = time_granularity
        self.step_wise = step_wise
        self.data=pd.DataFrame(columns=["ds","y"])
        self.future=None
        if step_wise == True:
            for i in range(len(init_data) - 1):
                if self.time_granularity == "H":
                    self.data = self.data.append({'ds': self.start_time+np.timedelta64(i, 'h'), 'y': init_data[i][0]}, ignore_index=True)
            # print(self.data)
    def fit(self, sample, start_time=None):
        if not self.step_wise:
            df_test = pd.DataFrame(columns=["ds","y"])
            for i in range(len(sample)):
                if self.time_granularity == "H":
                    da_test = df_test.append({'ds': start_time+np.timedelta64(i, 'h'), 'y': sample[i][0]}, ignore_index=True)
            self.data = df_test
        else:
            if(self.data.shape[0]>48):
                self.data = self.data.drop(index=[0])
            self.data = self.data.append({'ds': self.data["ds"].values[-1]+np.timedelta64(1, 'h'), 'y': sample[-1][0]}, ignore_index=True)
        # print(self.data)
        # self.model.fit(self.data)
    def predict(self, horizon=1):
        with suppress_stdout_stderr():
            self.model = Prophet(uncertainty_samples=None)
            self.model.fit(self.data)
            self.future = self.model.make_future_dataframe(periods=horizon,freq=self.time_granularity)
            forecast = self.model.predict(self.future)
            if(forecast["yhat"].values[-1]>200000):
                self.data_error = self.data
            # print(forecast)
            return [forecast["yhat"].values[-1*i] for i in range(1,horizon+1)]
        
class Prophet_fb_CP:
    def __init__(self, start_time, time_granularity="H",step_wise=True,init_data=None,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        self.start_time = pd.to_datetime(start_time)
        self.time_granularity = time_granularity
        self.step_wise = step_wise
        self.data=pd.DataFrame(columns=["ds","y"])
        self.future=None
        if step_wise == True:
            for i in range(len(init_data) - 1):
                if self.time_granularity == "H":
                    self.data = self.data.append({'ds': self.start_time+np.timedelta64(i, 'h'), 'y': init_data[i][0]}, ignore_index=True)
                elif self.time_granularity == "D":
                    self.data = self.data.append({'ds': self.start_time+np.timedelta64(i, 'D'), 'y': init_data[i][0]}, ignore_index=True)
                elif self.time_granularity == "T":
                    self.data = self.data.append({'ds': self.start_time+np.timedelta64(i, 'm'), 'y': init_data[i][0]}, ignore_index=True)
<<<<<<< HEAD
                elif self.time_granularity == "S":
                    self.data = self.data.append({'ds': self.start_time+np.timedelta64(i, 'm'), 'y': init_data[i][0]}, ignore_index=True)
=======
>>>>>>> d71b8ae5951d40bd6752d4bc0475319152b70f14
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        # self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
        
    def fit(self, sample, start_time=None):
        if not self.step_wise:
            df_test = pd.DataFrame(columns=["ds","y"])
            for i in range(len(sample)):
                if self.time_granularity == "H":
                    da_test = df_test.append({'ds': start_time+np.timedelta64(i, 'h'), 'y': sample[i][0]}, ignore_index=True)
                elif self.time_granularity == "D":
                    da_test = df_test.append({'ds': start_time+np.timedelta64(i, 'D'), 'y': sample[i][0]}, ignore_index=True)
                elif self.time_granularity == "T":
                    da_test = df_test.append({'ds': start_time+np.timedelta64(i, 'm'), 'y': sample[i][0]}, ignore_index=True)
<<<<<<< HEAD
                elif self.time_granularity == "S":
                    self.data = self.data.append({'ds': self.data["ds"].values[-1]+np.timedelta64(1, 's'), 'y': sample[-1][0]}, ignore_index=True)
=======
>>>>>>> d71b8ae5951d40bd6752d4bc0475319152b70f14
            self.data = df_test
        else:
            if(self.data.shape[0]>48):
                self.data = self.data.drop(index=[0])
            if(self.time_granularity == "H"):
                self.data = self.data.append({'ds': self.data["ds"].values[-1]+np.timedelta64(1, 'h'), 'y': sample[-1][0]}, ignore_index=True)
            elif(self.time_granularity == "D"):
                self.data = self.data.append({'ds': self.data["ds"].values[-1]+np.timedelta64(1, 'D'), 'y': sample[-1][0]}, ignore_index=True)
            elif self.time_granularity == "T":
                self.data = self.data.append({'ds': self.data["ds"].values[-1]+np.timedelta64(1, 'm'), 'y': sample[-1][0]}, ignore_index=True)
<<<<<<< HEAD
            elif self.time_granularity == "S":
                self.data = self.data.append({'ds': self.data["ds"].values[-1]+np.timedelta64(1, 's'), 'y': sample[-1][0]}, ignore_index=True)

=======
>>>>>>> d71b8ae5951d40bd6752d4bc0475319152b70f14
        # print(self.data)
        # self.model.fit(self.data)
    def predict(self, horizon=1):
        with suppress_stdout_stderr():
            self.model = Prophet(uncertainty_samples=None)
            self.model.fit(self.data)
            self.future = self.model.make_future_dataframe(periods=horizon,freq=self.time_granularity)
            forecast = self.model.predict(self.future)
            pred = [forecast["yhat"].values[-1*i] for i in range(1,horizon+1)]
            pred = [min(max(i, self.lower_threshold), self.upper_threshold) for i in pred]
            self.predictions = np.append(self.predictions, pred[0])
            if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
                y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
                return y_pred, y_upper_i, y_lower_i 
            else:
                return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
        
        
class Prophet_df:
    def __init__(self, time_granularity="H"):
        self.data = None
        self.time_granularity = time_granularity
    def fit(self, sample_df):
        self.data = sample_df
        # print(self.data)
    def predict(self, horizon=1):
        with suppress_stdout_stderr():
            self.model = Prophet(uncertainty_samples=None)
            self.model.fit(self.data)
            self.future = self.model.make_future_dataframe(periods=horizon,freq=self.time_granularity)
            forecast = self.model.predict(self.future)
            # print(forecast)
            return [forecast["yhat"].values[-1*i] for i in range(1,horizon+1)]


    
class AutoARIMA:
    def __init__(self, Conformal = False, CP_method = "ACP"):
        '''

        '''
        self.model = None
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)

    def fit(self, sample):
        self.model = auto_arima(sample)
        
    def predict(self, horizon):
        pred = self.model.predict(horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        
        return pred    

class AutoARIMA_CP:
    def __init__(self, warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = auto_arima(sample)
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.model.predict(horizon)
        pred = min(max(pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    

    
    
class ARIMA:
    def __init__(self,p=1, d=1, q=1, Conformal = False, CP_method = "ACP"):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        self.p, self.d, self.q = p, d, q
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        # self.model = auto_arima
    def fit(self, sample):
        # self.model = auto_arima(sample)
        # self.model = sm.tsa.statespace.SARIMAX(sample, order=(p, d, q))
        self.model = sm.tsa.arima.model.ARIMA(sample, order=(self.p, self.d, self.q))
        self.training_res = self.model.fit()
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        return pred
    
class ARIMA_CP:
    def __init__(self, p=1, d=1, q=1,warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        self.p, self.d, self.q = p, d, q
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        # self.model = auto_arima(sample)
        # self.model = sm.tsa.statespace.SARIMAX(sample, order=(p, d, q))
        self.model = sm.tsa.arima.model.ARIMA(sample, order=(self.p, self.d, self.q))
        self.training_res = self.model.fit()
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        pred = min(max(pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    
class SARIMA:
    def __init__(self,p=1, d=1, q=1,seasonal_order=(0,0,0,0)):
        '''
        SARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX
        '''
        self.model = None
        self.training_res = None
        self.p, self.d, self.q = p, d, q
        self.seasonal_order = seasonal_order
        self.predictions = np.empty(0)
        
        # self.model = auto_arima
    def fit(self, sample):
        # self.model = auto_arima(sample)
        self.model = sm.tsa.statespace.sarimax.SARIMAX(sample, 
                                                       order=(self.p, self.d, self.q),
                                                      initialization='approximate_diffuse')
        self.training_res = self.model.fit(disp=0)
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        return pred
    
class SARIMA_CP:
    def __init__(self,p=1, d=1, q=1,seasonal_order=(0,0,0,0), warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        self.p, self.d, self.q = p, d, q   
        self.seasonal_order = seasonal_order
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.statespace.sarimax.SARIMAX(sample, 
                                                       order=(self.p, self.d, self.q),
                                                      initialization='approximate_diffuse')
        self.training_res = self.model.fit(disp=0)
        
    def predict(self, horizon):
        pred = self.training_res.forecast(steps=horizon)
        pred = min(max(pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
class SARIMAX:
    '''
    SARIMAX模型
    支持概率预测，可以产生一定置信度的预测区间
    详情参考：
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX
    '''
    def __init__(self,p=1, d=1, q=1, seasonal_order=(0,0,0,0)):
        self.model = None
        self.training_res = None
        self.p, self.d, self.q = p, d, q
        self.seasonal_order = seasonal_order
        self.predictions = np.empty(0)
        # self.model = auto_arima
    def fit(self, sample):
        # self.model = auto_arima(sample)
        # self.model = sm.tsa.statespace.SARIMAX(sample, order=(p, d, q))
        self.model = sm.tsa.statespace.sarimax.SARIMAX(sample, order=(self.p, self.d, self.q), 
                                                      seasonal_order=self.seasonal_order)
        self.training_res = self.model.fit(disp=0)
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        return pred
    

class SARIMAX_CP:
    def __init__(self,p=1, d=1, q=1,seasonal_order=(0,0,0,0), warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        self.p, self.d, self.q = p, d, q   
        self.seasonal_order = seasonal_order
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.statespace.sarimax.SARIMAX(sample, order=(self.p, self.d, self.q), 
                                                      seasonal_order=self.seasonal_order)
        self.training_res = self.model.fit(disp=0)
        
    def predict(self, horizon):
        pred = self.training_res.forecast(steps=horizon)
        pred = min(max(pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):    

class ETS:
    def __init__(self, trend=None, damped_trend=False,
                 seasonal=None, seasonal_periods=None, initialization_method='estimated'):
        '''
        指数平滑模型，接口来自statsmodels
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html#statsmodels.tsa.holtwinters.ExponentialSmoothing.fit
        https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
        '''
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.model = None
        self.training_res = None
        self.predictions = np.empty(0)
        
    def fit(self, sample):
        self.model = sm.tsa.holtwinters.ExponentialSmoothing(sample, trend = self.trend,
                                                            damped_trend = self.damped_trend,
                                                            seasonal = self.seasonal,
                                                            seasonal_periods = self.seasonal_periods,
                                                    initialization_method = self.initialization_method)
        self.training_res = self.model.fit()
        
        return self.training_res
    def predict(self, horizon):
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        return pred

class ETS_CP:
    def __init__(self,trend=None, damped_trend=False,seasonal=None, seasonal_periods=None, initialization_method='estimated', warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.holtwinters.ExponentialSmoothing(sample, trend = self.trend,
                                                            damped_trend = self.damped_trend,
                                                            seasonal = self.seasonal,
                                                            seasonal_periods = self.seasonal_periods,
                                                    initialization_method = self.initialization_method)
        self.training_res = self.model.fit()
        
    def predict(self, horizon):
        pred = self.training_res.forecast(steps=horizon)
        pred = min(max(pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
    
    
class ARDL:
    '''
    自回归滞后变量模型
    详情参考：
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.ardl.ARDL.html#statsmodels.tsa.ardl.ARDL
    '''
    def test(self):
        print("hello world!")

    
# class GPR:
#     def __init__(self, kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * 
#   ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)), alpha=1e-10, step = 1, seasonal = False, seasonal_period=np.inf):
#         '''
#         高斯过程回归模型，接口来自sklearn
#         kernel：高斯过程使用的核函数，可选项：WhiteKernel, ExpSineSquared, ConstantKernel, RBF, DotProduct, Matern, RationalQuadratic，详情参考https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
#         '''
#         self.kernel = kernel
#         self.alpha = 1e-10
#         self.step = step
#         self.last_sample = None
#         self.seasonal = seasonal
#         self.seasonal_period = seasonal_period
#     def fit(self, sample):
#         self.model = sklearn.gaussian_process.GaussianProcessRegressor(kernel=self.kernel, 
#                                                                        n_restarts_optimizer=1, 
#                                                                        normalize_y=True,
#                                                                        alpha=self.alpha)
#         if self.seasonal == False and self.step > 0:
#             sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
#             sample_y = sample[self.step:len(sample)]
#             self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
#         else:
#             sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
#             sample_y = sample
#             self.last_sample = [[len(sample) % self.seasonal_period]]
#         self.model.fit(sample_x, sample_y)
        
#         return self.model
#     def predict(self, horizon):
#         y_pred, y_std = self.model.predict(self.last_sample, return_std=True)
        
#         return y_pred[0]
    
class GPR:
    def __init__(self, kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * 
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)), alpha=1e-10, step = 1, seasonal = True, seasonal_period=24):
        '''
        高斯过程回归模型，接口来自sklearn
        kernel：高斯过程使用的核函数，可选项：WhiteKernel, ExpSineSquared, ConstantKernel, RBF, DotProduct, Matern, RationalQuadratic，详情参考https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
        '''
        self.kernel = kernel
        self.alpha = 1e-10
        self.step = step
        self.last_sample = None
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.predictions = np.empty(0)

    def fit(self, sample):
        self.model = sklearn.gaussian_process.GaussianProcessRegressor(kernel=self.kernel, 
                                                                       n_restarts_optimizer=1, 
                                                                       normalize_y=True,
                                                                       alpha=self.alpha)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample, return_std=True)
        self.predictions = np.append(self.predictions, y_pred[0,0])

        return y_pred[0]

class GPR_CP:
    def __init__(self, kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * 
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)), gpr_alpha=1e-10, step = 1, seasonal = True, seasonal_period=24,
                 warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        self.kernel = kernel
        self.gpr_alpha = 1e-10
        self.step = step
        self.last_sample = None
        self.seasonal = seasonal
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = GaussianProcessRegressor(kernel=self.kernel, 
                                                                       n_restarts_optimizer=1, 
                                                                       normalize_y=True,
                                                                       alpha=self.gpr_alpha)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample, return_std=True)
        y_pred=y_pred.reshape(-1,1)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        
        self.predictions = np.append(self.predictions, y_pred[0,0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred[0])
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred[0], None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
class Bagging_sklearn:
    def __init__(self, base_model="GPR", n_estimators=10, seasonal=False, step=24, seasonal_period=24):
        '''
        Bagging模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
        '''
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step

    def fit(self, sample):
        if self.base_model == "GPR":
            base_model = GaussianProcessRegressor(kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * 
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)),n_restarts_optimizer=1,normalize_y=True,alpha=1e-10)
        elif self.base_model == "DTR":
            base_model = DecisionTreeRegressor(max_depth=7)
        elif self.base_model == "SVR":
            base_model = SVR()
        self.model = BaggingRegressor(base_estimator=base_model, n_estimators=self.n_estimators)

        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample)
        self.predictions = np.append(self.predictions, y_pred[0,0])

        return y_pred[0]
    
class Bagging_CP:
    def __init__(self, base_model="DTR", n_estimators=3, seasonal=False, step=24, seasonal_period=24,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        SVR模型，接口来自sklearn
        '''
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    def fit(self, sample):
        if self.base_model == "GPR":
            base_model = GaussianProcessRegressor(kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * 
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)),n_restarts_optimizer=1,normalize_y=True,alpha=1e-10)
        elif self.base_model == "DTR":
            base_model = DecisionTreeRegressor(max_depth=7)
        self.model = BaggingRegressor(base_estimator=base_model, n_estimators=self.n_estimators)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred= self.model.predict(self.last_sample)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    
class SVR_CP:
    def __init__(self, kernel="rbf", svr_gamma="auto", C=100, epsilon=0.1, seasonal=False, step=24, seasonal_period=24,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        SVR模型，接口来自sklearn
        详见：
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
        https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
        '''
        self.kernel = kernel
        self.svr_gamma = svr_gamma
        self.C = C
        self.epsilon = epsilon
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    def fit(self, sample):
        self.model = SVR(kernel=self.kernel, C=self.C, gamma=self.svr_gamma, epsilon=self.epsilon)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred= self.model.predict(self.last_sample)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
        

class DTR_sklearn:
    def __init__(self, max_depth=3, seasonal=False, step=24, seasonal_period=24):
        '''
        DTR模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
        '''
        self.max_depth = max_depth
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step

    def fit(self, sample):
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample)
        self.predictions = np.append(self.predictions, y_pred[0,0])

        return y_pred[0]
    
class DTR_CP:
    def __init__(self, max_depth=7, seasonal=False, step=24, seasonal_period=24,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        SVR模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
        '''
        self.max_depth = max_depth
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    def fit(self, sample):
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred= self.model.predict(self.last_sample)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
        
class RFR_sklearn:
    def __init__(self, n_estimators=100, seasonal=False, step=24, seasonal_period=24):
        '''
        DTR模型，接口来自sklearn
        '''
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step

    def fit(self, sample):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample)
        self.predictions = np.append(self.predictions, y_pred[0,0])

        return y_pred[0]
    
class RFR_CP:
    def __init__(self, n_estimators=100, seasonal=False, step=24, seasonal_period=24,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        RandomForest模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        '''
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    def fit(self, sample):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs=-1)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred= self.model.predict(self.last_sample)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    
class ETR_sklearn:
    def __init__(self, n_estimators=100, seasonal=False, step=24, seasonal_period=24):
        '''
        ExtraTreesRegressor模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor
        '''
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step

    def fit(self, sample):
        self.model = ExtraTreesRegressor(n_estimators=self.n_estimators)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample)
        self.predictions = np.append(self.predictions, y_pred[0,0])

        return y_pred[0]
    
class ETR_CP:
    def __init__(self, n_estimators=100, seasonal=False, step=24, seasonal_period=24,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        ExtraTreesRegressor模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor
        '''
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    def fit(self, sample):
        self.model = ExtraTreesRegressor(n_estimators=self.n_estimators, n_jobs=-1)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred= self.model.predict(self.last_sample)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)    
        
class GBR_sklearn:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls', seasonal=False, step=24, seasonal_period=24):
        '''
        GradientBoosting模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor        self.base_model = base_model
        '''
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.loss = loss
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step

    def fit(self, sample):
        self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,max_depth=self.max_depth,random_state=self.random_state,loss=self.loss)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample)
        self.predictions = np.append(self.predictions, y_pred[0,0])

        return y_pred[0]
    
class GBR_CP:
    def __init__(self,n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls' , seasonal=False, step=24, seasonal_period=24,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        GradientBoosting模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
        '''
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.loss = loss
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    def fit(self, sample):
        self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,max_depth=self.max_depth,random_state=self.random_state,loss=self.loss)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred= self.model.predict(self.last_sample)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)

class UCM:
    '''
    支持概率预测，可以产生一定置信度的预测区间
    详情参考：
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html#statsmodels.tsa.statespace.structural.UnobservedComponents
    '''
    def __init__(self,):
        self.model = None
        self.training_res = None
        self.predictions = np.empty(0)

        # self.model = auto_arima
    def fit(self, sample):
        # self.model = auto_arima(sample)
        # self.model = sm.tsa.statespace.SARIMAX(sample, order=(p, d, q))
        self.model = sm.tsa.statespace.structural.UnobservedComponents(sample, 'local level', cycle=True,
                                        damped_cycle=True,
                                        stochastic_cycle=True)
        self.training_res = self.model.fit(disp=0)
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])

        return pred

class Adaboost_sklearn:
    def __init__(self, base_model="GPR", n_estimators=10, seasonal=False, step=24, seasonal_period=24):
        '''
        Adaboost模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor        '''
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step

    def fit(self, sample):
        if self.base_model == "GPR":
            base_model = GaussianProcessRegressor(kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * 
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)),n_restarts_optimizer=1,normalize_y=True,alpha=1e-10)
        elif self.base_model == "DTR":
            base_model = DecisionTreeRegressor(max_depth=7)
        elif self.base_model == "SVR":
            base_model = SVR()
        self.model = AdaBoostRegressor(base_estimator=base_model, n_estimators=self.n_estimators)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred, y_std = self.model.predict(self.last_sample)
        self.predictions = np.append(self.predictions, y_pred[0,0])

        return y_pred[0]
    
class Adaboost_CP:
    def __init__(self, base_model="DTR", n_estimators=3, seasonal=False, step=24, seasonal_period=24,
                warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0):
        '''
        Adaboost模型，接口来自sklearn
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
        '''
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.step = step
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold

        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    def fit(self, sample):
        if self.base_model == "GPR":
            base_model = GaussianProcessRegressor(kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * 
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)),n_restarts_optimizer=1,normalize_y=True,alpha=1e-10)
        elif self.base_model == "DTR":
            base_model = DecisionTreeRegressor(max_depth=7)
        elif self.base_model == "SVR":
            base_model = SVR()
        self.model = AdaBoostRegressor(base_estimator=base_model, n_estimators=self.n_estimators)
        if self.seasonal is False and self.step > 0:
            sample_x = np.array([sample[i-self.step: i] for i in range(self.step,len(sample))]).reshape(-1,self.step)
            sample_y = sample[self.step:len(sample)]
            self.last_sample = np.array(sample[len(sample)-self.step:len(sample)]).reshape(1,self.step)
        else:
            sample_x = np.array([i % self.seasonal_period for i in range(len(sample))]).reshape(-1,1)
            sample_y = sample
            self.last_sample = [[len(sample) % self.seasonal_period]]
        self.model.fit(sample_x, sample_y)
        
        return self.model
    def predict(self, horizon):
        y_pred= self.model.predict(self.last_sample)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    


class UCM_CP:
    def __init__(self, 
                 warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        '''
        self.model = None
        self.training_res = None
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.statespace.structural.UnobservedComponents(sample, 'local level', cycle=True,
                                        damped_cycle=True,
                                        stochastic_cycle=True)
        self.training_res = self.model.fit(disp=0)
        
    def predict(self, horizon):
        y_pred = self.training_res.forecast(steps=horizon)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
class DynamicFactor:
    '''
    支持概率预测，可以产生一定置信度的预测区间
    详情参考：
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html#statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ
    '''
    def __init__(self,):
        self.model = None
        self.training_res = None
        self.predictions = np.empty(0)

        # self.model = auto_arima
    def fit(self, sample):
        # self.model = auto_arima(sample)
        # self.model = sm.tsa.statespace.SARIMAX(sample, order=(p, d, q))
        self.model = sm.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ(sample)
        self.training_res = self.model.fit(disp=0)
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        return pred
    
class DynamicFactor_CP:
    def __init__(self, 
                 warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        
        '''
        self.model = None
        self.training_res = None
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ(sample)
        self.training_res = self.model.fit(disp=0)
        
    def predict(self, horizon):
        y_pred = self.training_res.forecast(steps=horizon)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
class LinearETS:
    '''
    支持概率预测，可以产生一定置信度的预测区间
    详情参考：
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing.html#statsmodels.tsa.statespace.exponential_smoothing.ExponentialSmoothing
    '''
    def __init__(self,):
        self.model = None
        self.training_res = None
        self.predictions = np.empty(0)
        # self.model = auto_arima
    def fit(self, sample):
        # self.model = auto_arima(sample)
        # self.model = sm.tsa.statespace.SARIMAX(sample, order=(p, d, q))
        self.model = sm.tsa.statespace.exponential_smoothing.ExponentialSmoothing(sample)
        self.training_res = self.model.fit(disp=0)
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        return pred
    
class LinearETS_CP:
    def __init__(self, 
                 warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.statespace.exponential_smoothing.ExponentialSmoothing(sample)
        self.training_res = self.model.fit(disp=0)
        
    def predict(self, horizon):
        y_pred = self.training_res.forecast(steps=horizon)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
class ThetaModel:
    '''
    详情参考：
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.theta.ThetaModel.html#statsmodels.tsa.forecasting.theta.ThetaModel
    '''
    def __init__(self,):
        self.model = None
        self.training_res = None
        self.predictions = np.empty(0)
        # self.model = auto_arima
    def fit(self, sample):
        # self.model = auto_arima(sample)
        # self.model = sm.tsa.statespace.SARIMAX(sample, order=(p, d, q))
        self.model = sm.tsa.forecasting.theta.ThetaModel(sample,period=2,use_test=True)
        self.training_res = self.model.fit()
        
        return self.training_res
    def predict(self, horizon):
        # pred = self.model.predict(horizon)
        pred = self.training_res.forecast(steps=horizon)
        self.predictions = np.append(self.predictions, pred[0])
        
        return np.array(pred)

class ThetaModel_CP:
    def __init__(self, 
                 warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.forecasting.theta.ThetaModel(sample,period=2,use_test=True)
        self.training_res = self.model.fit()
        
    def predict(self, horizon):
        y_pred = self.training_res.forecast(steps=horizon)
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        self.predictions = np.append(self.predictions, y_pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
class Markov:
    '''
    详情参考：
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html#statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression
    '''
    def __init__(self,):
        self.model = None
        self.training_res = None
        self.predictions = np.empty(0)
    def fit(self, sample):
        self.model = sm.tsa.regime_switching.markov_autoregression.MarkovAutoregression(sample,
                                                                                       k_regimes = 2,
                                                                                       order = 1)
        self.training_res = self.model.fit()
        
        return self.training_res
    def predict(self, horizon):
        pred = self.training_res.predict()
        self.predictions = np.append(self.predictions, pred[0])
        
        return np.array(pred)

class Markov_CP:
    def __init__(self, 
                 warm_up_data=None, look_back = 30, Conformal = False, CP_method = "ACP",
                alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        '''
        ARIMA模型，接口来自statsmodels
        支持概率预测，可以产生一定置信度的预测区间
        详情参考：
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
        https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
        '''
        self.model = None
        self.training_res = None
        
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.predictions = np.empty(0)
        self.CP_module = None
        self.warm_up_data = warm_up_data
        self.look_back = look_back
        self.warm_up_length = len(self.warm_up_data)
        self.alpha = alpha
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        # self.model = auto_arima
        if self.Conformal:
            for i in range(len(self.warm_up_data)-self.look_back):
                self.fit(self.warm_up_data[i : i+self.look_back])
                pred,_,_ = self.predict(1)
            self.CP_module = CPmodule(self.warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
    
    def fit(self, sample):
        self.model = sm.tsa.regime_switching.markov_autoregression.MarkovAutoregression(sample,
                                                                                       k_regimes = 2,
                                                                                       order = 1)
        self.training_res = self.model.fit()
        
    def predict(self, horizon):
        y_pred = self.training_res.predict()
        y_pred = min(max(y_pred, self.lower_threshold), self.upper_threshold)
        # print(y_pred)
        self.predictions = np.append(self.predictions, y_pred[0])
        if self.Conformal and self.predictions.shape[0] > (self.warm_up_length-self.look_back):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        else:
            return y_pred, None, None
    def update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        self.CP_module.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
    # def train(self, sample, y_truth):
    
class EnsembleModel:
    def __init__(self, model_number = 6, models = ["GPR","ARIMA","ETS","UCM","SARIMAX","Naive"], choice_interval = 30, method = "TFPL"):
        self.model_number = model_number
        self.models = []
        self.counter = 0
        self.choice_interval = choice_interval
        self.model_number = model_number
        self.predictions = [np.array([]) for i in range(model_number+2)]
        
        self.errors = [np.array([]) for i in range(model_number+2)]
        self.choice = random.randint(0, model_number-1)
        self.choice_series = np.array([])
        self.method = method
        self.greedy_choice = self.choice
        self.e_greedy_choice = self.choice
        self.A, self.R, self.T, self.D = 0., 0., 0., 2
        self.recent_A, self.recent_R = np.array([]), np.array([])
        for model in models:
            if(model == "GPR"):
                model_ = GPR(seasonal = True, seasonal_period = 24)
                self.models.append(model_)
            elif(model == "ARIMA"):
                model_ = SARIMA()
                self.models.append(model_)
            elif(model == "SARIMA"):
                model_ = SARIMA()
                self.models.append(model_)
            elif model == 'ETS':
                model_ = ETS()
                self.models.append(model_)
            elif model == 'UCM':
                model_ = UCM()
                self.models.append(model_)
            elif model == 'SARIMAX':
                model_ = SARIMAX()
                self.models.append(model_)
            elif model == "Naive":
                model_ = Naive()
                self.models.append(model_)
            elif model == "AutoARIMA":
                model_ = AutoARIMA()
                self.models.append(model_)             
            
            
    def fit(self, sample):
        for model_id in range(self.model_number):
            self.models[model_id].fit(sample)

    def predict(self, horizon = 1):
        for model_id in range(self.model_number):
            pred = self.models[model_id].predict(horizon)
            self.predictions[model_id] = np.append(self.predictions[model_id], pred)
        self.predictions[-2] = np.append(self.predictions[-1], self.predictions[self.greedy_choice][-1])
        self.predictions[-1] = np.append(self.predictions[-1], self.predictions[self.e_greedy_choice][-1])
        self.choice_series = np.append(self.choice_series, self.choice)
        return [self.predictions[self.choice][-1]]
    def update(self, y_truth):
        A_tem = 0
        if(0):
            # print(y_truth)
            for model_id in range(self.model_number):
                error = np.abs(y_truth - self.predictions[model_id][-1])
                self.R = max(self.R, error[0])
                A_tem += error[0]
                # print(A_tem)
                self.errors[model_id] = np.append(self.errors[model_id], error)
        else:
            errors_tmp =np.array([])
            for model_id in range(self.model_number):
                error = np.abs(y_truth - self.predictions[model_id][-1])
                self.R = max(self.R, error[0])
                A_tem += error[0]
                # print(A_tem)
                errors_tmp = np.append(errors_tmp, error)
            sorted_errors = sorted(set(errors_tmp))
            dic_errors = {x:(i/(len(errors_tmp)-1)) for i, x in enumerate(sorted_errors)}
            for model_id in range(self.model_number):
                self.errors[model_id] = np.append(self.errors[model_id], dic_errors[errors_tmp[model_id]])
        self.errors[-2] = np.append(self.errors[-1], self.errors[self.greedy_choice][-1])
        self.errors[-1] = np.append(self.errors[-1], self.errors[self.e_greedy_choice][-1])
        self.counter += 1
        # self.windows = np.append(self.windows, window)
        self.T += 1
        self.recent_A = np.append(self.recent_A, A_tem)
        if self.recent_A.shape[0] > self.choice_interval:
            self.recent_A = self.recent_A[1:]
        self.A = max(self.recent_A)
        if(self.counter == self.choice_interval):
            self.update_choice(method = self.method)
            self.counter = 0
            
    def update_choice(self, method = "FTPL", epsilon = 0.9):
        look_back = 30
        # for errors in self.errors:
        #     print(errors)
        recent_errors = [np.sum(errors[:]) for errors in self.errors]
        # print(f"recent_error:{recent_errors}")
        min_index, min_number = min(enumerate(recent_errors[:-2]), key=operator.itemgetter(1))
        randnum = random.random()
        if(randnum < (self.model_number * epsilon - 1) / (self.model_number - 1)):
            self.e_greedy_choice = min_index
        else:
            self.e_greedy_choice = random.randint(0, self.model_number - 1)
        self.greedy_choice = min_index
        if method == "epsilon-greedy":
            self.choice = self.e_greedy_choice
        elif method == 'greedy':
            self.choice = self.greedy_choice
        elif method == 'FTPL':
            Disturb = [ self.T / self.T * np.random.exponential(  ( (self.R*self.A*(self.T+1))  /self.D ) **0.5)    for i in range(self.model_number + 2)]
            # print(f"扰动为：{Disturb}, 原始error为：{recent_errors}")
            p = [recent_errors[i] - Disturb[i] for i in range(self.model_number + 1)]
            min_index, min_number = min(enumerate(p), key=operator.itemgetter(1))
            self.choice = min_index

class EnsembleModule():
<<<<<<< HEAD
    def __init__(self, predict_models,model_names,alpha, loss_fuc="rank",warm_up_data=None, look_back = 48, choice_interval=30, ensemble_method = "FTPL", Conformal = False, CP_method = "ACP", gamma = 0.05, tau = 0.5, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
=======
    def __init__(self, predict_models,model_names,loss_fuc="rank",warm_up_data=None, look_back = 48, choice_interval=24, ensemble_method = "FTPL", Conformal = False, CP_method = "ACP",
                    alpha = 0.1, gamma = 0.05, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
>>>>>>> d71b8ae5951d40bd6752d4bc0475319152b70f14
        if not (warm_up_data is None):
             warm_up_length = len(warm_up_data)
        self.model_number = len(model_names)
        self.Conformal = Conformal
        self.CP_method = CP_method
        self.loss_fuc = loss_fuc
        self.CP_module = None
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.choice_interval=choice_interval
        self.seasonal_period = seasonal_period
        self.upper_threshold, self.lower_threshold = upper_threshold, lower_threshold
        
        self.counter = 0
        self.predictions = np.empty(0)
        self.greedy_predictions = np.empty(0)
        self.e_greedy_predictions = np.empty(0)
        self.look_back = look_back
        
        self.flag=False
        self.errors = [np.array([]) for i in range(self.model_number+3)]
        self.choice = random.randint(0, self.model_number-1)
        self.choice_series = np.array([])
        self.ensemble_method = ensemble_method
        self.greedy_choice = self.choice
        self.e_greedy_choice = self.choice
        self.en_choice = self.choice
        self.A, self.R, self.T, self.D = 0., 0., 0., 2
        self.recent_A, self.recent_R = np.array([]), np.array([])
        self.train_set_FFORMS=[ ]
        self.train_label_FFORMS=[ ]
        self.time_step=0
        self.current_window=[]
        if not (warm_up_data is None):
            for i in trange(0, warm_up_length-self.look_back):
                self.time_step=i
                y_pred,_,_ =self.predict(predict_models=predict_models,model_names=model_names,horizon=1, time_step=i)
                if self.ensemble_method in ['FFORMS','FFORMA']:
                    for t in warm_up_data[i:self.look_back+i]:
                            self.current_window=np.append(self.current_window,t[0])
                self.update(y_truth=warm_up_data[self.look_back+i],predict_models=predict_models,model_names = model_names, time_step=i)
                
                #保持记录序列与对应的最佳模型
                if self.ensemble_method in ['FFORMS','FFORMA'] and len(self.errors[0])>=self.look_back:
                    new_errors = [np.sum(errors[-self.look_back:]) for errors in self.errors]
                    min_index, min_number = min(enumerate(new_errors[:-3]), key=operator.itemgetter(1))
                    label=min_index
                    if  i <= 2* self.look_back:
                        self.train_set_FFORMS.append(self.current_window)
                        self.train_label_FFORMS.append(label)
                    else:
                        self.train_set_FFORMS.pop(0)
                        self.train_set_FFORMS.append(self.current_window)
                        self.train_label_FFORMS.pop(0)
                        self.train_label_FFORMS.append(label)
                self.current_window=[]
                
        if self.Conformal and not(warm_up_data is None):
            
            self.CP_module = CPmodule(warm_up_data, self.predictions, method=self.CP_method, alpha=self.alpha,
                                     gamma=self.gamma, look_back=self.look_back, upper_threshold=self.upper_threshold,
                                     lower_threshold=self.lower_threshold)
            
            # for i in trange(0, warm_up_length-self.look_back):
            #     y_pred,y_upper,y_lower =self.predict(predict_models=predict_models,model_names=model_names,horizon=1, time_step=i)
                
                
    def predict(self, predict_models,model_names,horizon=1, time_step=-1):
        if  self.ensemble_method=='FFORMA' and self.flag: #FFORMA for model combination
            y_pred=0
            for i in range(self.model_number):
                y_pred_tmp=predict_models[model_names[i]][time_step:time_step+horizon]
                y_pred+=self.choice[i]*y_pred_tmp
        elif self.choice < self.model_number:
            y_pred = predict_models[model_names[self.choice]][time_step:time_step+horizon]
        elif self.choice == self.model_number:
            self.choice = self.greedy_choice
            y_pred = predict_models[model_names[self.greedy_choice]][time_step:time_step+horizon]
        elif self.choice == self.model_number + 1:
            self.choice = self.e_greedy_choice
            y_pred = predict_models[model_names[self.e_greedy_choice]][time_step:time_step+horizon]
        elif self.choice == self.model_number + 2:
            self.choice = self.en_choice
            y_pred = predict_models[model_names[self.en_choice]][time_step:time_step+horizon]

            
        self.predictions = np.append(self.predictions, y_pred[0])
        self.choice_series = np.append(self.choice_series, self.choice)
        
        if self.Conformal and not (self.CP_module is None):
            y_pred, y_upper_i, y_lower_i = self.CP_module.online_predict(y_pred)
            return y_pred, y_upper_i, y_lower_i 
        
        return y_pred,None,None
    
    def update(self, y_truth, predict_models,model_names, time_step=-1):
        A_tem = 0
        if(not self.loss_fuc == "rank"):
            # print(y_truth)
            for model_id in range(self.model_number):
                if y_truth > predict_models[model_names[model_id]][time_step]:
                    error = 2 * self.tau * (y_truth - predict_models[model_names[model_id]][time_step])
                else:
                    error = 2 * (1-self.tau) * (predict_models[model_names[model_id]][time_step] - y_truth)
                # error = np.abs(y_truth - predict_models[model_names[model_id]][time_step])
                self.R = max(self.R, error[0])
                A_tem += error[0]
                self.errors[model_id] = np.append(self.errors[model_id], error)
        else:
            errors_tmp =np.array([])
            for model_id in range(self.model_number):
                if y_truth > predict_models[model_names[model_id]][time_step]:
                    error = 2 * self.tau * (y_truth - predict_models[model_names[model_id]][time_step])
                else:
                    error = 2 * (1-self.tau) * (predict_models[model_names[model_id]][time_step] - y_truth)
                # error = np.abs(y_truth - predict_models[model_names[model_id]][time_step])
                self.R = max(self.R, error[0])
                A_tem += error[0]
                errors_tmp = np.append(errors_tmp, error)
            sorted_errors = sorted(set(errors_tmp))
            dic_errors = {x:(i/(len(errors_tmp)-1)) for i, x in enumerate(sorted_errors)}
            for model_id in range(self.model_number):
                self.errors[model_id] = np.append(self.errors[model_id], dic_errors[errors_tmp[model_id]])
        self.errors[-3] = np.append(self.errors[-1], self.errors[self.greedy_choice][-1])
        self.errors[-2] = np.append(self.errors[-1], self.errors[self.e_greedy_choice][-1])
        self.errors[-1] = np.append(self.errors[-1], self.errors[self.en_choice][-1])
        self.counter += 1
        # self.windows = np.append(self.windows, window)
        self.T += 1
        self.recent_A = np.append(self.recent_A, A_tem)
        if self.recent_A.shape[0] > self.choice_interval:
            self.recent_A = self.recent_A[1:]
        self.A = max(self.recent_A)
        if(self.counter == self.choice_interval):
            self.update_choice(method = self.ensemble_method)
            self.counter = 0
            

    def update_choice(self, method = "FTPL", epsilon = 0.9):
        look_back = 30
        # for errors in self.errors:
        #     print(errors)
        recent_errors = [np.sum(errors) for errors in self.errors]
        # print(f"recent_error:{recent_errors}")
        min_index, min_number = min(enumerate(recent_errors[:-3]), key=operator.itemgetter(1))
        randnum = random.random()
        if(randnum < (self.model_number * epsilon - 1) / (self.model_number - 1)):
            self.e_greedy_choice = min_index
        else:
            self.e_greedy_choice = random.randint(0, self.model_number - 1)
        self.greedy_choice = min_index
        if method in ["FTPL", "FFORMS", "FFORMA"]:
            if len(self.train_set_FFORMS) <=2:
                self.en_choice=self.greedy_choice
            else:
                #错位训练以：使用当前窗口序列预测下一个序列对应的最佳模型
                train_set = lgb.Dataset(data=np.array(self.train_set_FFORMS[:-1]), label=np.array(self.train_label_FFORMS[1:]).reshape(-1) )
                fforms_params = {"boosting_type": "gbdt",
                     "objective": "multiclass",
                     "num_class": self.model_number,
                     "num_leaves": 31,
                     "learning_rate": 0.01,
                     "feature_fraction": 0.9,
                     "bagging_fraction": 0.8,
                     "bagging_freq": 5,
                      "verbosity": -1}
                gbm = lgb.train(fforms_params,
                        train_set=train_set,
                        num_boost_round=500)
                self.en_choice = np.argmax(gbm.predict(np.array(self.current_window).reshape(1,-1)))
        if method == "epsilon-greedy":
            self.choice = self.e_greedy_choice
        elif method == 'greedy':
            self.choice = self.greedy_choice
        elif method == 'FTPL':
            Disturb = [ self.look_back / self.T * np.random.exponential(  ( (self.R*self.A*(self.T+1))  /self.D ) **0.5)    for i in range(self.model_number + 2)]
            # print(f"扰动为：{Disturb}, 原始error为：{recent_errors}")
            error_pred = [np.quantile(errors,0.5) for errors in self.errors]
            # 用预测器/误差分位数？
            p = [recent_errors[i] - Disturb[i] + error_pred[i] for i in range(self.model_number + 2)]
            min_index_after, min_number_after = min(enumerate(p), key=operator.itemgetter(1))
            self.choice = min_index_after
            # print(f"最终选择的模型id是：{self.choice}/{min_index}。")
        elif method == 'FFORMS':
            self.choice = self.en_choice
        elif method == 'FFORMA':
            if len(self.train_set_FFORMS) <=2:
                self.choice=self.greedy_choice
            else:
                self.flag=True
                self.choice=gbm.predict(np.array(self.current_window).reshape(1,-1))[0]
        
        
    
        
def Forcecsat_test(CP_models, predict_method, look_back, warm_up_length, data, lower_threshold, alpha=0.1, gamma=0.05, method = "ACP", ensemble_method = "greedy"):
    predict_model = None
    if predict_method == "Ensemble":
        predict_model = EnsembleModel(method = ensemble_method)
    elif predict_method == 'ARIMA':
        predict_model = SARIMA()
    elif predict_method == 'GPR':
        predict_model = GPR(seasonal = True, seasonal_period = 24)
    elif predict_method == 'ETS':
        predict_model = ETS()
    elif predict_method == 'UCM':
        predict_model = ETS()
    elif predict_method == 'DynamicFactor':
        predict_model = DynamicFactor()
    CPmodel = ConformalPrediction(data[:warm_up_length], predict_model, look_back=look_back, method=method, alpha=alpha, gamma=gamma, lower_threshold=lower_threshold)
    model_name = predict_method+"-"+method
    if "Ensemble" in model_name:
        model_name += "-"+ensemble_method
    total_length = len(data)
    
    if "Ensemble" in model_name and ensemble_method =="FTPL":
        for i in trange(warm_up_length,total_length):
            CPmodel.online_train(data[i-look_back:i], data[i]) 
    else:
        for i in range(warm_up_length,total_length):
                CPmodel.online_train(data[i-look_back:i], data[i]) 
    CP_models[model_name] = CPmodel
    
            
def Forecast_test_ver2(CP_models,predict_models, predict_method, look_back, warm_up_length, data, lower_threshold,time_granularity, start_time=None, alpha=0.1, gamma=0.05, Conformal=True, CP_method = "ACP", ensemble_method = "greedy"):
    st_time = time.perf_counter()
    predict_model = None
    warnings.filterwarnings("ignore")

    if predict_method == 'ARIMA111':
        predict_model = SARIMA_CP(p=1,d=1,q=1,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA101':
        predict_model = SARIMA_CP(p=1,d=0,q=1,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA222':
        predict_model = SARIMA_CP(p=2,d=2,q=2,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA010':
        predict_model = SARIMA_CP(p=0,d=1,q=0,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA100':
        predict_model = SARIMA_CP(p=1,d=0,q=0,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA011':
        predict_model = SARIMA_CP(p=0,d=1,q=1,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA110':
        predict_model = SARIMA_CP(p=1,d=1,q=0,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA211':
        predict_model = SARIMA_CP(p=2,d=1,q=1,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA201':
        predict_model = SARIMA_CP(p=2,d=0,q=1,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ARIMA301':
        predict_model = SARIMA_CP(p=3,d=0,q=1,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'GPR_external':
        predict_model = GPR_CP(seasonal=True, seasonal_period=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method, alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'SVR':
        predict_model = SVR_CP(seasonal=False, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method, alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'SVR_external':
        predict_model = SVR_CP(seasonal=True, seasonal_period=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method, alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'GPR':
        predict_model = GPR_CP(seasonal=False, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'DTR':
        predict_model = DTR_CP(max_depth=7,seasonal=False, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ETS24':
        predict_model = ETS_CP(seasonal="add",seasonal_periods=24,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ETS12':
        predict_model = ETS_CP(seasonal="add",seasonal_periods=12,warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ETS':
        predict_model = ETS_CP(warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'Naive':
        predict_model = Naive_CP(warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'Naive':
        predict_model = Naive_CP(warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == "NaiveDrift":
        predict_model = NaiveDrift_CP(warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == "AutoARIMA":
        predict_model = AutoARIMA_CP(warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == "UCM":
        predict_model = UCM_CP(warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,
                    alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == "Prophet":
        predict_model = Prophet_fb_CP(time_granularity=time_granularity, start_time=start_time,init_data=data[:look_back],warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method,alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == "LSTM":
        predict_model = LSTM_test_CP(train_epoch=100, lr=1e-4, input_length=12,init_data=data[:look_back],warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method, alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == "GRU":
        predict_model = GRU_test_CP(train_epoch=100, lr=1e-4, input_length=12,init_data=data[:look_back],warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method, alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == "RNN":
        predict_model = RNN_test_CP(train_epoch=100, lr=1e-4, input_length=12,init_data=data[:look_back],warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = Conformal, CP_method = CP_method, alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'Bagging-GPR':
        predict_model = Bagging_CP(base_model="GPR", n_estimators=3,seasonal=False, seasonal_period=24, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = True, CP_method = "ACP",alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'Bagging-DTR':
        predict_model = Bagging_CP(base_model="DTR", n_estimators=30,seasonal=False, seasonal_period=24, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = True, CP_method = "ACP",alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'Adaboost-GPR':
        predict_model = Adaboost_CP(base_model="GPR", n_estimators=3,seasonal=False, seasonal_period=24, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = True, CP_method = "ACP",alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'Adaboost-DTR':
        predict_model = Adaboost_CP(base_model="DTR", n_estimators=30,seasonal=False, seasonal_period=24, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = True, CP_method = "ACP",alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'RFR':
        predict_model = RFR_CP(n_estimators=100,seasonal=False, seasonal_period=24, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = True, CP_method = "ACP",alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'ETR':
        predict_model = ETR_CP(n_estimators=100,seasonal=False, seasonal_period=24, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = True, CP_method = "ACP",alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
    elif predict_method == 'GBR':
        predict_model = GBR_CP(seasonal=False, seasonal_period=24, step=24, warm_up_data=data[:warm_up_length], look_back = look_back, Conformal = True, CP_method = "ACP",alpha = alpha, gamma = gamma, lower_threshold=lower_threshold)
 
    model_name = predict_method+"-"+CP_method
    total_length = len(data)
    if "Ensemble" in model_name:
        model_name += "-"+ensemble_method
        
    for i in range(warm_up_length, len(data)):
        predict_model.fit(data[i-look_back:i])
        y_pred, y_upper_i, y_lower_i = predict_model.predict(1)
        predict_model.update(y_pred, y_upper_i, y_lower_i, data[i])
        
    predict_models[model_name] = predict_model.predictions
    CP_models[model_name] = predict_model.CP_module
    ed_time = time.perf_counter()
    print(f"{model_name}的训练和实验共花费：{ed_time-st_time:.2f}s。")