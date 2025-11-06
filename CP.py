import numpy as np
from tqdm import trange
import statsmodels.api as sm
import statsmodels
from scipy.stats import norm
import sklearn
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF, DotProduct, Matern, RationalQuadratic


class ConformalPrediction:
    def __init__(self, data, predict_model, method="ACP", alpha=0.1, gamma=0.05, look_back=30, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.model = predict_model
        self.alpha_t = self.alpha
        self.look_back = look_back
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.seasonal_period = seasonal_period
        self.upper_bound = np.empty(0)
        self.lower_bound = np.empty(0)
        self.width = np.empty(0)
        self.pred = np.empty(0)
        self.res_cal = np.empty(0)
        self.coverage = 0
        self.quantileloss1 = 0
        self.quantileloss2 = 0
        self.y_sum = 0
        
        # print(f"""Start to warm-up the conformal prediction moudule with method={self.method}, alpha={self.alpha}, gamma={self.gamma}""")
        
        for i in range(data.shape[0]-self.look_back):
            if self.method == "NonCP-ARIMA":
                training_mod = statsmodels.tsa.statespace.sarimax.SARIMAX(data[i:self.look_back+i], order=(1, 1, 1),initialization='approximate_diffuse')
                training_res = training_mod.fit(disp=0)
                y_prediction = training_res.get_forecast(steps=1)
                pred_int = y_prediction.conf_int(alpha=self.alpha)
                y_pred_arima = [training_res.forecast(steps=1)]
                self.upper_bound = np.append(self.upper_bound, max(min(pred_int[0,1], self.upper_threshold), self.lower_threshold))
                self.lower_bound = np.append(self.lower_bound, min(max(pred_int[0,0], self.lower_threshold), self.upper_threshold))
                self.width = np.append(self.width, (min(pred_int[0,1], self.upper_threshold) - max(pred_int[0,0], self.lower_threshold)) / (2 * np.abs(y_pred_arima[0])+ 1e-10))
                self.pred = np.append(self.pred, min(max(y_pred_arima[0], self.lower_threshold),self.upper_threshold))
                cover = float((self.lower_bound[-1] <= data[self.look_back+i]) & (data[self.look_back+i] <= self.upper_bound[-1]))
                self.res_cal = np.append(self.res_cal, y_pred_arima[0] - data[self.look_back+i])
                self.coverage += cover
                if(data[self.look_back+i] <= self.upper_bound[-1]):
                    self.quantileloss1 += self.alpha * 2 * float(np.abs((y_pred_arima[0] - data[self.look_back+i]))) 
                    self.quantileloss2 += self.alpha * 2 * float((y_pred_arima[0] - data[self.look_back+i]))
                else : 
                    self.quantileloss1 += (1 - self.alpha) * 2 * float(np.abs((y_pred_arima[0] - data[self.look_back+i])))
                    self.quantileloss2 += self.alpha * 2 * float((y_pred_arima[0] - data[self.look_back+i]))
                self.y_sum += data[self.look_back+i]
                continue
            elif self.method == "NonCP-GPR":
                training_mod = sklearn.gaussian_process.GaussianProcessRegressor(kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)), alpha=self.alpha, n_restarts_optimizer=1, normalize_y=True)
                # sample_x = np.array([i % self.seasonal_period for i in range(self.look_back)]).reshape(-1,1)
                # sample_y = data[i:self.look_back+i]
                # last_sample = [[self.look_back % self.seasonal_period]]
                self.step = 5
                sample_x = np.array([data[j - self.step: j] for j in range(self.step + i,self.look_back + i)]).reshape(-1,self.step)
                sample_y = data[i + self.step: i + self.look_back]
                last_sample = np.array(data[self.look_back + i - self.step:self.look_back + i]).reshape(1,self.step)
                
                training_mod.fit(sample_x, sample_y)
                
                y_pred, y_std = training_mod.predict(last_sample, return_std=True)
                y_pred=y_pred.reshape(-1,1)
                # print(y_pred,y_std)                                                                      
                self.upper_bound = np.append(self.upper_bound, max(min(y_pred[0,0]+norm.ppf(1-self.alpha/2)*y_std[0], self.upper_threshold),self.lower_threshold))
                self.lower_bound = np.append(self.lower_bound, min(max(y_pred[0,0]-norm.ppf(1-self.alpha/2)*y_std[0], self.lower_threshold),self.upper_threshold))
                self.width = np.append(self.width, (self.upper_bound[-1] - self.lower_bound[-1]) / (2 * np.abs(y_pred[0,0])+ 1e-10))
                self.pred = np.append(self.pred, min(max(y_pred[0,0], self.lower_threshold),self.upper_threshold))
                cover = float((self.lower_bound[-1] <= data[self.look_back+i]) & (data[self.look_back+i] <= self.upper_bound[-1]))
                self.res_cal = np.append(self.res_cal, y_pred[0,0] - data[self.look_back+i])
                self.coverage += cover
                if(data[self.look_back+i] <= self.upper_bound[-1]):
                    self.quantileloss1 += self.alpha * 2 * float(np.abs(y_pred[0,0] - data[self.look_back+i])) 
                    self.quantileloss2 += self.alpha * 2 * float((y_pred[0,0] - data[self.look_back+i]))
                else : 
                    self.quantileloss1 += (1 - self.alpha) * 2 * float(np.abs( y_pred[0,0] - data[self.look_back+i]))
                    self.quantileloss2 += self.alpha * 2 * float((y_pred[0,0] - data[self.look_back+i]))
                self.y_sum += data[self.look_back+i]
                continue    

            self.model.fit(data[i:self.look_back+i])
            y_pred = self.model.predict(1)
            if(isinstance(self.model, EnsembleModel)):
                self.model.update(data[self.look_back+i])



            if i >= self.look_back:    
                if self.method == "Gaussian":
                    window = norm.ppf(1 - self.alpha/2) * np.std(self.res_cal)
                    y_lower_i, y_upper_i = y_pred[0] - window, y_pred[0] + window

                elif self.method == "CP":
                    res_cal_cp = np.abs(self.res_cal)
                    window = np.quantile(res_cal_cp,(1-self.alpha))
                    y_lower_i, y_upper_i = y_pred[0]-window, y_pred[0]+window
                elif self.method == "ACP":
                    res_cal_acp = np.abs(self.res_cal)
                    if(self.alpha_t >= 1):
                        self.alpha_t = 1
                    elif(self.alpha_t <= 0):
                        self.alpha_t = 0 
                    window = np.quantile(res_cal_acp,(1-self.alpha_t))
                    y_lower_i, y_upper_i = y_pred[0]-window, y_pred[0]+window
                    err = 1-float((y_lower_i <= data[self.look_back+i]) & (data[self.look_back+i] <= y_upper_i))
  
                    self.alpha_t = self.alpha_t + self.gamma*(self.alpha-err)

                self.upper_bound = np.append(self.upper_bound, max(min(y_pred[0] + window, self.upper_threshold), self.lower_threshold))
                self.lower_bound = np.append(self.lower_bound, min(max(y_pred[0] - window, self.lower_threshold), self.upper_threshold))
                cover = float((self.lower_bound[-1] <= data[self.look_back+i]) & (data[self.look_back+i] <= self.upper_bound[-1]))
                self.coverage += cover
                self.width = np.append(self.width, (self.upper_bound[-1] - self.lower_bound[-1]) / (2 * np.abs(y_pred[0])+ 1e-10))
            
            # print(window)
            if(isinstance(self.model, EnsembleModel)):
                self.model.update(data[self.look_back+i])
            self.pred = np.append(self.pred, min(max(y_pred[0], self.lower_threshold),self.upper_threshold))
            self.res_cal = np.append(self.res_cal, y_pred[0] - data[self.look_back+i])
            
        # print(f"Finish the warming up.")
        
    def online_predict(self, new_sample):
        if self.method == "NonCP-ARIMA":
            training_mod = statsmodels.tsa.statespace.sarimax.SARIMAX(new_sample, order=(1, 1, 1),initialization='approximate_diffuse')
            training_res = training_mod.fit(disp=0)
            y_prediction = training_res.get_forecast(steps=1)
            pred_int = y_prediction.conf_int(alpha=self.alpha)
            y_pred = [training_res.forecast(steps=1)]
            y_upper_i = max(min(pred_int[0,1],self.upper_threshold), self.lower_threshold)
            y_lower_i = min(max(pred_int[0,0],self.lower_threshold), self.upper_threshold)
        elif self.method == "NonCP-GPR":
            training_mod = sklearn.gaussian_process.GaussianProcessRegressor(kernel=WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))+ConstantKernel(constant_value=2) * ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45)), alpha=self.alpha, n_restarts_optimizer=1, normalize_y=True)
            sample_x = np.array([i % self.seasonal_period for i in range(len(new_sample))]).reshape(-1,1)
            sample_y = new_sample
            last_sample = [[len(new_sample) % self.seasonal_period]]
            training_mod.fit(sample_x, sample_y)
            y_pred, y_std = training_mod.predict(last_sample, return_std=True)
            y_pred=y_pred.reshape(-1,1)
            y_pred = y_pred[0]
            y_upper_i = max(min(y_pred[0]+norm.ppf(1-self.alpha/2)*y_std[0],self.upper_threshold), self.lower_threshold)
            y_lower_i = min(max(y_pred[0]-norm.ppf(1-self.alpha/2)*y_std[0],self.lower_threshold), self.upper_threshold)
            
        else:
            self.model.fit(new_sample)
            y_pred = self.model.predict(1)

            if self.method == "Gaussian":
                window = norm.ppf(1 - self.alpha/2) * np.std(self.res_cal)
            elif self.method == "CP":
                res_cal_cp = np.abs(self.res_cal)
                # window = np.quantile(res_cal_cp,(1-alpha))
                window = np.quantile(res_cal_cp,(1-self.alpha))
            elif self.method == "ACP":
                res_cal_acp = np.abs(self.res_cal)
                if(self.alpha_t >= 1):
                    self.alpha_t = 1
                elif(self.alpha_t <= 0):
                    self.alpha_t = 0 
                window = np.quantile(res_cal_acp,(1-self.alpha_t))
            y_lower_i, y_upper_i = min(max(y_pred[0]-window,self.lower_threshold), self.upper_threshold), max(min(y_pred[0]+window,self.upper_threshold),self.lower_threshold)
        y_pred = [max(min(y_pred[0], self.upper_threshold), self.lower_threshold)]

        return y_pred, y_upper_i, y_lower_i
    
    def online_update(self, y_pred, y_upper_i, y_lower_i, y_truth):
        # if(isinstance(self.model, EnsembleModel)):
        #     self.model.update(y_truth)
        if self.method == "ACP":
            err = 1-float((y_lower_i <= y_truth) & (y_truth <= y_upper_i))
            cover = float((y_lower_i <= y_truth) & (y_truth <= y_upper_i))
            self.coverage += cover
            self.alpha_t = self.alpha_t + self.gamma*(self.alpha-err)
            # print(self.alpha_t)
        else:
            cover = float((y_lower_i <= y_truth) & (y_truth <= y_upper_i))
            self.coverage += cover
            if(y_truth <= y_upper_i):
                self.quantileloss1 += self.alpha * 2 * float(np.abs(y_pred - y_truth))
                self.quantileloss2 += self.alpha * 2 * float((y_pred - y_truth))
            else : 
                self.quantileloss1 += (1 - self.alpha) * 2 * float(np.abs( y_pred - y_truth))
                self.quantileloss2 += self.alpha * 2 * float((y_pred - y_truth))
            self.y_sum += y_truth
        self.upper_bound = np.append(self.upper_bound, min(y_upper_i, self.upper_threshold))
        self.lower_bound = np.append(self.lower_bound, max(y_lower_i, self.lower_threshold))
        self.width = np.append(self.width, (min(y_upper_i, self.upper_threshold) - max(y_lower_i, self.lower_threshold)) / (2 * np.abs(y_pred[0]) + 1e-10))
        self.pred = np.append(self.pred,y_pred[0])
        self.res_cal = np.append(self.res_cal, y_pred[0] - y_truth)
        
    def online_train(self, new_sample, y_truth):
        y_pred,y_upper_i, y_lower_i = self.online_predict(new_sample)
        self.online_update(y_pred, y_upper_i, y_lower_i, y_truth)
        
        
class CPmodule:
    def __init__(self, data, predict_results, alpha, method="ACP", gamma=0.05, look_back=30, upper_threshold=np.inf, lower_threshold=0,seasonal_period=np.inf):
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = self.alpha
        self.look_back = look_back
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.seasonal_period = seasonal_period
        self.upper_bound = np.empty(0)
        self.lower_bound = np.empty(0)
        self.width = np.empty(0)
        self.res_cal = np.empty(0)
        self.coverage = 0
        self.quantileloss1 = 0
        self.quantileloss2 = 0
        self.y_sum = 0

        
        # print(f"""Start to warm-up the conformal prediction moudule with method={self.method}, alpha={self.alpha}, gamma={self.gamma}""")
        
        for i in range(data.shape[0]-self.look_back):

            y_pred = [predict_results[i]]

            if i >= self.look_back:    
                window = 0
                if self.method == "Gaussian":
                    window = norm.ppf(1 - self.alpha/2) * np.std(self.res_cal)
                    y_lower_i, y_upper_i = y_pred[0] - window, y_pred[0] + window

                elif self.method == "CP":
                    res_cal_cp = np.abs(self.res_cal)
                    window = np.quantile(res_cal_cp,(1-self.alpha))
                    y_lower_i, y_upper_i = y_pred[0]-window, y_pred[0]+window
                elif self.method == "ACP":
                    res_cal_acp = np.abs(self.res_cal)
                    if(self.alpha_t >= 1):
                        self.alpha_t = 1
                    elif(self.alpha_t <= 0):
                        self.alpha_t = 0 
                    window = np.quantile(res_cal_acp,(1-self.alpha_t))
                    y_lower_i, y_upper_i = y_pred[0]-window, y_pred[0]+window
                    err = 1-float((y_lower_i <= data[self.look_back+i]) & (data[self.look_back+i] <= y_upper_i))
  
                    self.alpha_t = self.alpha_t + self.gamma*(self.alpha-err)
                # print(window)
                self.upper_bound = np.append(self.upper_bound, max(min(y_pred[0] + window, self.upper_threshold), self.lower_threshold))
                self.lower_bound = np.append(self.lower_bound, min(max(y_pred[0] - window, self.lower_threshold), self.upper_threshold))
                cover = float((self.lower_bound[-1] <= data[self.look_back+i]) & (data[self.look_back+i] <= self.upper_bound[-1]))
                self.coverage += cover
                if(data[self.look_back+i] <= self.upper_bound[-1]):
                    self.quantileloss1 += self.alpha * 2 * float(np.abs((y_pred[0] - data[self.look_back+i])))
                    self.quantileloss2 += self.alpha * 2 * float((y_pred[0] - data[self.look_back+i]))
                else : 
                    self.quantileloss1 += (1 - self.alpha) * 2 * float(np.abs((y_pred[0] - data[self.look_back+i])))
                    self.quantileloss2 += self.alpha * 2 * float((y_pred[0] - data[self.look_back+i]))
                    
                self.y_sum += data[self.look_back+i]

                    
                self.width = np.append(self.width, (self.upper_bound[-1] - self.lower_bound[-1]) / (2 * np.abs(y_pred[0])+ 1e-10))
            
            # print(window)

            self.res_cal = np.append(self.res_cal, y_pred[0] - data[self.look_back+i])
            
        # print(f"Finish the warming up.")
        
    def online_predict(self, y_pred):

        window = 0
        if self.method == "Gaussian":
            window = norm.ppf(1 - self.alpha/2) * np.std(self.res_cal)
        elif self.method == "CP":
            res_cal_cp = np.abs(self.res_cal)
            # window = np.quantile(res_cal_cp,(1-alpha))
            window = np.quantile(res_cal_cp,(1-self.alpha))
        elif self.method == "ACP":
            res_cal_acp = np.abs(self.res_cal)
            if(self.alpha_t >= 1):
                self.alpha_t = 1
            elif(self.alpha_t <= 0):
                self.alpha_t = 0 
            window = np.quantile(res_cal_acp,(1-self.alpha_t))
        y_lower_i, y_upper_i = min(max(y_pred[0]-window,self.lower_threshold), self.upper_threshold), max(min(y_pred[0]+window,self.upper_threshold),self.lower_threshold)
        y_pred = [max(min(y_pred[0], self.upper_threshold), self.lower_threshold)]

        return y_pred, y_upper_i, y_lower_i
    
    def online_update(self, y_pred, y_upper_i, y_lower_i, y_truth):

        if self.method == "ACP":
            err = 1-float((y_lower_i <= y_truth) & (y_truth <= y_upper_i))
            cover = float((y_lower_i <= y_truth) & (y_truth <= y_upper_i))
            self.coverage += cover
            self.alpha_t = self.alpha_t + self.gamma*(self.alpha-err)
            # print(self.alpha_t)
        else:
            cover = float((y_lower_i <= y_truth) & (y_truth <= y_upper_i))
            self.coverage += cover
        self.upper_bound = np.append(self.upper_bound, min(y_upper_i, self.upper_threshold))
        self.lower_bound = np.append(self.lower_bound, max(y_lower_i, self.lower_threshold))
        self.width = np.append(self.width, (min(y_upper_i, self.upper_threshold) - max(y_lower_i, self.lower_threshold)) / (2 * np.abs(y_pred[0]) + 1e-10))
        self.res_cal = np.append(self.res_cal, y_pred[0] - y_truth)
        
    def online_train(self, y_pred, y_truth):
        y_pred,y_upper_i, y_lower_i = self.online_predict(y_pred)
        self.online_update(y_pred, y_upper_i, y_lower_i, y_truth)