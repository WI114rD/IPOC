import pandas as pd
import time
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from pmdarima import auto_arima
import numpy as np
from tqdm import trange
import seaborn as sns
import scipy.io as sio
from scipy import fft, arange, signal
import warnings
from CP import ConformalPrediction
from ForecastModels import ARIMA, SARIMA, EnsembleModule, Forecast_test_ver2
import multiprocessing
from data_preprocess import data_producer


warnings.filterwarnings("ignore")
dataname_list = ['Abase','FaaS','Alibaba']#F #FaaS
for dataname in dataname_list:
    print(dataname)
    df_fromdata, random_series = data_producer(dataset=dataname, random_serie_number=10)
    print(random_series)
    localtime = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    filename = f"Forecast-{dataname}-{localtime}"
    if os.path.exists(f'../result/{filename}.csv'):
        df_res = pd.read_csv(f'../result/{filename}.csv')
        print("读取之前保存的实验结果")
    else:
        df_res = pd.DataFrame(columns=['cluster_label','predict_method', 'ensemble_method', 'CP_method', 'alpha','gamma','quantileloss','rmse','rmse_trans','rmse_rank', 'mae','mae_trans','mae_rank','smape','smape_trams','smape_rank','coverage','median width','average width','max width','min width'])
        print("未发现保存结果，创建新文件")
    df_res = pd.DataFrame(columns=['cluster_label','predict_method', 'ensemble_method', 'CP_method', 'alpha','gamma','quantileloss1', 'quantileloss2','lookback','choice_interval','training_time','rmse','rmse_trans','rmse_rank', 'mae','mae_trans','mae_rank','smape','smape_trams','smape_rank','coverage','median width','average width','max width','min width'])
    
    predict_methods = ["ARIMA011","ARIMA111", "ARIMA101", "ARIMA110", "ARIMA010", "ARIMA201", "ARIMA100", "Naive","NaiveDrift","GPR","SVR","SVR_external","ETS","ARIMA301","Prophet","LSTM","GRU","RNN",
             "DTR",  "Bagging-GPR", "Bagging-DTR","Adaboost-GPR","Adaboost-DTR","RFR","ETR",
                       "Ensemble","Ensemble","Ensemble","Ensemble"]
    CP_methods = ["ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP", "ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP","ACP"]
    ensemble_methods = ["greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","greedy","FTPL","greedy","FFORMS","FFORMA"]
    model_number = len(predict_methods)
    model_names = []
    for i in range(len(CP_methods)):
        model_name = predict_methods[i]+"-"+CP_methods[i]
        if "Ensemble" in model_name:
            model_name += "-"+ensemble_methods[i]
        model_names.append(model_name)
    print(model_names)
    CP_processes = []
    CP_models=multiprocessing.Manager().dict()
    predict_models=multiprocessing.Manager().dict()
    RMSE_rank, MAE_rank, SMAPE_rank, Quan1_rank, Quan2_rank = [[] for i in range(model_number)],[[] for i in range(model_number)],[[] for i in range(model_number)],[[] for i in range(model_number)],[[] for i in range(model_number)]
    RMSE_ana, MAE_ana, SMAPE_ana, Quan1_ana, Quan2_ana = [[] for i in range(model_number)],[[] for i in range(model_number)],[[] for i in range(model_number)], [[] for i in range(model_number)],[[] for i in range(model_number)]
    #look_back_list=[30,45,60,80,120]
    look_back_list=[]
    if dataname=='Abase':
        look_back_list=[50]
    elif dataname=='FaaS':
        look_back_list=[40]
    elif dataname=='Alibaba':
        look_back_list=[30]
    alpha_list=[0.05,0.10,0.15,0.2,0.25]
    start_id = 0
    look_back=30
    choice_interval=30
    gamma = 0.05
    warm_up_length = 200
    start_time = time.perf_counter()
    repeat_times = 1
    ensemble_repeat_times = 5
    for alpha in alpha_list:
        for cluster_id in range(start_id, len(random_series)):

            random_cluster = random_series[cluster_id]
            df_select = df_fromdata[df_fromdata["id"] == random_cluster]
            random_series_length = 1200
            if(dataname == "Alibaba"):
                index = pd.date_range("2018-01-01 00:00:00",periods=df_select.shape[0], freq="S")
                df_select["datetime"] = index
            print("The length of series is:"+str(df_select.shape[0]))
            if(df_select.shape[0] > 1200):
                random_start = random.randint(0, df_select.shape[0] - random_series_length - 1)
                df_select.sort_values("datetime",inplace=True)
                df_select = df_select[random_start : random_start + random_series_length]

            print(df_select)

            print(f"当前处理第{cluster_id+1}个集群，id为：{random_cluster}")
            start_time_epoch = time.perf_counter()

            # ts = df_1.fillna(0)["total_usage"]
            ts = df_select["y"]
            origin_data = ts.values.reshape(-1, 1).astype("float32")
            scaler = MinMaxScaler()
            scaler.fit(origin_data)
            data = scaler.transform(origin_data)
            # data = origin_data
            lower_threshold = scaler.transform(np.array(0).reshape(1,1))
            start_time_data = df_select["datetime"].values[0]
            time_gs = {"Abase":"H", "HDFS":"D", "FaaS":"T", "Alibaba":"S"}
            time_granularity = time_gs[dataname]


            total_length = data.shape[0]
            Quan1, Quan2, RMSE,RMSE_trans,MAE,MAE_trans,SMAPE, SMAPE_trans,Coverage,Median_width,Average_width,Max_width,Min_width = {},{},{},{},{},{},{},{},{},{},{},{},{}
            ensemble_test = {}
            for model_name in model_names:
                Quan1[model_name],Quan2[model_name], RMSE[model_name], MAE[model_name], SMAPE[model_name], RMSE_trans[model_name], MAE_trans[model_name], SMAPE_trans[model_name], Coverage[model_name], Median_width[model_name], Average_width[model_name], Max_width[model_name], Min_width[model_name] = np.inf, np.inf, np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0,np.inf,np.inf,np.inf,np.inf


            forecast_horizon = 1
            print(f"当前模型输入长度为{look_back},输出长度为{forecast_horizon}。")


            print(f"The median value for {random_cluster} is {np.median(df_select['y'])}.")
            for __t in range(repeat_times):

                for model_id in range(model_number):
                    if "Ensemble" in model_names[model_id]:
                        continue
                    # predict_method, look_back, warm_up_length, data, lower_threshold, alpha=0.1, gamma=0.05, method = "ACP"

                    CP_processes.append(multiprocessing.Process(target=Forecast_test_ver2, kwargs={"CP_models":CP_models,"predict_models":predict_models, "predict_method": predict_methods[model_id], "look_back":look_back,
                                                                               "warm_up_length":warm_up_length, "data":data, "lower_threshold":lower_threshold,"Conformal":True, "time_granularity":time_granularity,
                                                                               "alpha":alpha, "gamma":gamma, "CP_method":CP_methods[model_id], "ensemble_method":ensemble_methods[model_id],
                    "start_time":start_time_data}))
                    CP_processes[-1].start()
                    # thread=myThread(model_id, "Thread-"+str(model_id), predict_methods[model_id],look_back, warm_up_length, data, lower_threshold, alpha=alpha, gamma=gamma, method = methods[model_id])
                    # thread.start()
                    # Threads.append(thread) 
                for process in CP_processes:
                    process.join()

                for model_id in range(model_number):
                    model_name = model_names[model_id]
                    if predict_methods[model_id] == "Ensemble":
                        for __et in range(ensemble_repeat_times):
                            if (not ensemble_methods[model_id] == "FTPL") and __et > 0:
                                continue

                            ensemble_test[model_name] = EnsembleModule(predict_models,model_names[:-4],alpha = alpha,warm_up_data = data, Conformal=True, ensemble_method=ensemble_methods[model_id],look_back=look_back,choice_interval=choice_interval)


                            RMSE[model_name] = min(np.sqrt(np.sum(np.power(scaler.inverse_transform(np.array(ensemble_test[model_name].predictions).reshape(-1,1)) - origin_data[look_back:],2))/ensemble_test[model_name].CP_module.res_cal.shape[0]),RMSE[model_name])
                            MAE[model_name] = min(np.sum(np.abs(scaler.inverse_transform(np.array(ensemble_test[model_name].predictions).reshape(-1,1)) - origin_data[look_back:]))/ensemble_test[model_name].CP_module.res_cal.shape[0],MAE[model_name])
                            # SMAPE[model_name] = min(np.sum(np.abs((scaler.inverse_transform(np.array(ensemble_test[model_name].predictions).reshape(-1,1)) - origin_data[look_back:]).reshape(-1,))/((np.array(origin_data[look_back:]).reshape(-1,)+scaler.inverse_transform(np.array(ensemble_test[model_name].predictions).reshape(-1,1)).reshape(-1,))/2.))/ensemble_test[model_name].CP_module.res_cal.shape[0],SMAPE[model_name])
                            RMSE_trans[model_name] = min(np.sqrt(np.sum(np.power(ensemble_test[model_name].CP_module.res_cal,2))/ensemble_test[model_name].CP_module.res_cal.shape[0]),RMSE_trans[model_name])
                            MAE_trans[model_name] = min(np.sum(np.abs(ensemble_test[model_name].CP_module.res_cal))/ensemble_test[model_name].CP_module.res_cal.shape[0],MAE_trans[model_name])
                            # SMAPE_trans[model_name] = min(np.sum(np.abs(ensemble_test[model_name].CP_module.res_cal.reshape(-1,))/((np.array(data[look_back:]).reshape(-1,)+ensemble_test[model_name].predictions.reshape(-1,))/2.))/ensemble_test[model_name].CP_module.res_cal.shape[0], SMAPE_trans[model_name])
                            Coverage[model_name] = max(ensemble_test[model_name].CP_module.coverage/ensemble_test[model_name].CP_module.width.shape[0],Coverage[model_name])
                            Median_width[model_name] = min(np.median(ensemble_test[model_name].CP_module.width),Median_width[model_name])
                            Max_width[model_name] = min(np.max(ensemble_test[model_name].CP_module.width),Max_width[model_name])
                            Min_width[model_name] = min(np.min(ensemble_test[model_name].CP_module.width),Min_width[model_name])
                            Quan1[model_name]= min((ensemble_test[model_name].CP_module.quantileloss1/ensemble_test[model_name].CP_module.y_sum)[0],Quan1[model_name])
                            Quan2[model_name]= min((ensemble_test[model_name].CP_module.quantileloss2/ensemble_test[model_name].CP_module.y_sum)[0],Quan2[model_name])
                            
                    else:
                        RMSE[model_name] = min(np.sqrt(np.sum(np.power(scaler.inverse_transform(np.array(predict_models[model_name]).reshape(-1,1)) - origin_data[look_back:],2))/CP_models[model_name].res_cal.shape[0]),RMSE[model_name])
                        MAE[model_name] = min(np.sum(np.abs(scaler.inverse_transform(np.array(predict_models[model_name]).reshape(-1,1)) - origin_data[look_back:]))/CP_models[model_name].res_cal.shape[0],MAE[model_name])
                        # SMAPE[model_name] = min(np.sum(np.abs((scaler.inverse_transform(np.array(predict_models[model_name]).reshape(-1,1)) - origin_data[look_back:]).reshape(-1,))/((np.array(origin_data[look_back:]).reshape(-1,)+scaler.inverse_transform(np.array(predict_models[model_name]).reshape(-1,1)).reshape(-1,))/2.))/CP_models[model_name].res_cal.shape[0],SMAPE[model_name])
                        RMSE_trans[model_name] = min(np.sqrt(np.sum(np.power(CP_models[model_name].res_cal,2))/CP_models[model_name].res_cal.shape[0]),RMSE_trans[model_name])
                        MAE_trans[model_name] = min(np.sum(np.abs(CP_models[model_name].res_cal))/CP_models[model_name].res_cal.shape[0],MAE_trans[model_name])
                        # SMAPE_trans[model_name] = min(np.sum(np.abs(CP_models[model_name].res_cal.reshape(-1,))/((np.array(data[look_back:]).reshape(-1,)+predict_models[model_name].reshape(-1,))/2.))/CP_models[model_name].res_cal.shape[0],SMAPE_trans[model_name])
                        Coverage[model_name] = max(CP_models[model_name].coverage/CP_models[model_name].width.shape[0],Coverage[model_name])
                        Median_width[model_name] = min(np.median(CP_models[model_name].width),Median_width[model_name])
                        Average_width[model_name] = min(np.mean(CP_models[model_name].width),Average_width[model_name])
                        Max_width[model_name] = min(np.max(CP_models[model_name].width),Max_width[model_name])
                        Min_width[model_name] = min(np.min(CP_models[model_name].width),Min_width[model_name])
                        Quan1[model_name]= min(CP_models[model_name].quantileloss1/CP_models[model_name].y_sum,Quan1[model_name])[0]
                        Quan2[model_name]= min(CP_models[model_name].quantileloss2/CP_models[model_name].y_sum,Quan2[model_name])[0]
                        


            RMSE_epoch, MAE_epoch, Quan1_epoch, Quan2_epoch = [RMSE[model_names[model_id]] for model_id in range(model_number)],[MAE[model_names[model_id]] for model_id in range(model_number)],[Quan1[model_names[model_id]] for model_id in range(model_number)],[Quan2[model_names[model_id]] for model_id in range(model_number)]
            # SMAPE_epoch = [SMAPE[model_names[model_id]] for model_id in range(model_number)]
            # print(Quan_epoch, type(Quan_epoch))
            RMSE_sorted = sorted(set(RMSE_epoch))
            MAE_sorted = sorted(set(MAE_epoch))
            Quan1_sorted = sorted(set(Quan1_epoch))
            Quan2_sorted = sorted(set(Quan2_epoch))
            # SMAPE_sorted = sorted(set(SMAPE_epoch))

            RMSE_dic,MAE_dic,Quan1_dic, Quan2_dic = {x:i+1 for i, x in enumerate(RMSE_sorted)},{x:i+1 for i, x in enumerate(MAE_sorted)},{x:i+1 for i, x in enumerate(Quan1_sorted)},{x:i+1 for i, x in enumerate(Quan2_sorted)}
            # SMAPE_dic = {x:i+1 for i, x in enumerate(SMAPE_sorted)}

            RMSE_best_epoch, MAE_best_epch, SMAPE_best_epch, Quan1_best_epch, Quan2_best_epch = -1, -1, -1, -1, -1

            for model_id in range(model_number):
                if RMSE_dic[RMSE_epoch[model_id]] == 1:
                    RMSE_best_epoch = model_id
                if MAE_dic[MAE_epoch[model_id]] == 1:
                    MAE_best_epoch = model_id
                if Quan1_dic[Quan1_epoch[model_id]] == 1:
                    Quan1_best_epoch = model_id
                if Quan2_dic[Quan2_epoch[model_id]] == 1:
                    Quan2_best_epoch = model_id
                # if SMAPE_dic[SMAPE_epoch[model_id]] == 1:
                    # SMAPE_best_epoch = model_id
                RMSE_rank[model_id].append(RMSE_dic[RMSE_epoch[model_id]])
                MAE_rank[model_id].append(MAE_dic[MAE_epoch[model_id]])
                Quan1_rank[model_id].append(Quan1_dic[Quan1_epoch[model_id]])
                Quan2_rank[model_id].append(Quan2_dic[Quan2_epoch[model_id]])
                # SMAPE_rank[model_id].append(SMAPE_dic[SMAPE_epoch[model_id]])

                RMSE_ana[model_id].append(RMSE_trans[model_names[model_id]])
                MAE_ana[model_id].append(MAE_trans[model_names[model_id]])
                Quan1_ana[model_id].append(Quan1[model_names[model_id]])
                Quan2_ana[model_id].append(Quan2[model_names[model_id]])
                # SMAPE_ana[model_id].append(SMAPE_trans[model_names[model_id]])

            RMSE_best_model, MAE_best_model, SMAPE_best_model, Quan1_best_model,Quan2_best_model = -1, -1, -1,-1, -1
            RMSE_best_mean, MAE_best_mean, SMAPE_best_mean, Quan1_best_mean, Quan2_best_mean = np.inf,np.inf,np.inf,np.inf,np.inf

            for model_id in range(model_number):
                model_name = model_names[model_id]
                RMSE_mean = np.mean(RMSE_rank[model_id])
                if RMSE_mean < RMSE_best_mean:
                    RMSE_best_model = model_id
                    RMSE_best_mean = RMSE_mean
                print(f"The RMSE of {model_name} for {random_cluster} is {RMSE[model_names[model_id]]:.4f}. ")
                print(f"本轮进行过Scaler的RMSE为：{RMSE_trans[model_name]}，在{cluster_id-start_id+1}轮训练中，RMSE平均数为{np.mean(RMSE_ana[model_id]):.4f}，中位数为{np.median(RMSE_ana[model_id]):.4f}，最大为{np.max(RMSE_ana[model_id]):.4f}，最小为{np.min(RMSE_ana[model_id]):.4f}。")
                print(f"本轮中排名{RMSE_rank[model_id][-1]}/{model_number},在{cluster_id-start_id+1}轮训练中,排名平均数为{RMSE_mean:0.4f}，排名中位数为{np.median(RMSE_rank[model_id]):0.4f}，最大排名为{np.max(RMSE_rank[model_id]):0.4f}，最小排名为{np.min(RMSE_rank[model_id]):0.4f}。")
                MAE_mean = np.mean(MAE_rank[model_id])
                if MAE_mean < MAE_best_mean:
                    MAE_best_model = model_id
                    MAE_best_mean = MAE_mean
                print(f"The MAE of {model_name} for {random_cluster} is {MAE[model_names[model_id]]:.4f}. ")
                print(f"本轮进行过Scaler的MAE为：{MAE_trans[model_name]}，在{cluster_id-start_id+1}轮训练中，MAE平均数为{np.mean(MAE_ana[model_id]):.4f}，中位数为{np.median(MAE_ana[model_id]):.4f}，最大为{np.max(MAE_ana[model_id]):.4f}，最小为{np.min(MAE_ana[model_id]):.4f}。")        
                print(f"本轮中排名{MAE_rank[model_id][-1]}/{model_number},在{cluster_id-start_id+1}轮训练中排名平均数为{MAE_mean:0.4f}，排名中位数为{np.median(MAE_rank[model_id]):0.4f}，最大排名为{np.max(MAE_rank[model_id]):0.4f}，最小排名为{np.min(MAE_rank[model_id]):0.4f}。")
                
                Quan1_mean = np.mean(Quan1_rank[model_id])
                if Quan1_mean < Quan1_best_mean:
                    Quan1_best_model = model_id
                    Quan1_best_mean = MAE_mean
                print(f"The Quantile Loss ver.1 of {model_name} for {random_cluster} is {Quan1[model_names[model_id]]:.4f}. ")
                print(f"本轮的Quantile Loss ver.1 为：{Quan1[model_name]}，在{cluster_id-start_id+1}轮训练中，Quantile Loss平均数为{np.mean(Quan1_ana[model_id]):.4f}，中位数为{np.median(Quan1_ana[model_id]):.4f}，最大为{np.max(Quan1_ana[model_id]):.4f}，最小为{np.min(Quan1_ana[model_id]):.4f}。")        
                print(f"本轮中排名{Quan1_rank[model_id][-1]}/{model_number},在{cluster_id-start_id+1}轮训练中排名平均数为{Quan1_mean:0.4f}，排名中位数为{np.median(Quan1_rank[model_id]):0.4f}，最大排名为{np.max(Quan1_rank[model_id]):0.4f}，最小排名为{np.min(Quan1_rank[model_id]):0.4f}。")
                
                
                Quan2_mean = np.mean(Quan2_rank[model_id])
                if Quan2_mean < Quan2_best_mean:
                    Quan2_best_model = model_id
                    Quan2_best_mean = MAE_mean
                print(f"The Quantile Loss ver.2 of {model_name} for {random_cluster} is {Quan2[model_names[model_id]]:.4f}. ")
                print(f"本轮的Quantile Loss ver.2 为：{Quan2[model_name]}，在{cluster_id-start_id+1}轮训练中，Quantile Loss平均数为{np.mean(Quan2_ana[model_id]):.4f}，中位数为{np.median(Quan2_ana[model_id]):.4f}，最大为{np.max(Quan2_ana[model_id]):.4f}，最小为{np.min(Quan2_ana[model_id]):.4f}。")        
                print(f"本轮中排名{Quan2_rank[model_id][-1]}/{model_number},在{cluster_id-start_id+1}轮训练中排名平均数为{Quan2_mean:0.4f}，排名中位数为{np.median(Quan2_rank[model_id]):0.4f}，最大排名为{np.max(Quan2_rank[model_id]):0.4f}，最小排名为{np.min(Quan2_rank[model_id]):0.4f}。")

                # SMAPE_mean = np.mean(SMAPE_rank[model_id])
                # if SMAPE_mean < SMAPE_best_mean:
                    # SMAPE_best_model = model_id
                    # SMAPE_best_mean = SMAPE_mean
                # print(f"The SMAPE of {model_name} for {random_cluster} is {SMAPE[model_names[model_id]]:.4f}. ")
                # print(f"本轮进行过Scaler的SMAPE为：{SMAPE_trans[model_name]}，在{cluster_id-start_id+1}轮训练中，SMAPE平均数为{np.mean(SMAPE_ana[model_id]):.4f}，中位数为{np.median(SMAPE_ana[model_id]):.4f}，最大为{np.max(SMAPE_ana[model_id]):.4f}，最小为{np.min(SMAPE_ana[model_id]):.4f}。") 
                # print(f"本轮中排名{SMAPE_rank[model_id][-1]}/{model_number},在{cluster_id-start_id+1}轮训练中排名平均数为{SMAPE_mean:0.4f}，排名中位数为{np.median(SMAPE_rank[model_id]):0.4f}，最大排名为{np.max(SMAPE_rank[model_id]):0.4f}，最小排名为{np.min(SMAPE_rank[model_id]):0.4f}。")

                print(f"The coverage of {model_name} is {Coverage[model_names[model_id]]:.4f}, and the median width is {Median_width[model_names[model_id]]:.4f}, the average width is {Average_width[model_names[model_id]]:.4f}, the maximum width is {Max_width[model_names[model_id]]:.4f}, the minimum width is {Min_width[model_names[model_id]]:.4f}.")

                print("\n")


                df_res=df_res.append({'predict_method':predict_methods[model_id], 
                                      'ensemble_method':ensemble_methods[model_id], 
                                     'CP_method':CP_methods[model_id],
                                      'rmse':RMSE[model_names[model_id]], 'mae':MAE[model_names[model_id]],
                                      'rmse_trans':RMSE_trans[model_names[model_id]], 'mae_trans':MAE_trans[model_names[model_id]],
                                      'rmse_rank':RMSE_rank[model_id][-1], 'mae_rank':MAE_rank[model_id][-1], 
                                      'quantileloss1':Quan1[model_names[model_id]],'quantileloss2':Quan2[model_names[model_id]],
                                      'cluster_label':random_cluster,
                                      'alpha':alpha,'gamma':gamma,
                                     'coverage':Coverage[model_name],
                                     'median width':Median_width[model_name],
                                     'average width':Average_width[model_name],
                                     'max width':Max_width[model_name],
                                     'min width':Min_width[model_name],
                                     'lookback': look_back, 
                                     'choice_interval':choice_interval,
                                     'training_time':time.perf_counter() - start_time_epoch},ignore_index=True)
            print(f"本轮训练中，RMSE表现最好的模型是{model_names[RMSE_best_epoch]}，MAE表现最好的模型是{model_names[MAE_best_epoch]}，Quantile ver.1表现最好的模型是{model_names[Quan1_best_epoch]}，Quantile ver.2表现最好的模型是{model_names[Quan2_best_epoch]}。")
            print(f"截止到本轮，RMSE平均表现最好的是{model_names[RMSE_best_model]}，MAE平均表现最好的模型是{model_names[MAE_best_model]}，Quantile ver.1平均表现最好的模型是{model_names[Quan1_best_model]}，Quantile ver.2平均表现最好的模型是{model_names[Quan2_best_model]}。")
            print(f"本轮训练花费时间{time.perf_counter() - start_time_epoch:.2f}s，训练共花费时间{time.perf_counter() - start_time:.2f}s，平均每轮花费时间{(time.perf_counter() - start_time)/(cluster_id - start_id + 1):.2f}s，")

            if cluster_id % 10 == 0 and cluster_id > 5:
                print("----------------------------保存实验结果----------------------------")
                df_res.to_csv(f'../result/alpha/{filename}_new.csv', index = False)
                if cluster_id % 40 == 0 and cluster_id > 10:
                    df_res.to_csv(f'../result/alpha/{filename}_{cluster_id}_new.csv', index = False)
        df_res.to_csv(f'../result/alpha/{filename}_new.csv', index = False)
        print("完成"+str(dataname)+"集群数据的测试。")