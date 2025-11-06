import statsmodels.tsa.stattools
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.autograd import Variable
import os
import time
import random


def cluster_select(df, cluster_id, metric_name, lower_bound=1000):
    df['cluster_' + metric_name + '_mean'] = df.groupby([cluster_id])[metric_name].transform(lambda s: s.median())
    df = df.loc[df['cluster_' + metric_name + '_mean'] >= lower_bound]
    df.drop(columns=['cluster_' + metric_name + '_mean'], inplace=True)
    df.sort_values(by=[cluster_id, 'datetime'], inplace=True)
    select_cluster = df[cluster_id].unique()
    return select_cluster, df[df[cluster_id].isin(select_cluster)]


def judge_stationarity(data_sanya_one):
    dftest = statsmodels.tsa.stattools.adfuller(data_sanya_one)
    # print(dftest)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    stationarity = 1
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        if dftest[0] > value:
                stationarity = 0
    # print(dfoutput)
    # print("是否平稳(1/0): %d" %(stationarity))
    return stationarity

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

def diff_func(data, order = 1):
    for i in range(order):
        data_diff = [data[i] - data[i-1] for i in range(1,len(data))]
        data = data_diff
    return data_diff

class timeseries(Dataset):
    def __init__(self, trainX, trainY):
        self.X = trainX
        self.Y = trainY

    def __getitem__(self, idx):
        x_t = self.X[idx]
        y_t = self.Y[idx]
        return x_t, y_t

    def __len__(self):
        return len(self.X)


def predict_iteration(model, data_X, look_ahead):
    '''
    model: 训练好的模型
    data_X: 用于预测的数据（数组）shape: [batch_size, look_back]
    res: ndarray数组 shape: [look_ahead]
    '''
    # data_X.reshape(-1, look_back, 1)
    batch_size = data_X.shape[0]
    res = []
    data = np.array(data_X)

    for i in range(look_ahead):
        data_tensor = torch.from_numpy(data)
        data_tensor = Variable(data_tensor)
        pred = model(data_tensor)
        pred = np.squeeze(pred)
        res.append(pred.detach().numpy())

        data = data[:, 1:]
        pred = pred.reshape((batch_size, 1, 1))
        data = np.append(data, pred.detach().numpy(), axis=1)

    res = np.array(res)
    return res

def data_producer(dataset, random_serie_number):
    if dataset == "FaaS":
        st = time.perf_counter()
        df = pd.read_parquet("../data/part-00000-b72e6747-b68c-4d59-81d7-d8f66e83598c-c000.gz.parquet")
        df = df[["time", "id","producer_speed"]]
        df.rename(columns={'time':'datetime','producer_speed':'y'},inplace=True)
        series_ids = df["id"].unique()
        print(f"load use {time.perf_counter() - st:.2f}s")
        print("总共待预测序列有%d条" % len(series_ids))
        select_cluster, df_nonzero = cluster_select(df,"id", "y", 0.5)
        print("除去metric平均值小于0.5的集群，总共待预测序列有%d条" % len(df_nonzero["id"].unique()))
        random_series = random.sample(range(0,select_cluster.shape[0]),random_serie_number)
        print(f"随机选出了{random_serie_number}条序列")
        random_series = select_cluster[random_series]
<<<<<<< HEAD
        random_series =['ouamoeww-1gvqal8o','2cz6dpny-igulm2gl','ljj5btuk-beqf1rrw',
 'eqjsor16-xitruh4v' ,'f7fsz8yv-4pcy68l8', 'qaq6exe4-3t5chvsj',
 'q2t6dg1j-j4ygfrhy', 'h2lkr6zp-b2ooc7nc' ,'3hhbkfa3-ey6rj33s',
 'cmy17msz-3sj5k3sf']
=======
>>>>>>> d71b8ae5951d40bd6752d4bc0475319152b70f14
        return df_nonzero, random_series
        # for i_ in range(series_number)
        # random_starts = random.sample(range(0, series_length),)
    elif dataset == "Abase":
        st = time.perf_counter()
        file_name = "abase_redis_total_read_qps_20220531_20220629_full_dept"
        df_nonzero = pd.read_csv(f"../data/{file_name}.csv")
        df_nonzero['date'] = df_nonzero['datetime'].map(lambda x: x.split()[0])
        df_nonzero['date'] = pd.to_datetime(df_nonzero['date'], dayfirst=True, format="%Y-%m-%d")
        df_nonzero['datetime'] = pd.to_datetime(df_nonzero['datetime'], dayfirst=True)
        print(f"load use {time.perf_counter() - st:.2f}s")
        print("总共待预测序列有%d条" % len(df_nonzero["cluster_label"].unique()))
        df_nonzero.rename(columns={'cluster_label':'id','qps':'y'},inplace=True)
        select_cluster, df_nonzero = cluster_select(df_nonzero,"id", "y")
        print("除去metric平均值小于1000的集群，总共待预测序列有%d条" % len(df_nonzero["id"].unique()))
        random_series = random.sample(range(0,select_cluster.shape[0]),random_serie_number)
        random_series = select_cluster[random_series]
<<<<<<< HEAD
        random_series=['toutiao.redis.ftyche' ,'toutiao.redis.aweme_user_teenager_predict',
 'toutiao.redis.im_group_active_fans',
 'toutiao.redis.lark_message_degrade_a', 'toutiao.redis.aweme_cvr_djw_test',
 'toutiao.redis.query_rewrite_cache_a' ,'toutiao.redis.aweme_review_dedup',
 'toutiao.redis.delay_distributed_cn',
 'toutiao.redis.entity_info_author_data',
 'toutiao.redis.fproject_supernormal_growth']
       #random_series=np.load('Abase_clusters.npy', allow_pickle=True)
        print(f"随机选出了{random_serie_number}条序列")
        return df_nonzero, random_series
    elif dataset == "Alibaba":
        st = time.perf_counter()
        file_name = "machine_usage"
        df_nonzero = pd.read_csv(f"../data/{file_name}.csv")
        df_nonzero.columns = ["machine_id","TIME_STAMP","cpu_util_percent","mem_util_percent","mem_gps","mpki","net_in","net_out","disk_usage_percent"]
        # df_nonzero['date'] = df_nonzero['datetime'].map(lambda x: x.split()[0])
        # df_nonzero['date'] = pd.to_datetime(df_nonzero['date'], dayfirst=True, format="%Y-%m-%d")
        # df_nonzero['datetime'] = pd.to_datetime(df_nonzero['datetime'], dayfirst=True)
        df_nonzero = df_nonzero[["machine_id","cpu_util_percent","TIME_STAMP"]]
        df_nonzero.rename(columns={'machine_id':'id','cpu_util_percent':'y'},inplace=True)
        select_cluster = df_nonzero["id"].unique()
        print(f"load use {time.perf_counter() - st:.2f}s")
        print("总共待预测序列有%d条" % len(df_nonzero["id"].unique()))
        random_series = random.sample(range(0,select_cluster.shape[0]),random_serie_number)
        random_series = select_cluster[random_series]
        random_series=['m_2580','m_1989','m_1949', 'm_2948' ,'m_2701', 'm_2043', 'm_2400', 'm_2003','m_2998' ,'m_2677']
        #random_series=np.load('Alibaba_clusters.npy', allow_pickle=True)
        print(f"随机选出了{random_serie_number}条序列")
        return df_nonzero, random_series
    
=======
        print(f"随机选出了{random_serie_number}条序列")
        return df_nonzero, random_series
>>>>>>> d71b8ae5951d40bd6752d4bc0475319152b70f14
