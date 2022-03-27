import pandas as pd
import numpy as np 
import random
from utils import * 
import time
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
import pickle 
import argparse

import warnings
warnings.filterwarnings("ignore")

HIBENCH_PATH = Path("/home/cc/HiBench/")
PREFIXES = {
    **dict.fromkeys(["als", "bayes", "gbt", "kmeans", "lda", "linear",
                     "lr", "pca", "rf", "svd", "svm"], "ml"),
    **dict.fromkeys(["dfsioe", "repartition", "sleep", "sort", "terasort",
                     "wordcount"], "micro"),
    **dict.fromkeys(["aggregation", "join", "scan"], "sql"),
    **dict.fromkeys(["identity", "repartition", "wordcount1", "fixwindow"], "streaming"),
    **dict.fromkeys(["pagerank"], "websearch"),
    **dict.fromkeys(["nweight"], "graph")
}

class OFFLINE():
    def __init__(self, configs, power_data, latency_data, threshold, known_config_count=960):
        self.X = configs
        self.N = len(self.X)

        self.threshold = threshold

        random_state = np.random.RandomState(0)
        avail_inds = list(range(self.N))

        self.sampled_ind = random_state.choice(avail_inds, known_config_count, replace=False).tolist()
        self.power = [power_data[i] for i in self.sampled_ind]
        
        latency = [latency_data[i] for i in self.sampled_ind]
        self.predicted_latency = GaussianProcessRegressor().fit(self.X.values[self.sampled_ind], latency).predict(self.X.values)

        self.safety_model = GaussianProcessRegressor()
        
        self.model_type = 'offline'
    
    def add_sample(self, ind, instruction, power):
        if ind in self.sampled_ind:
            pos = self.sampled_ind.index(ind)
            self.power[pos] = power
        else:
            self.power.append(power)
            self.sampled_ind.append(ind)
    
    def choose_configuration(self):
        predicted_power = self.safety_model.fit(self.X.values[self.sampled_ind], self.power).predict(self.X.values)
        avail_inds = [i for i in range(self.N) if predicted_power[i] < self.threshold]

        if len(avail_inds) == 0:
            sorted_power = np.argsort(predicted_power)
            avail_inds = [i for i in sorted_power if i not in self.removed_inds]
            return avail_inds[0]

        new_ind = np.argmin(self.predicted_latency[avail_inds])
        return avail_inds[new_ind]

    def run(self, ind, instruction, power):
        self.add_sample(ind, instruction, power)
        return self.choose_configuration()

def run_offline(init_idx, benchmark, model, directory, sleep_time = 20):
    random.seed(0)
    configs = pd.read_csv('configs.csv', header=0)
    current_ind = init_idx
    sampled_ind = [init_idx]

    ## set initial config params
    write_configurations(configs.iloc[init_idx])
    
    start = time.time()
    apply_system_configurations(configs.iloc[init_idx])
    overhead_h = [time.time() - start]
    overhead_s = []

    clean_perf()

    start_power()

    script_path = HIBENCH_PATH.joinpath(
        "bin/workloads", PREFIXES[benchmark], benchmark, "spark/run.sh")
    process = subprocess.Popen([script_path])

    poll = process.poll()

    start_ind = 2

    time.sleep(10)

    start_perf()

    while poll is None:
        time.sleep(sleep_time)

        max_power = np.max(get_power(start_ind)) # retrieve power data of last `interval` seconds
        avg_instruction = stop_and_average_perf() # retrieve average instruction count

        ## choose new config
        start_s = time.time()
        current_ind = model.run(current_ind, avg_instruction, max_power)
        overhead_s.append(time.time()-start_s)

        start_perf()
        start_ind = get_last_ind()

        ## apply new param settings
        start = time.time()
        apply_system_configurations(configs.iloc[current_ind])
        overhead_h.append(time.time() - start)

        sampled_ind.append(current_ind)

        poll = process.poll()
    
    stop_perf()

    monitor_log, report_data = stop_power(benchmark)

    if poll == 0: 
        write_more_results(benchmark, sampled_ind, monitor_log, report_data, model.model_type, overheads=[overhead_s, overhead_h], directory=directory)


configs = pd.read_csv('configs.csv', header=0)
X = normalized(configs)

parser = argparse.ArgumentParser(description='Process arguments for running Offline')
parser.add_argument('-a', '--application', type=str, 
                    help='application you wish to run')
parser.add_argument('-s', '--starting_config', type=int, 
                    help='starting configurations that Offline should run with')
parser.add_argument('-c', '--constraint', type=float, 
                    help='constraint that Offline should meet')
parser.add_argument('-t', '--time_interval', type=int, 
                    help='time interval that Offline should use')
parser.add_argument('-d', '--save_dir', type=str, 
                    help='directory to save results in')

args = parser.parse_args()

with open(f'data/{args.application}.pkl', 'rb') as f:
    _, latency, energy, power = pickle.load(f)
max_power = [max(i) for i in power]

model = OFFLINE(X, max_power, latency, args.constraint)
run_offline(args.starting_config, args.application, model, directory=args.save_dir, sleep_time = args.time_interval)
