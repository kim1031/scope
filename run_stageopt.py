import pandas as pd
import numpy as np 
import random
from utils import * 
import time
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
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

class STAGEOPT():
    def __init__(self, configs, threshold, nu=0.5, model_type='stageopt'):
        self.X = configs
        self.N = len(self.X)

        self.instructions = []
        self.power = []
        self.sampled_ind = []

        self.beta = 1
        self.similar = 0
    
        self.model = GaussianProcessRegressor(random_state=0, kernel=Matern(nu=nu))
        self.safety_model = GaussianProcessRegressor(random_state=0, kernel=Matern(nu=nu))   
        
        self.safe_set = []

        self.model_type = model_type
        
        self.random_state = np.random.RandomState(0)

        self.threshold = threshold
            
    def add_sample(self, ind, instruction, power):
        self.power.append(power)
        self.instructions.append(instruction)
        self.sampled_ind.append(ind)
    
    def expand_safeset(self):
        mean, std = self.safety_model.fit(self.X.values[self.sampled_ind], self.power).predict(self.X.values, return_std = True)

        ucb = mean + self.beta * std
        new_safe_set = [x for x in range(self.N) if ucb[x] <= self.threshold]

        if len(new_safe_set) == len(self.safe_set):
            self.similar += 1
        else:
            self.similar = 0
        
        self.safe_set = new_safe_set
        stds = [std[x] for x in self.safe_set]

        without_sampled = [x for x in self.safe_set if x not in self.sampled_ind]
        if len(without_sampled) == 0:
            aq = np.argsort(ucb)
            aq_unique = [x for x in aq if x not in self.sampled_ind]
            return aq_unique[0]
        
        aq = np.argsort(-np.array(stds))
        aq_unique = [x for x in aq if x not in self.sampled_ind]
        return self.safe_set[aq_unique[0]]

    def optimize_latency(self):
        mean, std = self.safety_model.fit(self.X.values[self.sampled_ind], self.power).predict(self.X.values, return_std = True)
        ucb = mean + self.beta * std

        mean, std = self.model.fit(self.X.values[self.sampled_ind], self.instructions).predict(self.X.values, return_std = True)
        ucb_l = mean + self.beta * std

        aq = np.argsort(-ucb_l)
        aq_unique = [x for x in aq if x in self.safe_set]

        if len(aq_unique) == 0:
            return np.argmin(ucb)

        return aq_unique[0]

    def choose_configuration(self):
        if len(self.sampled_ind) >= 5 or self.similar >= 3:
            return self.optimize_latency()
        return self.expand_safeset()
    
    def run(self, ind, instruction, power):
        self.add_sample(ind, instruction, power)
        return self.choose_configuration()


def run_stageopt(init_idx, benchmark, model, threshold, directory, sleep_time = 20):
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
    
    ## sleep for 10 at start since this is mostly just starting up
    time.sleep(10)
    start_ind = get_last_ind()

    start_perf()

    while poll is None:
        time.sleep(sleep_time)

        max_power = np.max(get_power(start_ind)) # retrieve power data of last `interval` seconds
        avg_instruction = stop_and_average_perf() # retrieve average instruction count

        ## choose new config
        start_s = time.time()
        current_ind = model.run(current_ind, avg_instruction, max_power)
        overhead_s.append(time.time()-start_s)

        ## apply new param settings
        start = time.time()
        apply_system_configurations(configs.iloc[current_ind])
        overhead_h.append(time.time()-start)

        start_perf()
        start_ind = get_last_ind()
            
        sampled_ind.append(current_ind)

        poll = process.poll()
    
    stop_perf()

    monitor_log, report_data = stop_power(benchmark)

    poll = process.poll()
    if poll == 0: 
        # workload finished successfully
        write_more_results(benchmark, sampled_ind, monitor_log, report_data, model.model_type, overheads=[overhead_s, overhead_h], directory=directory)

configs = pd.read_csv('configs.csv', header=0)
X = normalized(configs)

parser = argparse.ArgumentParser(description='Process arguments for running StageOPT')
parser.add_argument('-a', '--application', type=str, 
                    help='application you wish to run')
parser.add_argument('-s', '--starting_config', type=int, 
                    help='starting configurations that StageOPT should run with')
parser.add_argument('-c', '--constraint', type=float, 
                    help='constraint that StageOPT should meet')
parser.add_argument('-t', '--time_interval', type=int, 
                    help='time interval that StageOPT should use')
parser.add_argument('-d', '--save_dir', type=str, 
                    help='directory to save results in')

args = parser.parse_args()

model = STAGEOPT(X, args.constraint, model_type='stageopt')
run_stageopt(args.starting_config, args.application, model, args.constraint, directory=args.save_dir, sleep_time = args.time_interval)