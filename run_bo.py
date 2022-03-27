import pandas as pd
import numpy as np 
import random
from utils import * 
import time
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
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

class BO_ONLINE():
    def __init__(self, configs):
        self.X = configs.values
        self.instructions = []
        self.sampled_ind = []
        self.model = GaussianProcessRegressor(random_state=0, kernel=Matern(nu=0.5))
        self.model_type = 'bo'
    
    def add_sample(self, ind, instruction, power):
        self.instructions.append(instruction)
        self.sampled_ind.append(ind)
    
    def choose_next_configuration(self):
        ## Fit surrogate model        
        mean, std = self.model.fit(self.X[self.sampled_ind], self.instructions).predict(self.X, return_std = True)
        
        ## Query function
        opt = np.max(self.instructions)
        with np.errstate(divide='warn'):
            imp = mean - opt
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0            
        aq = np.argsort(ei)[::-1]           
        aq_unique = [x for x in aq if x not in self.sampled_ind]
        new_idx = aq_unique[0]        

        return new_idx

def run_bo(init_idx, benchmark, model, directory, sleep_time = 20):
    random.seed(0)
    configs = pd.read_csv('configs.csv', header=0)
    current_ind = init_idx
    sampled_ind = [init_idx]

    ## set initial config params
    write_configurations(configs.iloc[init_idx])
    
    start = time.time()
    apply_system_configurations(configs.iloc[init_idx])
    end = time.time()
    overhead_h = [end-start]
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

        power = get_power(start_ind) # retrieve power data of last `interval` seconds
        avg_instruction = stop_and_average_perf() # retrieve average instruction count

        ## choose new config
        start_s = time.time()
        model.add_sample(current_ind, avg_instruction, power)
        current_ind = model.choose_next_configuration()
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

parser = argparse.ArgumentParser(description='Process arguments for running BO')
parser.add_argument('-a', '--application', type=str, 
                    help='application you wish to run')
parser.add_argument('-s', '--starting_config', type=int, 
                    help='starting configurations that BO should run with')
parser.add_argument('-t', '--time_interval', type=int, 
                    help='time interval that BO should use')
parser.add_argument('-d', '--save_dir', type=str, 
                    help='directory to save results in')

args = parser.parse_args()

model = BO_ONLINE(X)
run_bo(args.starting_config, args.application, model, directory=args.save_dir, sleep_time = args.time_interval)

