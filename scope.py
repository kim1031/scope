import pandas as pd
import numpy as np 
import random
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.neural_network import MLPRegressor

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

class SCOPE():
    def __init__(self, configs, threshold, nu=0.5, model_type = 'gp', gamma = 2.5):
        self.X = configs
        self.N = len(self.X)

        self.instructions = []
        self.power = []
        self.sampled_ind = []
        self.last_safe_ind = -1
    
        self.gamma = gamma

        if 'mlp' in model_type:
            self.model = MLPRegressor(random_state=0)
            self.safety_model = MLPRegressor(random_state=0)
        elif 'linear' in model_type:
            self.model = LinearRegression()
            self.safety_model = LinearRegression()
        elif 'rf' in model_type:
            self.model = RandomForestRegressor(random_state=0)
            self.safety_model = RandomForestRegressor(random_state=0)
        else:
            self.model = GaussianProcessRegressor(random_state=0, kernel=Matern(nu=nu))
            self.safety_model = GaussianProcessRegressor(random_state=0, kernel=Matern(nu=nu))   
        
        if model_type != 'gp':
            self.model_type = 'scope_'+model_type
        elif gamma >= 2.25:
            self.model_type = 'scope_nr'
        else:
            self.model_type = 'scope_' + str(int(gamma*100))
        
        self.random_state = np.random.RandomState(0)

        self.threshold = threshold
        self.predicted_power = []
        self.predicted_reward = []
            
    def add_sample(self, ind, instruction, power):
        if ind in self.sampled_ind:
            pos = self.sampled_ind.index(ind)
            self.instructions[pos] = instruction
            self.power[pos] = power
        else:
            self.power.append(power)
            self.instructions.append(instruction)
            self.sampled_ind.append(ind)
        
        if self.last_safe_ind == -1 or power <= self.threshold:
            self.last_safe_ind = ind
    
    def train_models(self):
        self.predicted_power = self.safety_model.fit(self.X.values[self.sampled_ind], self.power).predict(self.X.values)
        self.predicted_reward = self.model.fit(self.X.values[self.sampled_ind], self.instructions).predict(self.X.values)

    def get_avail_inds(self):
        dists = np.linalg.norm(np.subtract(self.X.values, self.X.values[self.last_safe_ind]), axis = 1)
        avail_inds = [x for x in np.where(dists <= self.gamma)[0] if x not in self.sampled_ind]
        return avail_inds

    def choose_any_configuration(self):
        avail_inds = self.get_avail_inds()

        if len(avail_inds) == 0:
            aq = np.argsort(self.predicted_power)
            aq_unique = [x for x in aq if x in self.sampled_ind]
            return aq_unique[0]

        ## identify next config to sample
        safe_set = [x for x in avail_inds if self.predicted_power[x] < self.threshold]

        if len(safe_set) == 0:
            aq = np.argsort(self.predicted_power)
            aq_unique = [x for x in aq if x in avail_inds]
            return aq_unique[0]
                          
        aq = np.argsort(-self.predicted_reward)         
        aq_unique = [x for x in aq if x in safe_set]
        new_idx = aq_unique[0]  
        return new_idx  
    
    def run(self, ind, instruction, power):
        self.add_sample(ind, instruction, power)
        self.train_models()
        return self.choose_any_configuration()

def run_scope(init_idx, benchmark, model, threshold, directory, sleep_time = 20):
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
        start = time.time()
        while (time.time() - start) < sleep_time:
            time.sleep(1)
            powers = get_power(start_ind)
            if (len(powers) > 0 and np.max(powers)> threshold) or (process.poll() is not None):
                break
        
        if process.poll() is not None:
            break

        ## retrieve average instruction count
        avg_instruction = stop_and_average_perf()

        ## choose new config
        start_s = time.time()
        current_ind = model.run(current_ind, avg_instruction, np.max(powers))
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
