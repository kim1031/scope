import pandas as pd
import numpy as np 
import random
from utils import * 
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

def rapl(init_idx, benchmark, directory, model_name = 'rapl'):
    clean_perf()
    start_power()

    script_path = HIBENCH_PATH.joinpath(
        "bin/workloads", PREFIXES[benchmark], benchmark, "spark/run.sh")
    process = subprocess.Popen([script_path])

    process.wait()

    monitor_log, report_data = stop_power(benchmark)

    write_more_results(benchmark, [init_idx], monitor_log, report_data, model_name, directory=directory)

parser = argparse.ArgumentParser(description='Process arguments for running RAPL')
parser.add_argument('-a', '--application', type=str, 
                    help='application you wish to run')
parser.add_argument('-c', '--constraint', type=float, 
                    help='constraint that oracle should meet')
parser.add_argument('-d', '--save_dir', type=str, 
                    help='directory to save results in')

args = parser.parse_args()

mult = 1000000 // 2

configs = pd.read_csv('configs.csv', header=0)
apply_system_configurations(configs.iloc[959])

disable_power_limit()

rapl(959, args.application, directory=args.save_dir)

disable_power_limit()