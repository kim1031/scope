import pandas as pd
import numpy as np 
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process arguments for getting Oracle results')
parser.add_argument('-a', '--application', type=str, 
                    help='application you wish to run')
parser.add_argument('-c', '--constraint', type=float, 
                    help='constraint that oracle should meet')

args = parser.parse_args()

with open(f"data/{args.application}.pkl", 'rb') as f:
    X, latency, energy, power = pickle.load(f)

safe_configs = [x for x in range(1920) if max(power[x]) <= args.constraint]
latencies = np.array(latency)[safe_configs]
best_ind = np.argmin(latencies)

print(latency[safe_configs[best_ind]])
