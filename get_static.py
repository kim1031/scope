import pandas as pd
import numpy as np 
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process arguments for getting Static results')
parser.add_argument('-a', '--application', type=str, 
                    help='application you wish to run')
parser.add_argument('-s', '--starting_config', type=int, 
                    help='starting configurations you wish to get results for')

args = parser.parse_args()

with open(f"data/{args.application}.pkl", 'rb') as f:
    X, latency, energy, power = pickle.load(f)

print(latency[args.starting_config], max(power[args.starting_config]))
