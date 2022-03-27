import pandas as pd
import numpy as np
import pickle

workload = 'wordcount'

workers = ['10.52.2.220','10.52.2.140','10.52.0.185','10.52.3.58']

X = pd.read_csv('all_conf.csv', header = 0)

N = len(X.values)

energy = []
power = []
latency = []

for i in range(N):

    temp_e = []
    temp_p = []

    with open(f'{workload}/{workload}_{i}/{workers[0]}', 'r') as f:
        data = f.readlines()
    for l in data[2:]:
        parsed = l.strip().split(',')
        temp_e.append(int(parsed[1]))
        temp_p.append(float(parsed[2]))

    for w in workers[1:]:
        with open(f'{workload}/{workload}_{i}/{w}', 'r') as f:
            data = f.readlines()
        ind = 0
        for l in data[2:]:
            if ind >= len(temp_e):
                break

            parsed = l.strip().split(',')
            temp_e[ind] += int(parsed[1])
            temp_p[ind] += float(parsed[2])

            ind += 1
        
        if ind < len(temp_e):
            temp_e = temp_e[:ind]
            temp_p = temp_p[:ind]
    
    latency.append(len(temp_e))
    energy.append(np.array(temp_e)/4)
    power.append(np.array(temp_p)/4)

with open(f'{workload}.pkl', 'wb') as f:
    pickle.dump([X, latency, energy, power], f)
    