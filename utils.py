import subprocess
from pathlib import Path
import time
import requests 
import numpy as np 
import pandas as pd

master = '10.52.3.118'
slaves = ['10.52.2.85','10.52.3.244','10.52.3.76','10.52.0.43']

HIBENCH_PATH = Path("/home/cc/HiBench/")

def write_configurations(config):
    to_write = ("hibench.spark.home    /home/cc/spark\n"
                "hibench.spark.master  spark://master:7077\n")
    for k, v in config.items():
        if k.lower() == "cpufreq":
            continue
        to_write += f"{k}    {v}\n"
    to_write += ("spark.eventLog.enabled              true\n"
                 "spark.eventLog.dir                  /home/cc/spark-events\n"
                 "spark.history.fs.logDirectory       /home/cc/spark-events\n"
                 "spark.metrics.appStatusSource.enabled              true\n")

    outpath = HIBENCH_PATH.joinpath("conf", "spark.conf")
    with open(outpath, "w") as f:
        f.write(to_write)

def apply_system_configurations(configuration):
    target_max = configuration["uncore_min"]
    target_min = configuration["uncore_max"]
    final = (target_min << 8) + target_max
    final_hex = hex(final)

    procs = []
    for w in [master]+slaves:
        cmd = ["python", "send_request.py", w, str(final_hex), configuration['hyperthreading'], str(configuration['cores']), str(configuration['sockets']), str(configuration['cpufreq'])]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        procs.append(p)
    
    for p in procs:
        p.communicate()

def start_power():
    for slave in slaves:
        subprocess.Popen(["ssh", f"cc@{slave}", "sudo /tmp/power_recorder 1000 /tmp/power_data"])
    subprocess.Popen(["sudo", "/tmp/power_recorder", "1000", "/tmp/power_data"])

def stop_power(benchmark):
    ## kill process in master
    subprocess.Popen("ps aux | grep power | awk '{print $2}' | sudo xargs kill -2", shell=True).wait()

    ## kill process in slave nodes
    for slave in slaves:
        subprocess.Popen(["ssh", slave, "ps aux | grep power | awk '{print $2}' | sudo xargs kill -2"]).wait()
        
    ## collect monitor log
    log_path = HIBENCH_PATH.joinpath(
        "report", benchmark, "spark", "monitor.log")
    if not log_path.is_file():
        print("monitor.log file not found... Exiting")
        exit()

    with open(log_path) as f:
        monitor_log = f.read()
    log_path.unlink()
    
    ## collect data
    report_path = HIBENCH_PATH.joinpath("report", "hibench.report")
    if not report_path.is_file():
        print("hibench.report file not found... Exiting")
        exit()

    with open(report_path) as f:
        lines = [e.strip().split(" ") for e in f.readlines() if len(e) > 10]
    exp_line = [e for e in lines[-1] if len(e) > 0]
    report_data = " ".join(exp_line)

    return monitor_log, report_data

def write_results(benchmark, init_idx, sampled_ind, monitor_log, report_data, model_type, start_inds = [], overheads = [], directory="test_scope"):
    experiment_direc = Path(f"{directory}/{benchmark}_{model_type}_{init_idx}")
    experiment_direc.mkdir(parents=True, exist_ok=True)

    # Write data for slaves
    procs = []
    for slave in slaves:
        p0 = subprocess.Popen(["scp", f"cc@{slave}:/tmp/power_data", f"/tmp/power_data_{slave}"])
        procs.append(p0)
    
    for s in range(4):
        outfile = experiment_direc.joinpath(f"{slaves[s]}")
        
        _ = procs[s].communicate()
        
        p1 = subprocess.Popen(["cp", f"/tmp/power_data_{slaves[s]}", outfile])
        _ = p1.communicate()
    
    # Write instruction count data just in case
    procs = []
    for slave in slaves:
        outfile = experiment_direc.joinpath(f"{slave}_instruction_count")
        p = subprocess.Popen(["scp", f"cc@{slave}:/tmp/perf_data", outfile])
        _ = p.communicate()

    # Write data for master
    outfile = experiment_direc.joinpath("master")
    p1 = subprocess.Popen(["cp", f"/tmp/power_data", outfile])
    _ = p1.communicate()

    report_outfile = experiment_direc.joinpath("report")
    with report_outfile.open("w") as f:
        f.write(monitor_log)
        f.write(report_data)
    
    config_outfile = experiment_direc.joinpath("config")
    with config_outfile.open("w") as f:
        for s in sampled_ind:
            f.write(str(s)+"\n")

    if start_inds:
        inds_outfile = experiment_direc.joinpath("start_inds")
        with inds_outfile.open("w") as f:
            f.write(str(start_inds))
    
    if overheads:
        overhead_outfile = experiment_direc.joinpath("overheads")
        with overhead_outfile.open("w") as f:
            f.write(str(overheads[0])+"\n"+str(overheads[1]))

def write_more_results(benchmark, sampled_ind, monitor_log, report_data, model_type, overheads = [], directory="test_scope"):
    experiment_direc = Path(f"{directory}/{benchmark}_{model_type}_{sampled_ind[0]}")
    experiment_direc.mkdir(parents=True, exist_ok=True)

    # Write data for slaves
    procs = []
    for slave in slaves:
        p0 = subprocess.Popen(["scp", f"cc@{slave}:/tmp/power_data", f"/tmp/power_data_{slave}"])
        procs.append(p0)
    
    for s in range(4):
        outfile = experiment_direc.joinpath(f"{slaves[s]}")
        
        _ = procs[s].communicate()
        
        p1 = subprocess.Popen(["cp", f"/tmp/power_data_{slaves[s]}", outfile])
        _ = p1.communicate()
    
    # Write instruction count data just in case
    procs = []
    for slave in slaves:
        outfile = experiment_direc.joinpath(f"{slave}_instruction_count")
        p = subprocess.Popen(["scp", f"cc@{slave}:/tmp/perf_data", outfile])
        _ = p.communicate()

    # Write data for master
    outfile = experiment_direc.joinpath("master")
    p1 = subprocess.Popen(["cp", f"/tmp/power_data", outfile])
    _ = p1.communicate()

    report_outfile = experiment_direc.joinpath("report")
    with report_outfile.open("w") as f:
        f.write(monitor_log)
        f.write(report_data)
    
    configs = pd.read_csv('configs.csv', header=0)
    config_outfile = experiment_direc.joinpath("config")
    with config_outfile.open("w") as f:
        for s in sampled_ind:
            f.write(str(dict(configs.iloc[s]))+"\n")
    
    if overheads:
        overhead_outfile = experiment_direc.joinpath("overheads")
        with overhead_outfile.open("w") as f:
            stringed = [str(i) for i in overheads]
            f.write('\n'.join(stringed))

def normalized(configs):
    data = configs[['hyperthreading', 'sockets', 'cores', 'uncore_min', 'cpufreq']]
    data["hyperthreading"].replace({"off": -1, "on": 1}, inplace=True)
    norm_data = (data - data.mean()) / (data.max() - data.min())
    return norm_data

def get_power(start_ind):
    ## get all power indicies, filter out 0, -nan as necessary
    session = requests.Session()
    powers = []
    for slave in slaves:
        resp = session.get(f"http://{slave}:5000/power_data")
        response = resp.text.split('\n')[start_ind:-1]
        one_power = [float(r.split(',')[-1]) for r in response if 'nan' not in r and r[-1] != '0']
        if len(powers) == 0:
            powers = np.array(one_power)
        else:
            lens = min(len(one_power), len(powers))
            powers = powers[:lens] + one_power[:lens]
        
    return powers/4

def get_last_power():
    ## get all power indicies, filter out 0, -nan as necessary
    session = requests.Session()
    responses = 0
    num = 0
    for slave in slaves:
        resp = session.get(f"http://{slave}:5000/power_data")
        last_line = resp.text.split('\n')[-2]
        if 'nan' not in last_line:
            responses += float(last_line.split(',')[-1])
            num += 1
    
    if num == 0:
        return 0
    
    return responses/num 

def get_last_ind():
    session = requests.Session()
    final_ind = 0
    for slave in slaves:
        resp = session.get(f"http://{slave}:5000/power_data")
        lines = resp.text.split('\n')
        final_ind += len(lines)
    
    return int(final_ind/4)-1

def clean_perf():
    session = requests.Session()
    for slave in slaves:
        session.get(f"http://{slave}:5000/clean_perf")

def start_perf():
    session = requests.Session()
    for slave in slaves:
        session.get(f"http://{slave}:5000/start_perf")

def stop_perf():
    session = requests.Session()
    for slave in slaves:
        session.get(f"http://{slave}:5000/stop_perf")

def stop_and_average_perf():
    session = requests.Session()
    count = 0
    duration = 0
    for slave in slaves:
        data = session.get(f"http://{slave}:5000/stop_perf")
        try:
            count += int(data.text.split('\n')[3].split()[0])
            duration += float(data.text.split('\n')[-3].split()[0])
        except:
            pass
    if duration == 0:
        return -1
    return count/duration

def set_power_limit(lim):
    session = requests.Session()
    for slave in slaves:
        session.get(f"http://{slave}:5000/set_power_limit", params = {'power':lim})

def disable_power_limit():
    session = requests.Session()
    for slave in slaves:
        session.get(f"http://{slave}:5000/disable_power_limit")

def get_data():
    instruction_count = 0
    duration = 0
    for slave in slaves:
        p = subprocess.Popen(["scp", f"cc@{slave}:/tmp/perf_data", f"/tmp/perf_data_{slave}"])
        p.communicate()

        with open(f"/tmp/perf_data_{slave}", 'r') as f:
            data = f.read()
            print(data)
            instruction_count += int(data.split('\n')[3].split()[0])
            duration += float(data.split('\n')[-3].split()[0])
    
    return instruction_count/4, duration/4