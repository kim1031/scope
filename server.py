import subprocess
import os 
from os import kill, wait
from signal import SIGINT, SIGUSR1
from flask import Flask, request
from werkzeug.serving import WSGIRequestHandler


app = Flask(__name__)
WSGIRequestHandler.protocol_version = "HTTP/1.1"

@app.route("/")
def hello_world():
    return "Hello, World!"

def cleanup():
    ps = subprocess.Popen(["ps", "aux"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    out, err = ps.communicate()
    if ps.returncode != 0:
        print(f"Something went wrong with the cleanup!: {err}")
        return

    for line in out.split("\n"):
        line = [entry for entry in line.split(" ") if len(entry) > 2]
        if len(line) > 1:
            pid, cmdline = line[1], line[-1]
            if "perf" in cmdline or "power" in cmdline or "periodic" in cmdline:
                print(f"Killing process {pid} ({cmdline})")
                kill(int(pid), SIGKILL)

@app.route("/start")
def start_collection():
    cleanup()
    
    # perf = subprocess.Popen(["./perf-sampler"],
    #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #                         universal_newlines=True)
    perf = subprocess.Popen(["python", "periodic-perf.py"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)

    
    power_recorder = subprocess.Popen(
        ["/tmp/power_recorder", "4000", "/tmp/power_data"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)

    return f"{str(perf.pid)},{str(power_recorder.pid)}\n"


@app.route("/stop")
def stop_collection():
    perf_pid = int(request.args.get("perf"))
    power_pid = int(request.args.get("power"))

    kill(perf_pid, SIGINT)
    kill(power_pid, SIGINT)
    
    # Reap child processes
    for _ in range(2):
        wait()
    # Will add error logging later

    return "Done!"


@app.route("/data")
def send_data():
    with open("/tmp/power_data") as f:
        power_data = f.read()

    return power_data

@app.route("/power_data")
def get_power_data():
    p = subprocess.Popen(["cat", "/tmp/power_data"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    return out

@app.route("/clean_perf")
def clean_perf_file():
    os.system("sudo rm /tmp/tmp_perf_data")
    os.system("sudo rm /tmp/perf_data")
    return "Done!"

@app.route("/start_perf")
def start_perf():
    # os.system("perf stat -a -e instructions:u 2> /tmp/tmp_perf_data &")
    os.system("perf stat -a -e instructions 2> /tmp/tmp_perf_data &")
    return "Done!"

@app.route("/stop_perf")
def stop_perf():
    os.system("ps aux | grep perf | awk '{print $2}' | sudo xargs kill -2")
    os.system('cat /tmp/tmp_perf_data >> /tmp/perf_data')
    out = os.popen('cat /tmp/tmp_perf_data').read()
    return out

@app.route("/perf_data")
def get_perf_data():
    with open("/tmp/perf_data") as f:
        perf_data = f.read()
    return perf_data

@app.route("/set_power_limit")
def set_power_limit():
    power = request.args.get("power")

    os.system(f"sudo powercap-set -p intel-rapl -z 0 -c 0 -l {power} -s 999500 -e 1")
    os.system(f"sudo powercap-set -p intel-rapl -z 1 -c 0 -l {power} -s 999500 -e 1")

    return "Done!"

@app.route("/disable_power_limit")
def disable_power_limit():
    power = request.args.get("power")
    os.system(f"sudo powercap-set -p intel-rapl -z 0 -c 0 -l 125000000 -e 0")
    os.system(f"sudo powercap-set -p intel-rapl -z 1 -c 0 -l 125000000 -e 0")

    os.system("sudo powercap-set -p intel-rapl -z 0 -e 0")
    os.system("sudo powercap-set -p intel-rapl -z 1 -e 0")
    return "Done!"

@app.route("/uncore")
def apply_uncore():
    freq = request.args.get("freq")

    p = subprocess.Popen(["bash", "./apply-uncore.sh", f"{freq}"])
    _, _ = p.communicate()
    if p.returncode != 0:
        return ("Could not apply!", 105)

    return "Done!"

@app.route("/core")
def apply_core():
    freq = request.args.get("freq")

    p = subprocess.Popen(["bash", "./apply-freq.sh", f"{freq}"])
    _, _ = p.communicate()
    if p.returncode != 0:
        return ("Could not apply!", 105)

    return "Done!"

def filter_hyperthreading(info, hyperthreading):
    if hyperthreading == 'on':
        return info, []

    hyperthreaded = []
    remaining = []
    for cpu, core, socket in info:
        if cpu != core:
            hyperthreaded.append([cpu, core, socket])
        else:
            remaining.append([cpu, core, socket])
        
    return remaining, hyperthreaded

def filter_sockets(info, nr_sockets):
    filtered = []
    remaining = []
    for cpu, core, socket in info:
        if socket >= nr_sockets:
            filtered.append([cpu, core, socket])
        else:
            remaining.append([cpu, core, socket])
    return remaining, filtered

def filter_cores(info, nr_cores):

    unique_cores = []
    for cpu, core, socket in info:
        if core not in unique_cores:
            unique_cores.append(core)
    filtered_cores = unique_cores[:nr_cores]

    filtered = []
    remaining = []
    for cpu, core, socket in info:
        if core not in filtered_cores:
            filtered.append([cpu, core, socket])
        else:
            remaining.append([cpu, core, socket])
    return remaining, filtered

def set_cpus(cpus_off, cpus_on):
    p1 = subprocess.Popen(["bash", "/tmp/toggle_cpus.sh", "0", f"{cpus_off}"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    p2 = subprocess.Popen(["bash", "/tmp/toggle_cpus.sh", "1", f"{cpus_on}"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    _ = p1.communicate()
    _ = p2.communicate()
    if p1.returncode != 0 or p2.returncode != 0:
        print(f"Could not set cpu")
        exit(1)

    p3 = subprocess.Popen(["ps", "aux"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p3.communicate()
    processes = (out + err).split("\n")
    for line in processes:
        if "/tmp/power_recorder" in line:
            line = line.split()
            pid = int(line[1])
            kill(pid, SIGUSR1)

@app.route("/all")
def apply_hardware():
    uncore = request.args.get("uncore")

    p = subprocess.Popen(["bash", "/tmp/apply-uncore.sh", f"{uncore}"])
    _, _ = p.communicate()
    if p.returncode != 0:
        return ("Could not apply!", 105)

    freq = request.args.get("freq")

    p = subprocess.Popen(["bash", "/tmp/apply-freq.sh", f"{freq}"])
    _, _ = p.communicate()
    if p.returncode != 0:
        return ("Could not apply!", 105)

    info = [[0, 0, 0], [1, 1, 1], [2, 2, 0], [3, 3, 1], [4, 4, 0], [5, 5, 1], [6, 6, 0], [7, 7, 1], [8, 8, 0], [9, 9, 1], [10, 10, 0], [11, 11, 1], [12, 12, 0], [13, 13, 1], [14, 14, 0], [15, 15, 1], [16, 16, 0], [17, 17, 1], [18, 18, 0], [19, 19, 1], [20, 20, 0], [21, 21, 1], [22, 22, 0], [23, 23, 1], [24, 0, 0], [25, 1, 1], [26, 2, 0], [27, 3, 1], [28, 4, 0], [29, 5, 1], [30, 6, 0], [31, 7, 1], [32, 8, 0], [33, 9, 1], [34, 10, 0], [35, 11, 1], [36, 12, 0], [37, 13, 1], [38, 14, 0], [39, 15, 1], [40, 16, 0], [41, 17, 1], [42, 18, 0], [43, 19, 1], [44, 20, 0], [45, 21, 1], [46, 22, 0], [47, 23, 1]]

    command = ["lscpu", "-p"]
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    out, err = p.communicate()
    on_cpus = set([int(i.split(",")[0]) for i in out.split("\n") if len(i.strip()) > 4 and i[0] != "#"])

    to_turn_off = []
    remaining, filtered = filter_sockets(info, int(request.args.get('sockets')))
    to_turn_off += filtered
    remaining, filtered = filter_cores(remaining, int(request.args.get('cores')))
    to_turn_off += filtered
    remaining, filtered = filter_hyperthreading(remaining, request.args.get('hyperthreading'))
    to_turn_off += filtered

    cpus_off = []
    for cpu, _, _ in to_turn_off:
        if cpu in on_cpus:
            cpus_off.append(str(cpu))
    cpus_off = " ".join(cpus_off)

    cpus_on = []
    for cpu, _, _ in remaining:
        if cpu not in on_cpus:
            cpus_on.append(str(cpu))
    cpus_on = " ".join(cpus_on)

    set_cpus(cpus_off, cpus_on)

    return "Done!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
