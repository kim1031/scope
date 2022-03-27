import requests
import sys 

session = requests.Session()

params = {"uncore": sys.argv[2], 'hyperthreading':sys.argv[3], 'cores':sys.argv[4], 'sockets':sys.argv[5], 'freq':sys.argv[6]}

resp = session.get(f"http://{sys.argv[1]}:5000/reconfigure", params=params)

if resp.status_code != 200:
    print(f"Error setting uncore frequencues on {sys.argv[1]}. Exiting...")
    exit()