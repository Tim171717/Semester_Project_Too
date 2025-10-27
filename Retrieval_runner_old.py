import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from fabric import Connection
from collections import OrderedDict
import pyautogui
import time
import shutil

from helper_functions import *

### --------------------------------------------------------------------------------------------
c = Connection('tfessler@rainbow', connect_kwargs={"password": "Dl2oP1AjOjO6"})
runname = 'Earthlike_Retrieval'
mass_distros = [
    ['known'],
    ['uniform', 0, 10],
    ['gaussian', 1, 0.1],
    ['gaussian', 1, 0.3],
]
m_truth = 1.0
spectrum = 'C:/Users/timlf/PycharmProjects/Semester_Project_Too/spectrumVH2O.txt'
### --------------------------------------------------------------------------------------------


file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
local_dir = os.path.join(file_dir,'Retrieval_studies', runname)
os.makedirs(local_dir,exist_ok=True)
config_dir = os.path.join(local_dir,'configs')
os.makedirs(config_dir,exist_ok=True)
spectra_dir = os.path.join(local_dir,'spectra')
os.makedirs(spectra_dir,exist_ok=True)
shutil.copy(spectrum, f'{spectra_dir}/input_spectrum.txt')
results_dir = os.path.join(local_dir,'results')
os.makedirs(results_dir,exist_ok=True)
os.chdir(local_dir)
home_dir = f"/home/ipa/quanz/user_accounts/tfessler"
remote_dir = f"{home_dir}/Retrieval_studies/{runname}"


def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)

Retnames = []
filenames = []

for n, mdis in enumerate(mass_distros):
    yaml_dict = make_yaml_dict(runname, n+1, mdis[0])
    if mdis[0] == 'known':
        yaml_dict['PHYSICAL PARAMETERS']['M_pl'] = OrderedDict({
                'truth': m_truth
            })
    elif mdis[0] == 'uniform':
        yaml_dict['PHYSICAL PARAMETERS']['M_pl'] = OrderedDict({
            'prior': OrderedDict({
                'kind': 'uniform',
                'prior_specs': OrderedDict({
                    'lower': mdis[1],
                    'upper': mdis[2]
                })
            }),
            'truth': m_truth
        })
    elif mdis[0] == 'gaussian':
        yaml_dict['PHYSICAL PARAMETERS']['M_pl'] = OrderedDict({
            'prior': OrderedDict({
                'kind': 'gaussian',
                'prior_specs': OrderedDict({
                    'mean': mdis[1],
                    'sigma': mdis[2]
                })
            }),
            'truth': m_truth
        })
    elif mdis[0] == 'log-gaussian':
        yaml_dict['PHYSICAL PARAMETERS']['M_pl'] = OrderedDict({
            'prior': OrderedDict({
                'kind': 'log-gaussian',
                'prior_specs': OrderedDict({
                    'log_mean': mdis[1],
                    'log_sigma': mdis[2]
                })
            }),
            'truth': m_truth
        })
    else:
        print("Unrecognized prior type: ", mdis[0])
        continue

    filename = f"config{n + 1}_{mdis[0]}.yaml"
    filenames.append(filename)

    with open(f"{config_dir}/{filename}", "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

    c.run(f"mkdir -p {remote_dir}/results/Retrieval{n+1}_{mdis[0]}")
    Retnames.append(f'Retrieval{n+1}_{mdis[0]}')

c.run(f"mkdir -p {remote_dir}/configs")
c.run(f"mkdir -p {remote_dir}/spectra")

for file in os.listdir(config_dir):
    local_path = os.path.join(config_dir, file)
    if os.path.isfile(local_path):
        c.put(local_path, remote=f"{remote_dir}/configs/{file}")

c.put(spectrum, remote=f"{remote_dir}/spectra/input_spectrum.txt")

nproc = 32 // len(mass_distros)
### --------------------------------------------
for filename, Retname in zip(filenames, Retnames):
    pyautogui.click(2000, 300)
    pyautogui.write(f"nohup nice -n 19 python {home_dir}/software/pyRetLIFE/scripts/run_plotting.py --config {remote_dir}/configs/{filename} --nproc {nproc} &>> {remote_dir}/results/{Retname}/output.txt &")
    pyautogui.press('enter')
    time.sleep(2)
### ----------------------------------------------