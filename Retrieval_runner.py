import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from fabric import Connection
from collections import OrderedDict
import pyautogui
import time

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
spectrum = ''
### --------------------------------------------------------------------------------------------


file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
local_dir = os.path.join(file_dir,'Retrieval_studies', runname)
os.makedirs(local_dir,exist_ok=True)
config_dir = os.path.join(local_dir,'configs')
os.makedirs(config_dir,exist_ok=True)
os.chdir(local_dir)
remote_dir = f"/home/ipa/quanz/user_accounts/tfessler/Retrieval_studies/{runname}"


def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)


filenames = []

for n, mdis in enumerate(mass_distros):
    filename = f"config{n+1}_{mdis[0]}.txt"
    filenames.append(filename)
    yaml_dict = make_yaml_dict(runname)
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

    with open(f"{config_dir}/{filename[:-4]}.yaml", "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)


c.run(f"mkdir -p {remote_dir}/configs")
c.run(f"mkdir -p {remote_dir}/results")
c.run(f"mkdir -p {remote_dir}/spectra")

for file in os.listdir(config_dir):
    local_path = os.path.join(config_dir, file)
    if os.path.isfile(local_path):
        c.put(local_path, remote=f"{remote_dir}/configs/{file}")


### --------------------------------------------
# for filename in filenames:
#     pyautogui.click(2000, 300)
#     pyautogui.write(f"python /home/ipa/quanz/user_accounts/tfessler/software/pyRetLIFE/scripts/create_spectrum.py --config {remote_dir}/{filename[:-4]}.yaml")
#     pyautogui.press('enter')
#     time.sleep(10)
### ----------------------------------------------