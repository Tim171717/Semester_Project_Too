import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from fabric import Connection
from collections import OrderedDict

c = Connection('tfessler@rainbow', connect_kwargs={"password": "Dl2oP1AjOjO6"})
runname = 'M'
local_dir = "config_files_run_" + runname
remote_dir = "/home/ipa/quanz/user_accounts/tfessler/config_files/run_" + runname
os.makedirs(local_dir, exist_ok=True)


def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)


ms = np.arange(0.6, 2.1, 0.2)
filenames = []

for m in ms:
    filename = f"spectrumVH2O_M={int(m*10)}"
    filenames.append(filename)
    yaml_dict = OrderedDict({
    		"RUN SETTINGS": OrderedDict({
        		"wavelength_range": [3.5, 19.5],
        		"output_folder": "/home/ipa/quanz/user_accounts/tfessler/results/spectrum_run_" + runname + '/' + filename + ".txt",
        		"include_scattering": OrderedDict({
            		"Rayleigh": False,
            		"thermal": False,
            		"direct_light": False,
            		"clouds": False,
        		}),
        		"include_CIA": True,
        		"resolution": 100,
        		"n_layers": 100,
        		"log_top_pressure": -4,
    		}),
    		"TEMPERATURE PARAMETERS": OrderedDict({
        		"parameterization": "polynomial",
        		"a_4": {"truth": 1.67},
        		"a_3": {"truth": 23.12},
        		"a_2": {"truth": 99.7},
        		"a_1": {"truth": 146.63},
        		"a_0": {"truth": 285.22},
    		}),
    		"PHYSICAL PARAMETERS": OrderedDict({
        		"P0": {"truth": 1.02329299228},
        		"d_syst": {"truth": 10.0},
        		"R_pl": {"truth": 1.0},
        		"M_pl": {"truth": float(round(m, 2))},
    		}),
    		"CHEMICAL COMPOSITION PARAMETERS": OrderedDict({
        		"mmw_inert": 28.9,
        		"condensation": True,
        		"N2": {"truth": 0.79},
        		"O2": {"truth": 0.2},
        		"CO2": {"lines": ["CO2"], "truth": 0.00041},
        		"H2O": {"lines": ["H2O"], "truth": 0.001},
        		"H2O_Drying": {"truth": 0.01},
        		"O3": {"lines": ["O3"], "truth": 3.0e-07},
        		"CH4": {"lines": ["CH4"], "truth": 1.7e-06},
        		"CO": {"lines": ["CO"], "truth": 1.23e-7},
    		}),
    		"CLOUD PARAMETERS": OrderedDict({
        		"settings_clouds": "patchy",
        		"cloud_fraction": {"truth": 0.67},
    		}),
	})

    with open(local_dir + '/' + filename + ".yaml", "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)


c.run(f"mkdir -p {remote_dir}")
for file in os.listdir(local_dir):
    local_path = os.path.join(local_dir, file)
    if os.path.isfile(local_path):
        c.put(local_path, remote=f"{remote_dir}/{file}")
        print(f"Uploaded {file}")

c.run('source /home/ipa/quanz/user_accounts/tfessler/myenv/bin/activate')
for filename in filenames:
    c.run(f"python /home/ipa/quanz/user_accounts/tfessler/software/pyRetLIFE/scripts/create_spectrum.py --config ~/config_files/VH2O_create_spectrum.yaml")