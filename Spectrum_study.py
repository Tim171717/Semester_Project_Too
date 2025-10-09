import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from fabric import Connection
from collections import OrderedDict
import pyautogui
import time

c = Connection('tfessler@rainbow', connect_kwargs={"password": "Dl2oP1AjOjO6"})
runname = 'bigger_M'
local_dir = "Run_" + runname
remote_dir = "/home/ipa/quanz/user_accounts/tfessler/config_files/run_" + runname
remote_outfol = "/home/ipa/quanz/user_accounts/tfessler/results/spectrum_run_" + runname
os.makedirs(local_dir, exist_ok=True)
file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)


def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)


ms = np.arange(0.6, 4.1, 0.4)
filenames = []

for m in ms:
    filename = f"spectrumVH2O_M={int(m*10)}.txt"
    filenames.append(filename)
    yaml_dict = OrderedDict({
    		"RUN SETTINGS": OrderedDict({
        		"wavelength_range": [3.5, 19.5],
        		"output_folder": remote_outfol + '/' + filename,
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

    with open(local_dir + '/' + filename[:-4] + ".yaml", "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)


c.run(f"mkdir -p {remote_dir}")
c.run(f"mkdir -p {remote_outfol}")
for file in os.listdir(local_dir):
    local_path = os.path.join(local_dir, file)
    if os.path.isfile(local_path):
        c.put(local_path, remote=f"{remote_dir}/{file}")


### --------------------------------------------
for filename in filenames:
    pyautogui.click(2000, 300)
    pyautogui.write(f"python /home/ipa/quanz/user_accounts/tfessler/software/pyRetLIFE/scripts/create_spectrum.py --config {remote_dir}/{filename[:-4]}.yaml")
    pyautogui.press('enter')
    time.sleep(10)

pyautogui.click(2000, 1000)
pyautogui.write(f'scp -r tfessler')
pyautogui.hotkey("ctrl", "alt", "2")
pyautogui.write(f'rainbow:{remote_outfol} {os.path.join(file_dir,local_dir)}')
pyautogui.press('enter')
time.sleep(1)
pyautogui.write('Dl2oP1AjOjO6')
pyautogui.press('enter')
time.sleep(3)
### ----------------------------------------------


plt.figure(figsize=(8, 5))

for filename in filenames:
    path = os.path.join(local_dir + '/spectrum_run_' + runname, filename)
    data = np.loadtxt(path, skiprows=0)
    plt.plot(data[:, 0], data[:, 1], label='M = ' + str(int(filename[15:len(filename)-4])/10))
plt.xlabel('wavelength')
plt.ylabel('intensity')
plt.title('Spectrum run ' + runname)
plt.legend()
plt.tight_layout()
plt.savefig('spectrum_comparisons/spectrum_run_' + runname + '.png')
plt.show()





















