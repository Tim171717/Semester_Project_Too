from argparse import ArgumentParser, Namespace
import os
import yaml
from collections import OrderedDict
import shlex,subprocess
import datetime

try:
    # Try normal import if pyRetLIFE is in your PYTHONPATH or installed
    from pyretlife.retrieval_plotting.run_plotting import retrieval_plotting_object
    import pyretlife
    print("pyretlife package path:", pyretlife.__file__)
    print("✅ Successfully imported from installed pyRetLIFE package.")
except ModuleNotFoundError:
    # If that fails, try adding your local path
    import sys
    sys.path.append("/home/tfessler/software/pyRetLIFE")
    try:
        from pyretlife.retrieval_plotting.run_plotting import retrieval_plotting_object
        import pyretlife
        print("pyretlife package path:", pyretlife.__file__)
        print("✅ Successfully imported after adding local path.")
    except Exception as e:
        print("❌ Import failed even after adding local path:", e)


def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)


def get_cli_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":


    # Read the command line arguments (config file path)
    args = get_cli_arguments()
    with open(str(args.config), 'r') as file:
        config_file = yaml.safe_load(file)

    directory = str(config_file['directory'])
    subprocess.run(f"mkdir -p {directory}/configs", shell=True, check=True)
    subprocess.run(f"mkdir -p {directory}/results", shell=True, check=True)

    ms = [float(m) for m in config_file['ms']]
    filenames = []

    for m in ms:
        filename = f"spectrumVH2O_M={int(m*10)}.txt"
        filenames.append(filename)
        yaml_dict = OrderedDict({
                "RUN SETTINGS": OrderedDict({
                    "wavelength_range": [3.5, 19.5],
                    "output_folder": f'{directory}/results/{filename}',
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

        with open(f'{directory}/configs/{filename[:-4]}.yaml', "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)

    for filename in filenames:
        inputs = shlex.split(f"python /home/ipa/quanz/user_accounts/tfessler/software/pyRetLIFE/scripts/create_spectrum.py" +
                             f' --config {directory}/configs/{filename[:-4]}.yaml')

        process = subprocess.Popen(inputs, env=os.environ)
        process.wait()
        process.terminate()
        print("\n" + "=" * 60)
        print(f"FILE '{filename}' DONE at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 60 + "\n")

