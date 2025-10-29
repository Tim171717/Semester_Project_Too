import os
import shlex,subprocess
import datetime

from helper_functions import *


def get_cli_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    return args


def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())
yaml.add_representer(OrderedDict, represent_ordereddict)

if __name__ == "__main__":
    args = get_cli_arguments()
    with open(str(args.config), 'r') as file:
        config_file = yaml.safe_load(file)

    runname = str(config_file['runname'])
    nproc = int(config_file['nproc'])
    home_directory = str(config_file["home_directory"])
    directory = home_directory + '/' + runname
    mass_distros = config_file['mass_distros']
    m_truth = str(config_file['truth'])
    spectrum = str(config_file['spectrum'])
    add_options = config_file['add_options']
    do_comparison = config_file['do_comparison'] if 'do_comparison' in config_file.keys() else True

    subprocess.run(f"mkdir -p {directory}/configs")
    subprocess.run(f"mkdir -p {directory}/spectra")

    Retnames = []
    filenames = []

    for n, mdis in enumerate(mass_distros):
        yaml_dict = make_yaml_dict(runname, n + 1, mdis[0])
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

        filename = f"config{n+1}_{mdis[0]}.yaml"
        filenames.append(filename)

        with open(f"{directory}/configs/{filename}", "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)

        subprocess.run(f"mkdir -p {directory}/results/Retrieval{n+1}_{mdis[0]}")
        Retnames.append(f'Retrieval{n+1}_{mdis[0]}')

    for filename, Retname in zip(filenames, Retnames):
        inputs = shlex.split(
            f"nohup nice -n 19 python {home_directory}/software/pyRetLIFE/scripts/run_plotting.py " +
            f"--config {directory}/configs/{filename} --nproc {nproc} &>> {directory}/results/{Retname}/output.txt &")
        process = subprocess.Popen(inputs, env=os.environ)
        process.wait()
        process.terminate()
        print("\n" + "=" * 60)
        print(f"Retrieval {filename}' DONE at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 60 + "\n")

    if do_comparison:
        labels = {f'Ret{n+1}': r for n, r in config_file['labels']}
        folders = {f'Ret{n+1}': f'{directory}/results/{r}' for n, r in enumerate(Retnames)}

        plot_retrievals(labels, folders, bins=60, fig_title=runname, savepath=f'{directory}/results/Retrieval_comparison.pdf')