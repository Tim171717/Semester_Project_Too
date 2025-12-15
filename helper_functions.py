from collections import OrderedDict
from pyretlife.retrieval_plotting.run_plotting import retrieval_plotting_object
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from argparse import ArgumentParser, Namespace
import yaml
import glob
import shutil
import os

from pyretlife.retrieval_plotting.posterior_plotting import (
    Generate_Parameter_Titles,
    Scale_Posteriors,
)



def make_yaml_dict(
        runname,
        num,
        distribution_name,
        spectrum_path='/home/ipa/quanz/user_accounts/tfessler/Retrieval_Tutorial_pRT3/spectra/VH2O_test_spectrum.txt',
        study_folder='/home/ipa/quanz/user_accounts/tfessler/Retrieval_studies'
):
    """
        Returns an OrderedDict object with the configuration to set up a config file for a pyRetLife retrieval.

        Parameters
        ----------
        runname :  str
            The name of the retrieval study run.

        num :  int
            Number of the config file created

        distribution_name :  str
            The distribution used in

        spectrum_path :  str, optional
            The path to the spectra used in the retrievals.
            Default set to '/home/ipa/quanz/user_accounts/tfessler/Retrieval_Tutorial_pRT3/spectra/VH2O_test_spectrum.txt'.

        study_folder :  str, optional
            The path to the home directory of the retrieval comparison. Default set to '/home/ipa/quanz/user_accounts/tfessler/Retrieval_studies'.
        """
    yaml_dict = OrderedDict({
        'GROUND TRUTH DATA': OrderedDict({
            'data_files': OrderedDict({
                'data': OrderedDict({
                    'path': spectrum_path,
                    'unit': 'micron, erg s-1 Hz-1 m-2'
                })
            })
        }),
        'RUN SETTINGS': OrderedDict({
            'wavelength_range': [3.5, 19.0],
            'output_folder': f'{study_folder}/{runname}/results/Retrieval{num}_{distribution_name}/',
            'include_scattering': OrderedDict({
                'Rayleigh': False,
                'thermal': False,
                'direct_light': False,
                'clouds': False
            }),
            'include_CIA': True,
            'resolution': 100,
            'n_layers': 100,
            'log_top_pressure': -4
        }),
        'TEMPERATURE PARAMETERS': OrderedDict({
            'parameterization': 'polynomial',
            'a_4': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'uniform',
                    'prior_specs': OrderedDict({
                        'lower': 0,
                        'upper': 10
                    })
                }),
                'truth': 1.67
            }),
            'a_3': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'uniform',
                    'prior_specs': OrderedDict({
                        'lower': 0.0,
                        'upper': 100.0
                    })
                }),
                'truth': 23.12
            }),
            'a_2': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'uniform',
                    'prior_specs': OrderedDict({
                        'lower': 0.0,
                        'upper': 500.0
                    })
                }),
                'truth': 99.7
            }),
            'a_1': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'uniform',
                    'prior_specs': OrderedDict({
                        'lower': 0.0,
                        'upper': 1000.0
                    })
                }),
                'truth': 146.63
            }),
            'a_0': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'uniform',
                    'prior_specs': OrderedDict({
                        'lower': 0.0,
                        'upper': 500.0
                    })
                }),
                'truth': 285.22
            })
        }),
        'PHYSICAL PARAMETERS': OrderedDict({
            'P0': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -4,
                        'log_upper': 2
                    })
                }),
                'truth': 1.02329299228
            }),
            'd_syst': OrderedDict({
                'truth': 10.0
            }),
            'R_pl': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'gaussian',
                    'prior_specs': OrderedDict({
                        'mean': 1,
                        'sigma': 0.2
                    })
                }),
                'truth': 1.0
            }),
            'M_pl': OrderedDict({}),
        }),
        'CHEMICAL COMPOSITION PARAMETERS': OrderedDict({
            'mmw_inert': 28.9,
            'condensation': True,
            'N2': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -10,
                        'log_upper': 0
                    })
                }),
                'truth': 0.79
            }),
            'O2': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -10,
                        'log_upper': 0
                    })
                }),
                'truth': 0.2
            }),
            'CO2': OrderedDict({
                'lines': ['CO2'],
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -10,
                        'log_upper': 0
                    })
                }),
                'truth': 0.00041
            }),
            'H2O': OrderedDict({
                'lines': ['H2O'],
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -10,
                        'log_upper': -1
                    })
                }),
                'truth': 0.001
            }),
            'H2O_Drying': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -5,
                        'log_upper': 0
                    })
                }),
                'truth': 0.01
            }),
            'O3': OrderedDict({
                'lines': ['O3'],
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -10,
                        'log_upper': 0
                    })
                }),
                'truth': 3e-07
            }),
            'CH4': OrderedDict({
                'lines': ['CH4'],
                'prior': OrderedDict({
                    'kind': 'log-uniform',
                    'prior_specs': OrderedDict({
                        'log_lower': -10,
                        'log_upper': 0
                    })
                }),
                'truth': 1.7e-06
            })
        }),
        'CLOUD PARAMETERS': OrderedDict({
            'settings_clouds': 'patchy',
            'cloud_fraction': OrderedDict({
                'prior': OrderedDict({
                    'kind': 'uniform',
                    'prior_specs': OrderedDict({
                        'lower': 0.0,
                        'upper': 1.0
                    })
                }),
                'truth': 0.67
            })
        })
    })
    return yaml_dict


def copy_spectrum(directory):
    matches = [
        f for f in glob.glob(os.path.join(directory, "input_*.txt"))
        if os.path.basename(f) != "input_spectrum.txt"]
    if matches:
        source = matches[0]
        destination = os.path.join(directory, "input_spectrum.txt")
        shutil.copy(source, destination)


def load_data(
        self,
        log_pressures=True,
        log_mass=True,
        log_abundances=True,
        log_particle_radii=True,
        plot_pt=True,
        plot_physparam=True,
        plot_clouds=True,
        plot_chemcomp=True,
        plot_scatt=True,
        plot_moon=False,
        plot_secondary_parameters=True,
        parameter_units='input',
        custom_unit_titles={},
        custom_parameter_titles={}
):
    """
    Loads and returns the data for the retrieved parameters as well as the upper-log-uniform (ULU), the truth values and the parameter names.

    Parameters
    ----------
    log_pressures :  bool, optional
        If True, applies a logarithmic scale to pressure values. Defaults to True.

    log_mass : bool, optional
        If True, applies a logarithmic scale to mass values. Defaults to True.

    log_abundances :  bool, optional
        If True, applies a logarithmic scale to abundance values. Defaults to True.

    log_particle_radii : bool, optional
        If True, applies a logarithmic scale to particle radii values. Defaults to True.

    plot_pt :  bool, optional
        If True, plots pressure-temperature parameters. Defaults to True.

    plot_physparam : bool, optional
        If True, plots physical parameters. Defaults to True.

    plot_clouds :  bool, optional
        If True, plots cloud parameters. Defaults to True.

    plot_chemcomp : bool, optional
        If True, plots chemical composition parameters. Defaults to True.

    plot_scatt : bool, optional
        If True, plots scattering parameters. Defaults to True.

    plot_moon :  bool, optional
        If True, plots moon parameters. Defaults to False.

    plot_secondary_parameters : bool, optional
        If True, plots secondary parameters. Defaults to True.

    parameter_units : str, optional
        The units to use for plotting. Defaults to 'input', which are the units specified in the config file.

    custom_unit_titles : dict, optional
        A dictionary mapping parameters to custom unit titles. Defaults to an empty dictionary.

    custom_parameter_titles={} : dict, optional
        A dictionary mapping parameters to custom titles. Defaults to an empty dictionary.
    """
    recompute = False
    self.calculate_posterior_spectrum(reevaluate_spectra=False)
    if np.shape(self.retrieved_fluxes)[0] != np.shape(self.posteriors)[0]:
        delattr(self, 'retrieved_fluxes')
        self.calculate_posterior_spectrum(reevaluate_spectra=True)
        recompute = True
        
    try:
        self.calculate_posterior_pt_profile(n_processes=4,reevaluate_PT=recompute)
            
        self.deduce_bond_albedo(stellar_luminosity=1.0,
                            	error_stellar_luminosity=0.01,
                            	planet_star_separation=1.0,
                        		error_planet_star_separation=0.01,
                        		true_equilibrium_temperature = 255,
                        		true_bond_albedo = 0.29,
                        		reevaluate_bond_albedo=recompute)
        self.deduce_abundance_profiles(reevaluate_abundance_profiles=recompute)
        
        self.deduce_gravity(true_gravity = 981)
        self.deduce_surface_temperature(true_surface_temperature = 273)
            
    except Exception as e:
        print(f"Error correcting data for {self.results_directory}: {e}")
        return None, None, None, None

    parameters_plotted = []
    for parameter in self.parameters:
        if (self.parameters[parameter]['type'] == 'TEMPERATURE PARAMETERS') and plot_pt:
            parameters_plotted += [parameter]
        elif (self.parameters[parameter]['type'] == 'PHYSICAL PARAMETERS') and plot_physparam:
            parameters_plotted += [parameter]
        elif (self.parameters[parameter]['type'] == 'CHEMICAL COMPOSITION PARAMETERS') and plot_chemcomp:
            parameters_plotted += [parameter]
        elif (self.parameters[parameter]['type'] == 'CLOUD PARAMETERS') and plot_clouds:
            parameters_plotted += [parameter]
        elif (self.parameters[parameter]['type'] == 'SCATTERING PARAMETERS') and plot_scatt:
            parameters_plotted += [parameter]
        elif (self.parameters[parameter]['type'] == 'SECONDARY PARAMETERS') and plot_secondary_parameters:
            parameters_plotted += [parameter]
        elif (self.parameters[parameter]['type'] == 'MOON PARAMETERS') and plot_moon:
            parameters_plotted += [parameter]

    local_post = self.posteriors.copy()
    local_truths = {parameter:self.parameters[parameter]['truth'] for parameter in parameters_plotted}

    Generate_Parameter_Titles(self)
    local_titles = {i:self.parameters[i]['title'] for i in parameters_plotted}
    for parameter in parameters_plotted:
        if parameter in custom_parameter_titles:
            local_titles[parameter] = custom_parameter_titles[parameter]

    retrieval_unit =  {i:self.parameters[i]['unit'] for i in parameters_plotted}
    if parameter_units == 'input':
        local_units = {i:self.parameters[i]['input_unit'] for i in parameters_plotted}
    else:
        local_units = retrieval_unit.copy()
        for parameter in parameters_plotted:
            if parameter in parameter_units:
                local_units[parameter] = parameter_units[parameter]

    for parameter in parameters_plotted:
        if not f"{local_units[parameter]:latex}" == '$\\mathrm{}$':
            unit = '\\left['+f"{local_units[parameter]:latex}"[1:-1]+'\\right]'
        else:
            unit = ''
        if parameter in custom_unit_titles:
            unit = '\\left['+custom_unit_titles[parameter][1:-1]+'\\right]'
        local_titles[parameter] = local_titles[parameter][:-1]+unit+'$'

    for parameter in parameters_plotted:
        local_post[parameter]   = self.units.truth_unit_conversion(parameter,retrieval_unit[parameter],local_units[parameter],local_post[parameter].to_numpy(),printing=False)
        local_truths[parameter] = self.units.truth_unit_conversion(parameter,retrieval_unit[parameter],local_units[parameter],local_truths[parameter],printing=False)

    local_post, local_truths, local_titles = Scale_Posteriors(self,local_post, local_truths, local_titles, parameters_plotted,
                                                              log_pressures=log_pressures, log_mass=log_mass,
                                                              log_abundances=log_abundances, log_particle_radii=log_particle_radii)

    ULU = [parameter for parameter in parameters_plotted if self.parameters[parameter]['prior']['kind'] == 'upper-log-uniform']

    dataset = {}
    for param in parameters_plotted:
        dataset[param] = local_post[param]

    return dataset, ULU, local_truths, parameters_plotted


def plot_retrievals(
        labels,
        folders,
        colors=None,
        params_to_plot=None,
        bins=50,
        fig_title=None,
        savepath=None,
        ULU_lim=[-0.15,0.75]
):
    """
    Plot the histograms of the fitted parameters for a series of retrievals.

    Parameters
    ----------
    labels : dict
        Dictionary mapping run names to legend labels.

    folders : dict
        Dictionary mapping run names to folder with the run data.

    colors : dict, optional
        Dictionary mapping run names to histogram colors. Default is a generic color palette with 10 unique options.

    params_to_plot : list, optional
        List of parameter names to plot. Default is all.

    bins : int or sequence, optional
        Number of histogram bins (default 50).

    fig_title : str, optional
        Title for the whole figure.

    savepath : str, optional
        Saves the figure at the given filepath. When no path is provided the figure is not saved.

    ULU_lim : list, optional
        Limits for the ULU correction (lower bound and smoothing factor). Default is [-0.15, 0.75].
    """
    retrieval_plotting_object.load_data = load_data

    datasets = {}
    ULUs = {}
    local_truths = []
    params = []
    for label in labels.keys():
        results = retrieval_plotting_object(folders[label])
        ds, ul, lt, pa = results.load_data()
        if ds is not None:
            datasets[label], ULUs[label], local_truths, params = ds, ul, lt, pa
	
    if not datasets:
        print("No datasets could be loaded.")
        return
    
    n_params = len(params)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3 * n_rows),
        constrained_layout=False
    )
    axs = axs.flatten()

    if colors is None:
        default_colors = [
            "mediumpurple", "indianred", "goldenrod", "steelblue", "darkorange", "seagreen", "firebrick", "royalblue", "darkkhaki"
        ]
        colors = {label: default_colors[i % len(default_colors)] for i, label in enumerate(datasets.keys())}

    for i, param in enumerate(params):
        if params_to_plot is not None and param not in params_to_plot:
            continue

        ax = axs[i]

        for run_name, data in datasets.items():
            if param in data.keys():
                if ULUs is not None and param in ULUs[run_name]:
                    h = np.histogram(data[param],density=True,bins=bins,range = (ULU_lim[0],0))
                    h2 = np.histogram(np.log10(1-10**(np.arange(-7,0,0.000001))),density=True,bins=h[1])
                    h = (h[0]/h2[0],h[1])
                    h = ax.hist(
                        h[1][: -1],
                        h[1],
                        weights = sp.ndimage.filters.gaussian_filter(h[0], [ULU_lim[1]], mode='constant'),
                        histtype='stepfilled',
                        color=colors.get(run_name, 'gray'),
                        density=True,
                        label=labels.get(run_name, run_name)
                    )
                else:
                    h = ax.hist(
                        data[param],
                        histtype='stepfilled',
                        color=colors.get(run_name, 'gray'),
                        alpha=0.6,
                        density=True,
                        bins=bins,
                        label=labels.get(run_name, run_name)
                    )

        ax.set_title(param, fontsize=12)
        ax.set_ylabel("Prob. density")
        ax.tick_params(labelsize=10)
        ax.grid(False)

        ax.axvline(local_truths[param], color='k', linestyle='--', linewidth=1.5)

    # Hide unused subplots
    for j in range(len(params), len(axs)):
        axs[j].axis('off')

    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.98, hspace=0.35, wspace=0.25)

    if fig_title:
        fig.suptitle(fig_title, fontsize=16, weight='bold', y=0.965)

    # Create a single legend above all plots
    handles, legend_labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(datasets),
        fontsize=11,
        frameon=False
    )
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()


def plot_comparison_intervals(
        labels,
        folders,
        colors=None,
        params_to_plot=None,
        figsize=(16, 16),
        n_cols=4,
        quantiles=(0.16, 0.5, 0.84),
        truth_band_width=0,
        fig_title=None,
        savepath=None,
):
    """
    Create stacked horizontal-interval comparison plots for multiple retrievals.

    Parameters
    ----------
    labels : dict
        Dictionary mapping run names to legend labels.

    folders : dict
        Dictionary mapping run names to folder with the run data.

    colors : dict, optional
        Dictionary mapping run names to histogram colors. Default is a generic color palette with 10 unique options.

    params_to_plot : list, optional
        List of parameter names to plot. Default is all.

    figsize : tuple, optional
        Size of figure in inches. Default is (16, 16).

    n_cols : int, optional
        Number of subplots in each row. Default is 4.

    quantiles : tuple, optional
        q value for which np.quantile computes the q-th percentile of the data along the specified axis.

    truth_band_width : float, optional
        Width of band to use for comparison. Default is no band.

    fig_title : str, optional
        Title for the whole figure.

    savepath : str, optional
        Saves the figure at the given filepath. When no path is provided the figure is not saved.
    """

    retrieval_plotting_object.load_data = load_data

    datasets = {}
    ULUs = {}
    local_truths = []
    params = []
    for label in labels.keys():
        results = retrieval_plotting_object(folders[label])
        ds, ul, lt, pa = results.load_data()
        if ds is not None:
            datasets[label], ULUs[label], local_truths, params = ds, ul, lt, pa

    n_params = len(params)

    if colors is None:
        default = ["indianred", "purple", "seagreen", "steelblue", "darkorange"]
        colors = {lab: default[i % len(default)] for i, lab in enumerate(labels.keys())}

    n_rows = int(np.ceil(n_params / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        constrained_layout=False
    )
    axs = axs.flatten()

    for i, param in enumerate(params):
        if params_to_plot is not None and param not in params_to_plot:
            continue

        ax = axs[i]

        t = local_truths[param]
        ax.axvspan(t * (1 - truth_band_width), t * (1 + truth_band_width),
                   color="lightgray", alpha=0.5, zorder=0)

        ax.axvline(x=t, color='gray', linestyle='--', linewidth=1.5, zorder=0)

        y_positions = np.arange(len(labels))

        for j, run in enumerate(labels.keys()):

            if param not in datasets[run]:
                continue

            samples = np.asarray(datasets[run][param])
            q_low, q_med, q_high = np.quantile(samples, quantiles)

            ax.hlines(
                y_positions[j],
                q_low, q_high,
                color=colors[run],
                linewidth=2
            )

            ax.plot(q_low,  y_positions[j], marker='|', color=colors[run], ms=8)
            ax.plot(q_high, y_positions[j], marker='|', color=colors[run], ms=8)

            ax.plot(q_med,  y_positions[j], marker='o', color=colors[run], ms=6)

        if i % n_cols == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels.values(), fontsize=11)
        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
        ax.set_ylim(-0.25, len(labels) - 0.75)
        ax.invert_yaxis()
        ax.set_title(param, fontsize=12, weight="bold", pad=3.5)
        ax.grid(False)

    for k in range(len(params), len(axs)):
        axs[k].axis("off")

    if fig_title:
        fig.suptitle(fig_title, fontsize=20, weight='bold', y=0.92)

    fig.subplots_adjust(hspace=0.3, wspace=0.01)

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()


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
    args = get_cli_arguments()
    with open(str(args.config), 'r') as file:
        config_file = yaml.safe_load(file)

    labels = config_file['labels']
    colors = config_file['colors'] if 'colors' in config_file.keys() else None
    folders = config_file['folders']
    fig_title = str(config_file['title']) if 'title' in config_file.keys() else None
    savepath = str(config_file['savepath']) if 'savepath' in config_file.keys() else None
    new_plot = config_file['new_plot'] if 'new_plot' in config_file.keys() else None

    if new_plot:
        plot_comparison_intervals(labels, folders, colors=colors, fig_title=fig_title, savepath=savepath)
    else:
        plot_retrievals(labels, folders, colors=colors, bins=60, fig_title=fig_title, savepath=savepath)