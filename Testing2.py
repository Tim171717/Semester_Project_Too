import numpy as np
import matplotlib.pyplot as plt
import paretlife ################



path = 'path/to/files'
results = retrieval_plotting_object(results_directory = path)
# Calculates and saves the PT profiles and spectra for plotting and 
results.calculate_posterior_pt_profile(n_processes=min([200,int(args.nproc)]),reevaluate_PT=False)

results.calculate_posterior_spectrum(n_processes=min([200,int(args.nproc)]),reevaluate_spectra=False)

results.deduce_bond_albedo(stellar_luminosity=1.0,
										error_stellar_luminosity=0.01,
                                        planet_star_separation=1.0,
                                        error_planet_star_separation=0.01,
                                        true_equilibrium_temperature = 255,
                                        true_bond_albedo = 0.29,
                                        reevaluate_bond_albedo=False)
results.deduce_abundance_profiles(reevaluate_abundance_profiles=False)
    
results.deduce_gravity(true_gravity = 981)
results.deduce_surface_temperature(true_surface_temperature = 273)

unit_titles = {'R_pl':'$\mathrm{R_{Earth}}$','M_pl':'$\mathrm{M_{Earth}}$'}





# get the indices of all parameters shown in the corner plot
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

# Copy the relevant data
local_post = self.posteriors.copy()
local_truths = {parameter:self.parameters[parameter]['truth'] for parameter in parameters_plotted}

# Generate the titles
Generate_Parameter_Titles(self)
local_titles = {i:self.parameters[i]['title'] for i in parameters_plotted}
for parameter in parameters_plotted:
    if parameter in custom_parameter_titles:
        local_titles[parameter] = custom_parameter_titles[parameter]

# Unit conversions for plotting if units=None retrieval units are plotted
# if units='input' the units in the input.ini file are plotted
retrieval_unit =  {i:self.parameters[i]['unit'] for i in parameters_plotted}
if parameter_units == 'input':
    local_units = {i:self.parameters[i]['input_unit'] for i in parameters_plotted}
else:
    local_units = retrieval_unit.copy()
    for parameter in parameters_plotted:
        if parameter in parameter_units:
            local_units[parameter] = parameter_units[parameter]

# Add the units to the titles
for parameter in parameters_plotted:
    if not f"{local_units[parameter]:latex}" == '$\\mathrm{}$':
        unit = '\\left['+f"{local_units[parameter]:latex}"[1:-1]+'\\right]'
    else:
        unit = ''
    if parameter in custom_unit_titles:
        unit = '\\left['+custom_unit_titles[parameter][1:-1]+'\\right]'
    local_titles[parameter] = local_titles[parameter][:-1]+unit+'$'

# Convert the units of the posterior and the true value
for parameter in parameters_plotted:
    local_post[parameter]   = self.units.truth_unit_conversion(parameter,retrieval_unit[parameter],local_units[parameter],local_post[parameter].to_numpy(),printing=False)
    local_truths[parameter] = self.units.truth_unit_conversion(parameter,retrieval_unit[parameter],local_units[parameter],local_truths[parameter],printing=False)

# Adust the local copy of the posteriors according to the users desires
local_post, local_truths, local_titles = Scale_Posteriors(self,local_post, local_truths, local_titles, parameters_plotted,
                                                          log_pressures=log_pressures, log_mass=log_mass,
                                                          log_abundances=log_abundances, log_particle_radii=log_particle_radii)

# Check if there were ULU posteriors
ULU = [parameter for parameter in parameters_plotted if self.parameters[parameter]['prior']['kind'] == 'upper-log-uniform']

if plot_corner:
    fig, axs = Corner_Plot(parameters_plotted,local_post,local_titles,local_truths,quantiles1d=quantiles1d,bins=bins,color=color,
                                    add_table=add_table,color_truth=color_truth,ULU=ULU if ULU != [] else None,ULU_lim=ULU_lim)
    if save:
        plt.savefig(self.results_directory+'Plots_New/plot_corner.pdf', bbox_inches='tight')
    else:
        return fig, axs
else:
    if not os.path.exists(self.results_directory + 'Plots_New/Posteriors/'):
        os.makedirs(self.results_directory + 'Plots_New/Posteriors/')
    for parameter in parameters_plotted:
        fig, axs = Posterior_Plot(local_post[parameter],local_titles[parameter],local_truths[parameter],
                            quantiles1d=quantiles1d,bins=bins,color=color,ULU=(parameter in ULU),ULU_lim=ULU_lim)

        if save:
            plt.savefig(self.results_directory+'/Plots_New/Posteriors/'+parameter+'.pdf', bbox_inches='tight')
        else:
            return fig, axs


import matplotlib.pyplot as plt
import numpy as np

def plot_retrieval_comparison(datasets, params, labels, colors, bins=50, fig_title=None):
    """
    Plot histograms of multiple retrieval runs for comparison.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping run labels to dict-like datasets, e.g.
        {
            "Perfect Mp": data1,
            "Unknown Mp": data2,
            "Gaussian 30%": data3
        }
        Each dataX should support access like dataX[param] (e.g. pandas DataFrame, dict, etc.)

    params : list of str
        List of parameter names to plot, e.g. ['Log(P0)', 'Log(H2O)', 'Mp', 'Rp'].

    labels : dict
        Dictionary mapping run names (keys in datasets) to legend labels.

    colors : dict
        Dictionary mapping run names to histogram colors.

    bins : int or sequence, optional
        Number of histogram bins (default 50).

    fig_title : str, optional
        Title for the whole figure.
    """
    n_params = len(params)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows), constrained_layout=True)
    axs = axs.flatten()

    for i, param in enumerate(params):
        ax = axs[i]

        for run_name, data in datasets.items():
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

        # Optional: vertical line (e.g., true value)
        # ax.axvline(true_values[param], color='k', linestyle='--', linewidth=1)

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    if fig_title:
        fig.suptitle(fig_title, fontsize=16, weight='bold')

    # Create a single legend above all plots
    handles, legend_labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=len(datasets), fontsize=11, frameon=False)

    plt.show()
