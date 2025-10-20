from collections import OrderedDict


def make_yaml_dict(runname):
    yaml_dict = OrderedDict({
        'GROUND TRUTH DATA': OrderedDict({
            'data_files': OrderedDict({
                'data': OrderedDict({
                    'path': f'/home/ipa/quanz/user_accounts/tfessler/Retrieval_studies/{runname}/spectra/input_spectrum.txt',
                    'unit': 'micron, erg s-1 Hz-1 m-2'
                })
            })
        }),
        'RUN SETTINGS': OrderedDict({
            'wavelength_range': [3.5, 19.0],
            'output_folder': f'/home/ipa/quanz/user_accounts/tfessler/Retrieval_studies/{runname}/results/',
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