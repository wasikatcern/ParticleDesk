"""
Utility modules for particle physics data analysis.
"""

from .data_fetcher import fetch_data_from_url, download_cern_dataset, load_csv_data
from .root_parser import (
    open_root_file, 
    list_trees, 
    get_tree_branches,
    read_tree_to_dataframe,
    read_histogram, # <-- Added this import
    get_root_file_info
)
from .physics_utils import (
    calculate_invariant_mass_4lepton,
    calculate_invariant_mass_2particle,
    calculate_transverse_mass,
    delta_r,
    apply_lepton_cuts,
    get_particle_info,
    MUON_MASS,
    ELECTRON_MASS,
    HIGGS_MASS_EXPECTED
)

__all__ = [
    'fetch_data_from_url',
    'download_cern_dataset',
    'load_csv_data',
    'open_root_file',
    'list_trees',
    'get_tree_branches',
    'read_tree_to_dataframe',
    'read_histogram', # <-- Added this export
    'get_root_file_info',
    'calculate_invariant_mass_4lepton',
    'calculate_invariant_mass_2particle',
    'calculate_transverse_mass',
    'delta_r',
    'apply_lepton_cuts',
    'get_particle_info',
    'MUON_MASS',
    'ELECTRON_MASS',
    'HIGGS_MASS_EXPECTED'
]
