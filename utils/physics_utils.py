"""
Utility module for particle physics calculations and analyses.
Includes invariant mass calculations, kinematic variables, and physics constants.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from particle import Particle


# Physics constants
SPEED_OF_LIGHT = 299792458  # m/s
ELECTRON_MASS = 0.000511  # GeV/c^2
MUON_MASS = 0.105658  # GeV/c^2
HIGGS_MASS_EXPECTED = 125.0  # GeV/c^2


def calculate_invariant_mass_4lepton(df: pd.DataFrame, 
                                     pt_cols: List[str],
                                     eta_cols: List[str],
                                     phi_cols: List[str],
                                     mass_cols: List[str]) -> np.ndarray:
    """
    Calculate invariant mass of 4-lepton system.
    
    Args:
        df: DataFrame with event data
        pt_cols: Column names for transverse momentum
        eta_cols: Column names for pseudorapidity
        phi_cols: Column names for azimuthal angle
        mass_cols: Column names for particle masses
        
    Returns:
        Array of invariant masses
    """
    # Calculate 4-momentum components for each lepton
    px_total = 0
    py_total = 0
    pz_total = 0
    e_total = 0
    
    for pt_col, eta_col, phi_col, mass_col in zip(pt_cols, eta_cols, phi_cols, mass_cols):
        pt = df[pt_col].values
        eta = df[eta_col].values
        phi = df[phi_col].values
        mass = df[mass_col].values
        
        # Convert to Cartesian coordinates
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        p = np.sqrt(px**2 + py**2 + pz**2)
        e = np.sqrt(p**2 + mass**2)
        
        px_total += px
        py_total += py
        pz_total += pz
        e_total += e
    
    # Calculate invariant mass
    m_inv = np.sqrt(e_total**2 - px_total**2 - py_total**2 - pz_total**2)
    
    return m_inv


def calculate_invariant_mass_2particle(pt1, eta1, phi1, m1, 
                                       pt2, eta2, phi2, m2):
    """
    Calculate invariant mass of 2-particle system.
    
    Args:
        pt1, eta1, phi1, m1: Kinematic variables for particle 1
        pt2, eta2, phi2, m2: Kinematic variables for particle 2
        
    Returns:
        Invariant mass
    """
    # Particle 1
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(pt1**2 + pz1**2 + m1**2)
    
    # Particle 2
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(pt2**2 + pz2**2 + m2**2)
    
    # Total system
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2
    e_tot = e1 + e2
    
    m_inv = np.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2)
    
    return m_inv


def calculate_transverse_mass(pt, phi, met, met_phi):
    """
    Calculate transverse mass (useful for W boson reconstruction).
    
    Args:
        pt: Transverse momentum of lepton
        phi: Azimuthal angle of lepton
        met: Missing transverse energy
        met_phi: Azimuthal angle of MET
        
    Returns:
        Transverse mass
    """
    mt = np.sqrt(2 * pt * met * (1 - np.cos(phi - met_phi)))
    return mt


def delta_r(eta1, phi1, eta2, phi2):
    """
    Calculate deltaR between two particles.
    
    Args:
        eta1, phi1: Pseudorapidity and phi of particle 1
        eta2, phi2: Pseudorapidity and phi of particle 2
        
    Returns:
        DeltaR value
    """
    deta = eta1 - eta2
    dphi = phi1 - phi2
    
    # Wrap phi difference to [-pi, pi]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    
    dr = np.sqrt(deta**2 + dphi**2)
    return dr


def apply_lepton_cuts(df: pd.DataFrame, pt_min: float = 5.0, 
                     eta_max: float = 2.5) -> pd.DataFrame:
    """
    Apply standard lepton selection cuts.
    
    Args:
        df: DataFrame with lepton data
        pt_min: Minimum transverse momentum (GeV)
        eta_max: Maximum absolute pseudorapidity
        
    Returns:
        Filtered DataFrame
    """
    mask = (df['pt'] > pt_min) & (np.abs(df['eta']) < eta_max)
    return df[mask]


def get_particle_info(pdg_id: int) -> dict:
    """
    Get particle information from PDG ID.
    
    Args:
        pdg_id: PDG particle ID
        
    Returns:
        Dictionary with particle properties
    """
    try:
        particle = Particle.from_pdgid(pdg_id)
        return {
            'name': particle.name,
            'mass': particle.mass / 1000 if particle.mass else None,  # Convert MeV to GeV
            'charge': particle.charge if particle.charge else 0,
            'pdg_id': pdg_id
        }
    except:
        return {
            'name': f'Unknown (PDG {pdg_id})',
            'mass': None,
            'charge': None,
            'pdg_id': pdg_id
        }
