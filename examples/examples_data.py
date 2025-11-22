"""
Curated examples from CERN Open Data and CMS analyses.
Each example includes source citation, description, and suggested prompts.
"""

EXAMPLES = {
    "higgs_4l": {
        "title": "Higgs Boson → Four Leptons (H→4ℓ) Discovery Analysis",
        "description": """
This is a simplified reimplementation of the historic CMS Higgs discovery analysis. 
The Higgs boson decays into four leptons (electrons or muons) through intermediate Z bosons: H → ZZ* → 4ℓ.
The analysis searches for a peak in the four-lepton invariant mass distribution around 125 GeV.
        """,
        "source": "https://opendata.cern.ch/record/5500",
        "publication": "Phys.Lett. B716 (2012) 30-61, arXiv:1207.7235",
        "dataset_url": "https://opendata.cern.ch/record/5200",
        "github": "https://github.com/cms-opendata-analyses/HiggsExample20112012",
        "data_files": [
            {"name": "4mu_2011.csv", "url": "https://opendata.cern.ch/record/5200/files/4mu_2011.csv"},
            {"name": "4mu_2012.csv", "url": "https://opendata.cern.ch/record/5200/files/4mu_2012.csv"},
            {"name": "4e_2011.csv", "url": "https://opendata.cern.ch/record/5200/files/4e_2011.csv"},
            {"name": "4e_2012.csv", "url": "https://opendata.cern.ch/record/5200/files/4e_2012.csv"},
            {"name": "2e2mu_2011.csv", "url": "https://opendata.cern.ch/record/5200/files/2e2mu_2011.csv"},
            {"name": "2e2mu_2012.csv", "url": "https://opendata.cern.ch/record/5200/files/2e2mu_2012.csv"}
        ],
        "suggested_prompts": [
            "Plot the four-lepton invariant mass distribution",
            "Show histogram of M4l between 70 and 200 GeV",
            "Calculate and display the invariant mass peak",
            "Plot the Higgs mass peak around 125 GeV",
            "Show the distribution of all four-lepton events"
        ],
        "columns_info": {
            "Run": "Run number",
            "Event": "Event number",
            "E1": "Energy of lepton 1 (GeV)",
            "px1": "x-component of momentum for lepton 1 (GeV/c)",
            "py1": "y-component of momentum for lepton 1 (GeV/c)",
            "pz1": "z-component of momentum for lepton 1 (GeV/c)",
            "pt1": "Transverse momentum of lepton 1 (GeV/c)",
            "eta1": "Pseudorapidity of lepton 1",
            "phi1": "Azimuthal angle of lepton 1 (radians)",
            "Q1": "Charge of lepton 1",
            "M": "Invariant mass of four-lepton system (GeV/c²)"
        }
    },
    
    "dimuon_spectrum": {
        "title": "Di-Muon Mass Spectrum Analysis",
        "description": """
Analysis of the invariant mass spectrum of muon pairs. This spectrum reveals several 
important physics resonances including J/ψ (3.1 GeV), ϒ (upsilon, 9.5 GeV), and Z boson (91 GeV).
This is a fundamental analysis in particle physics demonstrating particle identification.
        """,
        "source": "https://opendata.cern.ch/record/545",
        "github": "https://github.com/cms-opendata-analyses/DimuonSpectrumNanoAODOutreachAnalysis",
        "dataset_url": "https://opendata.cern.ch/record/545",
        "data_files": [],  # Uses NanoAOD format, more complex
        "suggested_prompts": [
            "Plot the di-muon invariant mass spectrum",
            "Show resonance peaks in the muon pair mass distribution",
            "Highlight the Z boson peak around 91 GeV",
            "Display J/psi and Upsilon resonances",
            "Plot muon momentum distributions"
        ],
        "columns_info": {
            "M": "Invariant mass of muon pair (GeV/c²)",
            "pt": "Transverse momentum (GeV/c)",
            "eta": "Pseudorapidity",
            "phi": "Azimuthal angle (radians)"
        }
    },
    
    "higgs_tau_tau": {
        "title": "Higgs → τ⁺τ⁻ (Tau Pair) Decay Analysis",
        "description": """
Study of Higgs boson decay into tau lepton pairs. The tau leptons are identified through 
their decay products. This channel provided crucial evidence for the Higgs boson coupling 
to fermions and helped confirm the Standard Model predictions.
        """,
        "source": "https://opendata.cern.ch/record/12350",
        "github": "https://github.com/cms-opendata-analyses/HiggsTauTauNanoAODOutreachAnalysis",
        "publication": "Phys.Lett. B842 (2023) 137531",
        "data_files": [],
        "suggested_prompts": [
            "Plot the di-tau invariant mass",
            "Show tau decay product distributions",
            "Display missing transverse energy",
            "Analyze tau identification variables",
            "Plot the Higgs signal in tau channel"
        ],
        "columns_info": {
            "m_vis": "Visible mass of tau pair (GeV/c²)",
            "pt_1": "Leading tau transverse momentum (GeV/c)",
            "pt_2": "Subleading tau transverse momentum (GeV/c)",
            "met": "Missing transverse energy (GeV)"
        }
    },
    
    "top_quark": {
        "title": "Top Quark Pair Production",
        "description": """
Analysis of top quark pair production events. Top quarks decay almost exclusively to 
a W boson and a b-quark. This analysis looks at events where both W bosons decay leptonically,
resulting in two leptons, two b-jets, and missing energy from neutrinos.
        """,
        "source": "https://opendata.cern.ch/",
        "data_files": [],
        "suggested_prompts": [
            "Plot the top quark mass distribution",
            "Show lepton transverse momentum",
            "Display b-jet multiplicity",
            "Analyze missing transverse energy",
            "Plot invariant mass of lepton + b-jet system"
        ],
        "columns_info": {
            "m_tt": "Invariant mass of top pair (GeV/c²)",
            "n_bjets": "Number of b-tagged jets",
            "met": "Missing transverse energy (GeV)"
        }
    },
    
    "z_boson": {
        "title": "Z Boson Production and Decay",
        "description": """
Study of Z boson production and decay into lepton pairs (electrons or muons).
The Z boson has a well-known mass of ~91 GeV and provides an excellent calibration 
point for detector performance and serves as a Standard Model benchmark.
        """,
        "source": "https://opendata.cern.ch/",
        "data_files": [],
        "suggested_prompts": [
            "Plot Z boson mass peak",
            "Show dilepton invariant mass around 91 GeV",
            "Display lepton kinematics for Z decays",
            "Plot Z transverse momentum",
            "Show rapidity distribution of Z bosons"
        ],
        "columns_info": {
            "M_ll": "Dilepton invariant mass (GeV/c²)",
            "pt_Z": "Z boson transverse momentum (GeV/c)",
            "y_Z": "Z boson rapidity"
        }
    }
}


def get_example(example_id: str) -> dict:
    """Get example information by ID."""
    return EXAMPLES.get(example_id, {})


def get_all_examples() -> dict:
    """Get all available examples."""
    return EXAMPLES


def get_example_ids() -> list:
    """Get list of all example IDs."""
    return list(EXAMPLES.keys())


def get_example_titles() -> dict:
    """Get mapping of example IDs to titles."""
    return {key: val["title"] for key, val in EXAMPLES.items()}
