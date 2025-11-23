"""
Utility module for parsing ROOT files from particle physics experiments.
Uses uproot library to read ROOT files without requiring ROOT installation.
"""

import uproot
import pandas as pd
import awkward as ak
import numpy as np
from typing import List, Dict, Any, Optional

def open_root_file(filepath: str):
    """
    Open a ROOT file and return the file object.
    
    Args:
        filepath: Path to ROOT file
        
    Returns:
        uproot.ReadOnlyDirectory object
    """
    try:
        return uproot.open(filepath)
    except Exception as e:
        raise Exception(f"Failed to open ROOT file: {str(e)}")

def list_trees(root_file) -> List[str]:
    """
    List all TTrees in a ROOT file.
    
    Args:
        root_file: uproot file object
        
    Returns:
        List of tree names
    """
    try:
        trees = []
        for key in root_file.keys():
            if root_file[key].classname == 'TTree':
                trees.append(key)
        return trees
    except Exception as e:
        raise Exception(f"Failed to list trees: {str(e)}")

def get_tree_branches(root_file, tree_name: str) -> List[str]:
    """
    Get list of branches in a specific tree.
    
    Args:
        root_file: uproot file object
        tree_name: Name of the tree
        
    Returns:
        List of branch names
    """
    try:
        tree = root_file[tree_name]
        return tree.keys()
    except Exception as e:
        raise Exception(f"Failed to get branches for tree '{tree_name}': {str(e)}")

def read_tree_to_dataframe(filepath: str, tree_name: str, branches: Optional[List[str]] = None, max_entries: Optional[int] = None) -> pd.DataFrame:
    """
    Read TTree data into a pandas DataFrame.
    
    Args:
        filepath: Path to ROOT file
        tree_name: Name of the TTree
        branches: List of branches to read (if None, read all)
        max_entries: Maximum number of entries to read
        
    Returns:
        DataFrame with the tree data
    """
    try:
        with uproot.open(filepath)[tree_name] as tree:
            if branches:
                arrays = tree.arrays(branches, entry_stop=max_entries, library="ak")
            else:
                arrays = tree.arrays(entry_stop=max_entries, library="ak")
        
        # Convert to pandas, handling nested arrays
        df = ak.to_dataframe(arrays)
        
        return df
    except Exception as e:
        raise Exception(f"Failed to read tree to DataFrame: {str(e)}")

def read_histogram(filepath: str, hist_name: str) -> Dict[str, Any]:
    """
    Read a TH1 or TH2 histogram from a ROOT file.
    
    Args:
        filepath: Path to ROOT file
        hist_name: Name of the histogram
        
    Returns:
        Dictionary containing histogram data (values, edges, title, type)
    """
    try:
        with uproot.open(filepath) as root_file:
            hist = root_file[hist_name]
            
            # Check histogram class type
            classname = hist.classname
            
            if 'TH1' in classname:
                # 1D Histogram
                values, edges = hist.numpy()
                return {
                    'name': hist_name,
                    'title': hist.title if hist.title else hist_name,
                    'type': 'TH1',
                    'values': values.tolist(),
                    'edges': edges.tolist(),
                    'bin_centers': ((edges[:-1] + edges[1:]) / 2).tolist()
                }
            elif 'TH2' in classname:
                # 2D Histogram
                values, x_edges, y_edges = hist.to_numpy()
                return {
                    'name': hist_name,
                    'title': hist.title if hist.title else hist_name,
                    'type': 'TH2',
                    'values': values.tolist(),
                    'x_edges': x_edges.tolist(),
                    'y_edges': y_edges.tolist(),
                }
            else:
                raise ValueError(f"Object '{hist_name}' is of type {classname}, not a supported histogram (TH1/TH2).")

    except Exception as e:
        raise Exception(f"Failed to read histogram '{hist_name}': {str(e)}")


def get_root_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a ROOT file.
    
    Args:
        filepath: Path to ROOT file
        
    Returns:
        Dictionary with file information
    """
    try:
        root_file = uproot.open(filepath)
        
        info = {
            'filepath': filepath,
            'keys': list(root_file.keys()),
            'trees': [],
            'histograms': []
        }
        
        for key in root_file.keys():
            obj = root_file[key]
            classname = obj.classname
            
            if classname == 'TTree':
                tree_info = {
                    'name': key,
                    'num_entries': obj.num_entries,
                    'branches': obj.keys()
                }
                info['trees'].append(tree_info)
            elif 'TH1' in classname or 'TH2' in classname:
                hist_info = {
                    'name': key,
                    'type': classname
                }
                info['histograms'].append(hist_info)
        
        return info
    except Exception as e:
        raise Exception(f"Failed to get ROOT file info: {str(e)}")
