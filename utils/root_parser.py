"""
Utility module for parsing ROOT files from particle physics experiments.
Uses uproot library to read ROOT files without requiring ROOT installation.
"""

import uproot
import pandas as pd
import awkward as ak
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
        raise Exception(f"Failed to get branches: {str(e)}")


def read_tree_to_dataframe(root_file, tree_name: str, 
                          branches: Optional[List[str]] = None,
                          max_entries: Optional[int] = None) -> pd.DataFrame:
    """
    Read a TTree from ROOT file into pandas DataFrame.
    
    Args:
        root_file: uproot file object
        tree_name: Name of the tree to read
        branches: List of branches to read (None = all)
        max_entries: Maximum number of entries to read
        
    Returns:
        DataFrame with the tree data
    """
    try:
        tree = root_file[tree_name]
        
        # Get data as awkward arrays
        if branches:
            arrays = tree.arrays(branches, entry_stop=max_entries, library="ak")
        else:
            arrays = tree.arrays(entry_stop=max_entries, library="ak")
        
        # Convert to pandas, handling nested arrays
        df = ak.to_dataframe(arrays)
        
        return df
    except Exception as e:
        raise Exception(f"Failed to read tree to DataFrame: {str(e)}")


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
