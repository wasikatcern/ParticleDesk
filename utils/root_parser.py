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


def read_tree_to_dataframe(root_file_path: str, tree_name: str, max_entries: Optional[int] = None) -> pd.DataFrame:
    """
    Read a TTree from a ROOT file into a pandas DataFrame, flattening awkward arrays.
    
    Args:
        root_file_path: Path to ROOT file
        tree_name: Name of the TTree
        max_entries: Maximum number of entries to read
        
    Returns:
        DataFrame with the tree data
    """
    try:
        with uproot.open(root_file_path) as root_file:
            tree = root_file[tree_name]
            
            # Read branches, excluding variable-length arrays that cannot be flattened easily
            # We explicitly read only the flat branches to ensure compatibility with pandas/AI analysis
            branches_to_read = [name for name, branch in tree.items() if not branch.is_array]
            
            if max_entries is None:
                arrays = tree.arrays(branches_to_read, library="pd")
                # Using library="pd" directly avoids the explicit awkward-to-pandas conversion for flat trees
            else:
                arrays = tree.arrays(branches_to_read, entry_stop=max_entries, library="pd")

            # This assumes the tree is flat. For complex trees, the user must select a TTree
            # that is suitable for flat analysis or the AI must be aware of the nested structure.
            return arrays
        
    except Exception as e:
        raise Exception(f"Failed to read tree to DataFrame: {str(e)}")

# --- NEW FUNCTION FOR HISTOGRAMS ---
def read_histogram(filepath: str, hist_name: str) -> Dict[str, Any]:
    """
    Read a TH1 or TH2 histogram from a ROOT file.
    
    Args:
        filepath: Path to ROOT file
        hist_name: Name of the histogram
        
    Returns:
        Dictionary containing histogram data for plotting/analysis.
    """
    try:
        with uproot.open(filepath) as root_file:
            hist = root_file[hist_name]
            
            if 'TH1' in hist.classname:
                values, edges = hist.to_numpy()
                return {
                    'type': 'TH1',
                    'values': values.tolist(),
                    'edges': edges.tolist(),
                    'name': hist_name,
                    'title': hist.title if hasattr(hist, 'title') and hist.title else hist_name
                }
            elif 'TH2' in hist.classname:
                values, x_edges, y_edges = hist.to_numpy()
                return {
                    'type': 'TH2',
                    'values': values.tolist(),
                    'x_edges': x_edges.tolist(),
                    'y_edges': y_edges.tolist(),
                    'name': hist_name,
                    'title': hist.title if hasattr(hist, 'title') and hist.title else hist_name
                }
            else:
                raise ValueError(f"Object '{hist_name}' is not a supported histogram type (TH1/TH2).")
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
