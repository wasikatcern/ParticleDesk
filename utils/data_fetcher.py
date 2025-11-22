"""
Utility module for fetching particle physics data from various sources.
Supports downloading from URLs, handling CERN Open Data, and local file operations.
"""

import requests
import pandas as pd
import io
import os
from typing import Optional, Tuple


def fetch_data_from_url(url: str) -> Tuple[pd.DataFrame, str]:
    """
    Fetch data from a URL. Supports CSV and direct file downloads.
    
    Args:
        url: URL to fetch data from
        
    Returns:
        Tuple of (DataFrame, file_type)
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file type from URL or content-type
        if url.endswith('.csv'):
            df = pd.read_csv(io.StringIO(response.text))
            return df, 'csv'
        elif url.endswith('.root'):
            # Save temporarily for ROOT file processing
            temp_path = '/tmp/temp_data.root'
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            return temp_path, 'root'
        else:
            # Try to parse as CSV
            try:
                df = pd.read_csv(io.StringIO(response.text))
                return df, 'csv'
            except:
                raise ValueError("Unable to determine file format. Please use CSV or ROOT files.")
                
    except Exception as e:
        raise Exception(f"Failed to fetch data from URL: {str(e)}")


def download_cern_dataset(record_id: str, filename: str) -> str:
    """
    Download a specific file from CERN Open Data portal.
    
    Args:
        record_id: CERN record ID (e.g., '5200')
        filename: Name of file to download
        
    Returns:
        Path to downloaded file
    """
    url = f"https://opendata.cern.ch/record/{record_id}/files/{filename}"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        filepath = f"data/{filename}"
        
        if filename.endswith('.csv'):
            with open(filepath, 'w') as f:
                f.write(response.text)
        else:
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
        return filepath
        
    except Exception as e:
        raise Exception(f"Failed to download from CERN: {str(e)}")


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data file into pandas DataFrame.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with the data
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise Exception(f"Failed to load CSV file: {str(e)}")
