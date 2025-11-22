"""
AI-powered analysis assistant using OpenAI.
Interprets natural language prompts and generates analysis code.
Reference: python_openai integration blueprint
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# This is using OpenAI's API, which points to OpenAI's API servers and requires your own API key.
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def analyze_with_ai(user_prompt: str, df: pd.DataFrame) -> dict:
    """
    Analyze particle physics data based on natural language prompt.
    Uses a safe declarative approach instead of executing arbitrary code.
    
    Args:
        user_prompt: Natural language analysis request
        df: DataFrame with particle physics data
        
    Returns:
        Dictionary with analysis results, plots, and explanations
    """
    
    if not openai_client:
        return {
            'success': False,
            'error': 'OpenAI API key not configured'
        }
    
    # Get data info
    data_info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'sample': df.head(3).to_dict('records')
    }
    
    # Create system prompt for the AI - request declarative instructions
    system_prompt = f"""You are a particle physics data analysis expert. You help users analyze particle physics data from experiments like CMS at CERN.

Available data columns: {', '.join(data_info['columns'])}
Number of events: {data_info['shape'][0]}

Common physics variables in the data:
- M, m_inv: Invariant mass (GeV/c²)
- pt, pt1, pt2: Transverse momentum (GeV/c)
- eta: Pseudorapidity
- phi: Azimuthal angle (radians)
- E: Energy (GeV)
- px, py, pz: Momentum components (GeV/c)
- Q: Electric charge

Your task is to interpret the user's request and return a declarative specification for the analysis.
Return ONLY a JSON object with these fields:

{{
    "explanation": "Brief explanation of what analysis will be performed",
    "plot_type": "histogram|scatter|2dhist|line|none",
    "column_x": "column name for x-axis (or main variable for histogram)",
    "column_y": "column name for y-axis (required for scatter/2dhist/line, null for histogram)",
    "bins": 50,
    "range_min": null,
    "range_max": null,
    "title": "Plot title with proper physics notation",
    "xlabel": "X-axis label with units",
    "ylabel": "Y-axis label with units",
    "statistics": ["mean", "std", "median"] or [] if none requested,
    "filters": {{"column_name": {{"min": value, "max": value}}}} or null
}}

Examples:
- For "plot invariant mass": {{"plot_type": "histogram", "column_x": "M", "bins": 50, "title": "Four-Lepton Invariant Mass", "xlabel": "M [GeV/c²]", "ylabel": "Events", "column_y": null}}
- For "show pt vs eta": {{"plot_type": "scatter", "column_x": "eta1", "column_y": "pt1", "title": "Transverse Momentum vs Pseudorapidity", "xlabel": "η", "ylabel": "p_T [GeV/c]"}}
- For "plot mass between 70 and 200 GeV": {{"plot_type": "histogram", "column_x": "M", "bins": 50, "filters": {{"M": {{"min": 70, "max": 200}}}}, "column_y": null}}
"""

    # Call OpenAI API
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse response
        spec = json.loads(response.choices[0].message.content)
        
        # Execute the analysis safely using the declarative specification
        try:
            result = execute_safe_analysis(df, spec)
            result['specification'] = spec
            return result
            
        except Exception as exec_error:
            return {
                'success': False,
                'error': f'Error executing analysis: {str(exec_error)}',
                'specification': spec,
                'explanation': spec.get('explanation', '')
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f'Error calling OpenAI API: {str(e)}'
        }


def execute_safe_analysis(df: pd.DataFrame, spec: dict) -> dict:
    """
    Safely execute analysis based on declarative specification.
    No arbitrary code execution - only predefined safe operations.
    
    Args:
        df: DataFrame with data
        spec: Declarative specification from AI
        
    Returns:
        Analysis results with plot and statistics
    """
    plot_type = spec.get('plot_type', 'none')
    column_x = spec.get('column_x')
    column_y = spec.get('column_y')
    
    # Validate plot type requirements
    if plot_type in ['scatter', '2dhist', 'line']:
        if not column_y:
            raise ValueError(f"Plot type '{plot_type}' requires column_y to be specified")
    
    # Validate columns exist
    if column_x and column_x not in df.columns:
        raise ValueError(f"Column {column_x} not found in data")
    if column_y and column_y not in df.columns:
        raise ValueError(f"Column {column_y} not found in data")
    
    # Apply filters if specified (safe operations only)
    filtered_df = df.copy()
    if spec.get('filters'):
        for col, filter_spec in spec['filters'].items():
            if col in filtered_df.columns:
                # Only allow safe comparison operations
                if 'min' in filter_spec:
                    filtered_df = filtered_df[filtered_df[col] >= filter_spec['min']]
                if 'max' in filter_spec:
                    filtered_df = filtered_df[filtered_df[col] <= filter_spec['max']]
    
    # Calculate statistics if requested
    stats_results = {}
    if spec.get('statistics') and column_x:
        stat_functions = {
            'mean': filtered_df[column_x].mean,
            'std': filtered_df[column_x].std,
            'median': filtered_df[column_x].median,
            'min': filtered_df[column_x].min,
            'max': filtered_df[column_x].max,
            'count': filtered_df[column_x].count
        }
        
        for stat in spec.get('statistics', []):
            if stat in stat_functions:
                stats_results[stat] = stat_functions[stat]()
    
    # Create plot based on type
    fig = None
    if plot_type != 'none':
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'histogram':
            bins = spec.get('bins', 50)
            range_min = spec.get('range_min')
            range_max = spec.get('range_max')
            
            data = filtered_df[column_x].dropna()
            
            if range_min is not None and range_max is not None:
                hist_range = (range_min, range_max)
            else:
                hist_range = None
            
            ax.hist(data, bins=bins, range=hist_range, 
                   edgecolor='black', alpha=0.7, color='steelblue')
            
            # Add Higgs mass line if plotting mass around that range
            if column_x in ['M', 'mass', 'm_inv', 'M4l']:
                data_min, data_max = data.min(), data.max()
                if data_min <= 125 <= data_max:
                    ax.axvline(125, color='red', linestyle='--', 
                              linewidth=2, label='Higgs mass (125 GeV)')
                    ax.legend()
        
        elif plot_type == 'scatter':
            data_x = filtered_df[column_x].dropna()
            data_y = filtered_df[column_y].dropna()
            
            # Only plot where both are valid
            valid_mask = ~(filtered_df[column_x].isna() | filtered_df[column_y].isna())
            ax.scatter(filtered_df[column_x][valid_mask], 
                      filtered_df[column_y][valid_mask],
                      alpha=0.5, s=10)
        
        elif plot_type == '2dhist':
            data_x = filtered_df[column_x].dropna()
            data_y = filtered_df[column_y].dropna()
            
            valid_mask = ~(filtered_df[column_x].isna() | filtered_df[column_y].isna())
            ax.hist2d(filtered_df[column_x][valid_mask],
                     filtered_df[column_y][valid_mask],
                     bins=50, cmap='Blues')
            plt.colorbar(ax.collections[0], ax=ax, label='Events')
        
        elif plot_type == 'line':
            ax.plot(filtered_df[column_x], filtered_df[column_y] if column_y else filtered_df.index)
        
        # Set labels and title
        ax.set_xlabel(spec.get('xlabel', column_x), fontsize=12)
        ax.set_ylabel(spec.get('ylabel', column_y if column_y else 'Events'), fontsize=12)
        ax.set_title(spec.get('title', 'Physics Analysis'), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    return {
        'success': True,
        'explanation': spec.get('explanation', ''),
        'figure': fig,
        'results': stats_results if stats_results else None,
        'statistics': spec.get('statistics', '')
    }


def suggest_analysis(df: pd.DataFrame) -> list:
    """
    Suggest possible analyses based on available data columns.
    
    Args:
        df: DataFrame with particle physics data
        
    Returns:
        List of suggested analysis prompts
    """
    
    suggestions = []
    columns = df.columns.tolist()
    
    # Check for common physics variables
    if any('M' in col or 'mass' in col.lower() for col in columns):
        suggestions.append("Plot the invariant mass distribution")
    
    if any('pt' in col.lower() for col in columns):
        suggestions.append("Show transverse momentum distribution")
    
    if any('eta' in col for col in columns):
        suggestions.append("Display pseudorapidity distribution")
    
    if any('phi' in col for col in columns):
        suggestions.append("Plot azimuthal angle distribution")
    
    # Check for multiple leptons (indicating 4-lepton analysis)
    lepton_indices = [col[-1] for col in columns if col[:-1] in ['pt', 'eta', 'phi', 'E'] and col[-1].isdigit()]
    if len(set(lepton_indices)) >= 4:
        suggestions.append("Analyze four-lepton system kinematics")
    
    return suggestions


def generate_physics_plot(df: pd.DataFrame, plot_type: str, **kwargs):
    """
    Generate common physics plots with proper styling.
    
    Args:
        df: DataFrame with data
        plot_type: Type of plot (mass, momentum, eta_phi, etc.)
        **kwargs: Additional parameters for customization
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'invariant_mass':
        mass_col = kwargs.get('mass_col', 'M')
        bins = kwargs.get('bins', 50)
        range_min = kwargs.get('range_min', df[mass_col].min())
        range_max = kwargs.get('range_max', df[mass_col].max())
        
        ax.hist(df[mass_col], bins=bins, range=(range_min, range_max), 
                edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Four-lepton invariant mass $M_{4\\ell}$ [GeV/c²]', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title('Four-Lepton Invariant Mass Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add Higgs mass line if in range
        if range_min <= 125 <= range_max:
            ax.axvline(125, color='red', linestyle='--', linewidth=2, label='Higgs mass (125 GeV)')
            ax.legend()
    
    elif plot_type == 'momentum':
        pt_col = kwargs.get('pt_col', 'pt1')
        bins = kwargs.get('bins', 50)
        
        ax.hist(df[pt_col].dropna(), bins=bins, edgecolor='black', alpha=0.7, color='forestgreen')
        ax.set_xlabel('Transverse momentum $p_T$ [GeV/c]', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title('Transverse Momentum Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    elif plot_type == 'eta_phi':
        eta_col = kwargs.get('eta_col', 'eta1')
        phi_col = kwargs.get('phi_col', 'phi1')
        
        scatter = ax.scatter(df[eta_col], df[phi_col], alpha=0.5, s=10)
        ax.set_xlabel('Pseudorapidity η', fontsize=12)
        ax.set_ylabel('Azimuthal angle φ [rad]', fontsize=12)
        ax.set_title('η-φ Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
