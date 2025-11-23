"""
AI-powered analysis assistant using Google Gemini (FREE).
Interprets natural language prompts and generates an analysis plan (JSON) 
which is then executed by the Python side.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
import math
import io 
import base64 
from typing import Dict, Any, List

# --- Local Utility Imports ---
# Import utility functions from the local package (e.g., for mass calculation)
# We use a try/except block for robustness in case this file is run standalone.
try:
    from utils.physics_utils import (
        calculate_invariant_mass_4lepton, 
        calculate_invariant_mass_2particle
    )
    # The constants from utils will be used by the model prompt
    from utils.physics_utils import MUON_MASS, ELECTRON_MASS, HIGGS_MASS_EXPECTED 
except ImportError:
    # Fallback for testing/standalone use
    def calculate_invariant_mass_4lepton(*args, **kwargs): return np.array([])
    def calculate_invariant_mass_2particle(*args, **kwargs): return np.array([])
    MUON_MASS = 0.105658
    ELECTRON_MASS = 0.000511
    HIGGS_MASS_EXPECTED = 125.0


# --- Gemini Client Setup ---
# The newest Gemini model is "gemini-2.5-flash" or "gemini-2.5-pro"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# This is using Google Gemini's free API
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# --- Available Analysis Functions (Internal) ---

def plot_histogram_internal(df: pd.DataFrame, ax: plt.Axes, plot_type: str, **kwargs) -> Dict[str, Any]:
    """Helper to run the plotting logic based on type."""

    # Get the column from kwargs or use a default fallback (M for invariant mass)
    col = kwargs.get('column', 'M_4l')
    
    if df.empty:
        raise ValueError("DataFrame is empty, cannot plot.")

    if plot_type == 'invariant_mass':
        
        # Map common names to the actual column name 'M'
        if col in ['M_4l', 'M', 'm4l']:
            # Assume 'M' is the 4-lepton mass column, or use the calculated one if present
            col_to_use = 'M' if 'M' in df.columns else 'M_4l_calculated'
        else:
            col_to_use = col
        
        if col_to_use not in df.columns:
            # Fallback to general histogram if specific column not found
            col_to_use = kwargs.get('data_col', df.columns[0])

        range_min = kwargs.get('range_min', df[col_to_use].min())
        range_max = kwargs.get('range_max', df[col_to_use].max())
        bins = kwargs.get('bins', 50)

        # Filter data to the specified range
        data_to_plot = df[col_to_use].dropna()
        data_to_plot = data_to_plot[(data_to_plot >= range_min) & (data_to_plot <= range_max)]


        ax.hist(data_to_plot, bins=bins, range=(range_min, range_max),
                edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Four-lepton invariant mass $M_{4\ell}$ [GeV/c²]', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title('Four-Lepton Invariant Mass Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add Higgs mass line if in range
        if range_min <= HIGGS_MASS_EXPECTED <= range_max:
            ax.axvline(HIGGS_MASS_EXPECTED, color='red', linestyle='--', linewidth=2, label=f'Higgs mass ({HIGGS_MASS_EXPECTED} GeV)')
            ax.legend()
            
        # Calculate peak
        if len(data_to_plot) > 0:
            hist_counts, bin_edges = np.histogram(data_to_plot, bins=bins, range=(range_min, range_max))
            peak_bin_index = np.argmax(hist_counts)
            peak_mass = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index+1]) / 2
            return {'peak_mass': float(peak_mass)}
        
        return {}


    elif plot_type == 'momentum':
        pt_col = kwargs.get('pt_col', 'pt1')
        bins = kwargs.get('bins', 50)
        
        ax.hist(df[pt_col].dropna(), bins=bins, edgecolor='black', alpha=0.7, color='forestgreen')
        ax.set_xlabel('Transverse momentum $p_T$ [GeV/c]', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title('Transverse Momentum Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Generic histogram case for any column
    elif plot_type == 'general_hist':
        col = kwargs.get('column', df.columns[0])
        bins = kwargs.get('bins', 50)
        range_min = kwargs.get('range_min', df[col].min())
        range_max = kwargs.get('range_max', df[col].max())

        ax.hist(df[col].dropna(), bins=bins, range=(range_min, range_max),
                edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    return {} # Placeholder for potential plot-specific metrics

def plot_2d_internal(df: pd.DataFrame, ax: plt.Axes, plot_type: str, **kwargs) -> Dict[str, Any]:
    """Helper to run the 2D plotting logic (scatter or heatmaps/density)."""

    if df.empty:
        raise ValueError("DataFrame is empty, cannot plot.")
        
    if plot_type == 'eta_phi':
        eta_col = kwargs.get('eta_col', 'eta1')
        phi_col = kwargs.get('phi_col', 'phi1')
        
        # Using 2D histogram for density map
        hist, x_edges, y_edges, im = ax.hist2d(
            df[phi_col].dropna().values, 
            df[eta_col].dropna().values, 
            bins=50, 
            cmap='viridis'
        )
        
        ax.set_xlabel(f'Azimuthal angle {phi_col} [rad]', fontsize=12)
        ax.set_ylabel(f'Pseudorapidity {eta_col}', fontsize=12)
        ax.set_title('Lepton $\\eta$ vs $\\phi$ Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Events Density', rotation=270, labelpad=15)
    
    elif plot_type == 'scatter':
        x_col = kwargs.get('x_col', df.columns[0])
        y_col = kwargs.get('y_col', df.columns[1])
        
        ax.scatter(df[x_col], df[y_col], s=5, alpha=0.5, color='darkorange')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    return {}


# --- Main AI Analysis Function ---

def analyze_with_ai(user_prompt: str, df: pd.DataFrame) -> dict:
    """
    Analyze particle physics data based on natural language prompt.
    Uses a safe declarative approach by asking the AI to output a JSON analysis plan
    which is then executed by the Python side.
    
    Args:
        user_prompt: Natural language analysis request
        df: DataFrame with particle physics data
        
    Returns:
        Dictionary with analysis results, plots (base64 PNG), and explanations
    """
    
    if not gemini_client:
        return {
            'success': False,
            'error': 'Gemini API key not configured. Get a free API key from https://ai.google.dev/'
        }
    
    # 1. Get data info
    # The snippet of dtypes is limited to prevent massive prompts
    data_info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items() if len(df.columns) < 50}
    }
    
    # 2. Define the analysis tools (functions the model can call)
    tool_defs = [
        types.Tool.from_function(
            func=lambda column, min_value=None, max_value=None, condition=None: {
                "filtered_shape": df.shape, # Placeholder for function signature only
            },
            name="filter_data",
            description="Filters the current DataFrame based on one or more conditions (e.g., 'pt1 > 20' or 'M_4l < 100'). Only apply a single column/min/max or a single condition string per call."
        ),
        types.Tool.from_function(
            func=lambda mass_type, pt_cols, eta_cols, phi_cols, mass_cols: {
                "calculated_mass_shape": df.shape, # Placeholder for function signature only
            },
            name="calculate_mass",
            description="Calculates the invariant mass of a system (e.g., '4-lepton' or 'dilepton'). Use this BEFORE plotting mass if the mass column is missing. Specify mass_type ('4l', '2l', 'transverse') and the lists of columns for each kinematic variable: pt_cols, eta_cols, phi_cols, mass_cols."
        ),
        types.Tool.from_function(
            func=lambda plot_type, column=None, bins=50, range_min=None, range_max=None: {
                "plot_info": "1D histogram parameters" # Placeholder
            },
            name="plot_histogram",
            description="Generates a 1D histogram. Use 'invariant_mass' for Higgs or Z mass plots (e.g., column='M_4l'), or 'general_hist' for other single columns (e.g., column='pt1'). You can specify the number of bins (default 50) and a custom range."
        ),
        types.Tool.from_function(
            func=lambda plot_type, x_col=None, y_col=None: {
                "plot_info": "2D plot parameters" # Placeholder
            },
            name="plot_2d",
            description="Generates a 2D plot. Use 'eta_phi' for pseudorapidity vs. azimuthal angle plots, or 'scatter' for general variable correlation plots."
        ),
        types.Tool.from_function(
            func=lambda column: {
                "statistics": "Descriptive statistics output" # Placeholder
            },
            name="get_stats",
            description="Calculates and returns descriptive statistics (mean, std, min, max, count) for a specified numeric column."
        )
    ]
    
    # 3. Construct the System Instruction
    system_prompt = f"""
    You are an expert AI Particle Physics Data Analyst. Your task is to plan the analysis of a pandas DataFrame (named 'df') based on the user's request.

    The DataFrame contains particle physics data with the following structure:
    - Shape: {data_info['shape']}
    - Columns: {data_info['columns']}
    - Data Types (snippet): {json.dumps(data_info['dtypes'])}

    **Physics Constants:**
    - Expected Higgs Mass (H): {HIGGS_MASS_EXPECTED:.1f} GeV
    - Muon Mass (mu): {MUON_MASS:.4f} GeV
    - Electron Mass (e): {ELECTRON_MASS:.4f} GeV

    **Analysis Plan Format:**
    1.  Your response MUST be a single JSON object.
    2.  The JSON MUST have two keys: 'explanation' (a string) and 'pipeline' (a list of tool calls).
    3.  'explanation' must be a concise, professional, step-by-step description of the analysis you plan to perform to address the user's prompt.
    4.  'pipeline' must be an ordered list of tool calls using the available functions. The output of one function (like 'filter_data') implicitly acts as the input for the next function, so do not explicitly pass dataframes.
    5.  You MUST use the column names exactly as they appear in the data (e.g., 'pt1', 'eta1', 'M_4l').
    6.  If the user asks for a plot, the last step in the pipeline MUST be a 'plot_histogram' or 'plot_2d' call.
    7.  If a column required for invariant mass calculation (e.g., 'M_4l') is missing, you MUST include a 'calculate_mass' step before the 'plot_histogram' step, providing the required column lists.

    **Example (User wants Higgs mass plot):**
    "explanation": "We will first filter the data to select only events within a relevant mass window (100-150 GeV) to reduce background, and then plot the invariant mass distribution to look for the Higgs signal peak.",
    "pipeline": [
        {{
            "function": "filter_data",
            "args": {{
                "column": "M_4l", 
                "min_value": 100, 
                "max_value": 150
            }}
        }},
        {{
            "function": "plot_histogram",
            "args": {{
                "plot_type": "invariant_mass", 
                "column": "M_4l",
                "bins": 100,
                "range_min": 100, 
                "range_max": 150
            }}
        }}
    ]
    """

    # 4. Make the API Call to generate the analysis plan
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                tools=tool_defs,
                response_mime_type="application/json",
                # The response schema defines the exact structure we expect from the model
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "explanation": {"type": "STRING"},
                        "pipeline": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "function": {"type": "STRING", "description": "The name of the tool function to call."},
                                    "args": {"type": "OBJECT", "description": "The arguments for the function."}
                                }
                            }
                        }
                    },
                    "required": ["explanation", "pipeline"]
                }
            )
        )
        
        # 5. Parse and Execute the Analysis Plan
        analysis_plan = json.loads(response.text)
        current_df = df.copy()
        pipeline_results = []
        
        plot_image_base64 = None
        
        for step in analysis_plan.get('pipeline', []):
            function_name = step.get('function')
            args = step.get('args', {})
            
            if function_name == 'filter_data':
                col = args.get('column')
                min_val = args.get('min_value')
                max_val = args.get('max_value')
                condition_str = args.get('condition')
                
                # Build filter mask
                mask = pd.Series(True, index=current_df.index)
                
                if col and col in current_df.columns:
                    if min_val is not None:
                        mask &= (current_df[col] >= min_val)
                    if max_val is not None:
                        mask &= (current_df[col] <= max_val)
                        
                elif condition_str:
                    try:
                        # Use pandas.eval for safe and efficient filtering based on a string condition
                        # Note: This relies on the current_df object being available in the eval context
                        mask &= current_df.eval(condition_str, engine='python')
                    except Exception as e:
                        result_msg = f"Filtering with custom condition failed: {str(e)}"
                        pipeline_results.append({'function': function_name, 'result': result_msg})
                        continue

                else:
                    result_msg = "Filtering skipped: no valid column or condition provided."
                    pipeline_results.append({'function': function_name, 'result': result_msg})
                    continue

                original_shape = current_df.shape
                current_df = current_df[mask].copy()
                
                result_msg = f"Data filtered: {original_shape[0]} events -> {current_df.shape[0]} events."
                pipeline_results.append({'function': function_name, 'result': result_msg})

            elif function_name == 'calculate_mass':
                mass_type = args.get('mass_type')
                
                if mass_type == '4l':
                    pt_cols = args.get('pt_cols', ['pt1', 'pt2', 'pt3', 'pt4'])
                    eta_cols = args.get('eta_cols', ['eta1', 'eta2', 'eta3', 'eta4'])
                    phi_cols = args.get('phi_cols', ['phi1', 'phi2', 'phi3', 'phi4'])
                    mass_cols = args.get('mass_cols', ['mass1', 'mass2', 'mass3', 'mass4'])
                    
                    if all(c in current_df.columns for c in pt_cols + eta_cols + phi_cols + mass_cols):
                        invariant_masses = calculate_invariant_mass_4lepton(
                            current_df, pt_cols, eta_cols, phi_cols, mass_cols
                        )
                        # Add the calculated mass column back to the DataFrame
                        current_df['M_4l_calculated'] = invariant_masses
                        result_msg = f"Calculated 4-lepton invariant mass for {len(invariant_masses)} events. New column: 'M_4l_calculated'."
                    else:
                        missing = [c for c in pt_cols + eta_cols + phi_cols + mass_cols if c not in current_df.columns]
                        result_msg = f"Failed to calculate 4-lepton mass: Missing columns: {', '.join(missing)}"
                        
                elif mass_type == '2l':
                    # This is a placeholder for 2-lepton mass calculation logic
                    result_msg = f"2-lepton mass calculation is supported but requires explicit column definition in the current plan."
                
                else:
                    result_msg = f"Mass calculation for type '{mass_type}' is planned but not fully implemented in this system."
                    
                pipeline_results.append({'function': function_name, 'result': result_msg})

            elif function_name == 'plot_histogram':
                # Generate the plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                try:
                    plot_metrics = plot_histogram_internal(current_df, ax, args.get('plot_type', 'general_hist'), **args)
                    ax.margins(x=0.01)
                    
                    # Convert plot to base64 string
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    plot_image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    result_msg = f"Generated 1D Histogram. Plot Type: {args.get('plot_type')}. "
                    if 'peak_mass' in plot_metrics:
                        result_msg += f"Observed Peak: {plot_metrics['peak_mass']:.2f} GeV."
                        
                except Exception as e:
                    result_msg = f"Plotting failed: {str(e)}"
                
                pipeline_results.append({'function': function_name, 'result': result_msg})


            elif function_name == 'plot_2d':
                # Generate the plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                try:
                    plot_2d_internal(current_df, ax, args.get('plot_type', 'scatter'), **args)
                    
                    # Convert plot to base64 string
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    plot_image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    result_msg = f"Generated 2D Plot. Plot Type: {args.get('plot_type')}."
                        
                except Exception as e:
                    result_msg = f"2D Plotting failed: {str(e)}"
                
                pipeline_results.append({'function': function_name, 'result': result_msg})
                
                
            elif function_name == 'get_stats':
                # Basic statistics step
                col = args.get('column')
                if col and col in current_df.columns:
                    stats = current_df[col].describe().to_dict()
                    result_msg = f"Statistics for {col}:\n{json.dumps(stats, indent=2)}\n"
                else:
                    result_msg = f"Statistics calculation skipped due to missing or invalid column: {col}."
                pipeline_results.append({'function': function_name, 'result': result_msg})

            else:
                pipeline_results.append({'function': function_name, 'result': f"Unknown function: {function_name}"})

        
        # 6. Collate the final result
        return {
            'success': True,
            'explanation': analysis_plan.get('explanation', 'Analysis pipeline executed.'),
            'results': pipeline_results,
            'final_df_shape': current_df.shape,
            'plot_base64': plot_image_base64
        }

    except Exception as e:
        # Catch JSON parsing error or API error
        return {'success': False, 'error': f"AI Analysis Failed: {type(e).__name__}: {str(e)}"}

if __name__ == '__main__':
    # This block is for local testing only
    print("AI Analysis Assistant Module Loaded.")
