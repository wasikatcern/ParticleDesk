"""
AI-powered analysis assistant using Google Gemini (FREE).
Interprets natural language prompts and generates a full analysis pipeline in JSON format.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from scipy.optimize import curve_fit
from scipy.stats import norm
import logging

# Ensure utility functions are imported for the execution engine
from .physics_utils import (
    calculate_invariant_mass_2particle,
    calculate_transverse_mass,
    delta_r
    # Note: calculate_invariant_mass_4lepton is too complex for general AI generation, 
    # but can be added as a specific function with a clear input map in the prompt.
)

logger = logging.getLogger(__name__)

# the newest Gemini model is "gemini-2.5-flash" or "gemini-2.5-pro"
# Do not change this unless explicitly requested by the user
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

def analyze_with_ai(user_prompt: str, df: pd.DataFrame) -> dict:
    """
    Analyze particle physics data based on natural language prompt.
    Generates and executes a multi-step declarative analysis pipeline.
    """
    
    if not gemini_client:
        return {
            'success': False,
            'error': 'Gemini API key not configured. Get a free API key from https://ai.google.dev/'
        }
    
    # --- STEP 1: AI GENERATION OF DECLARATIVE SPECIFICATION ---
    data_info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'sample': df.head(1).to_dict('records')
    }
    
    # Detailed System Prompt for complex physics analysis
    system_prompt = f"""You are a particle physics data analysis expert. Your role is to convert a user's natural language request into a multi-step, declarative analysis pipeline in JSON format.

The analysis MUST proceed in this order: 
1. **Calculations**: Create new derived variables (e.g., invariant mass, DeltaR).
2. **Analysis Steps**: Apply filters (cuts) to the data.
3. **Visualization**: Define one or more plot layers (for overlapping plots).
4. **Statistical Report**: Perform statistical operations (e.g., fitting).

Available data columns: {', '.join(data_info['columns'])}
Number of events: {data_info['shape'][0]}

**Available Calculation Functions and their arguments (use for 'calculate_variable' step):**
- **invariant_mass_2particle**: Calculates $M_{inv}$ for two particles. Requires: `pt1`, `eta1`, `phi1`, `mass1` (or fixed mass constant like `MUON_MASS`), and similarly for particle 2.
- **delta_r**: Calculates $\Delta R$ between two particles. Requires: `eta1`, `phi1`, `eta2`, `phi2`.
- **calculate_transverse_mass**: Calculates $M_T$. Requires: `pt`, `phi`, `met`, `met_phi`.

**Return ONLY a single JSON object with the following structure:**

{{
    "explanation": "Brief explanation of the analysis strategy and physics context.",
    "pipeline": [
        {{
            "type": "calculate_variable",
            "new_column": "M_ll",
            "function": "invariant_mass_2particle",
            "args": {{"pt1": "lep1_pt", "eta1": "lep1_eta", "phi1": "lep1_phi", "mass1": "MUON_MASS", "pt2": "lep2_pt", "eta2": "lep2_eta", "phi2": "lep2_phi", "mass2": "MUON_MASS"}}
        }},
        {{
            "type": "apply_filter",
            "column": "lep1_pt",
            "min": 20.0,
            "max": null,
            "condition_type": "min_max"
        }}
    ],
    "plotting": [
        {{
            "data_col": "M_ll",
            "plot_type": "histogram",
            "label": "Z Boson Peak",
            "filters": {{"M_ll": {{"min": 60, "max": 120}}}},
            "config": {{"bins": 100, "range_min": 60, "range_max": 120, "color": "steelblue", "alpha": 0.7}}
        }}
    ],
    "statistics_report": [
        {{
            "column": "M_ll",
            "operation": "fit_gaussian",
            "range": [80, 100]
        }}
    ],
    "metadata": {{
        "title": "Invariant Mass Distribution for User-Defined Z Boson Selection",
        "xlabel": "Dilepton Invariant Mass $M_{\\ell\\ell}$ [GeV/c²]",
        "ylabel": "Events / bin"
    }}
}}

Ensure all column names used in 'args', 'column', and 'data_col' fields exist in the data columns provided or were created in a previous 'calculate_variable' step. Use proper physics notation in 'title', 'xlabel', and 'ylabel'. Focus on providing a complete and correct analysis pipeline for the user's request.
"""
    
    # Call Gemini API with JSON response schema enforced
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[
                    types.Part(text=system_prompt + "\n\nUser request: " + user_prompt)
                ])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        spec = json.loads(response.text)
        
    except Exception as e:
        return {'success': False, 'error': f"AI generation failed: {str(e)}"}

    # --- STEP 2: PYTHON EXECUTION ENGINE (THE FIXED ANALYSIS) ---
    filtered_df = df.copy()
    analysis_results = {}
    
    # Execute Pipeline Steps (Calculations and Filters)
    try:
        if 'pipeline' in spec:
            for step in spec['pipeline']:
                if step['type'] == 'calculate_variable':
                    # Execute calculation function
                    func_name = step['function']
                    func_args = step['args']
                    
                    # Map constant names to their values (e.g., "MUON_MASS")
                    mapped_args = {}
                    for k, v in func_args.items():
                        if isinstance(v, str) and v.isupper() and 'MASS' in v:
                            try:
                                # This assumes physics_utils is imported correctly and has constants
                                mapped_args[k] = getattr(importlib.import_module(".physics_utils", package=".") if __name__ == '__main__': pass else __import__('utils.physics_utils', fromlist=['*']), v)
                            except AttributeError:
                                # Fallback to looking in the dataframe or assuming the string is the value
                                mapped_args[k] = float(v) if v.replace('.', '', 1).isdigit() else v
                        else:
                            mapped_args[k] = v

                    # Prepare function call (dynamic dispatch)
                    if func_name == 'invariant_mass_2particle':
                        new_col_data = calculate_invariant_mass_2particle(
                            filtered_df, 
                            pt1=filtered_df[mapped_args['pt1']].values,
                            eta1=filtered_df[mapped_args['eta1']].values,
                            phi1=filtered_df[mapped_args['phi1']].values,
                            m1=mapped_args.get('mass1', 0.0), # Use 0.0 if not specified
                            pt2=filtered_df[mapped_args['pt2']].values,
                            eta2=filtered_df[mapped_args['eta2']].values,
                            phi2=filtered_df[mapped_args['phi2']].values,
                            m2=mapped_args.get('mass2', 0.0) # Use 0.0 if not specified
                        )
                    elif func_name == 'delta_r':
                         new_col_data = delta_r(
                            filtered_df[mapped_args['eta1']].values,
                            filtered_df[mapped_args['phi1']].values,
                            filtered_df[mapped_args['eta2']].values,
                            filtered_df[mapped_args['phi2']].values
                         )
                    elif func_name == 'calculate_transverse_mass':
                         new_col_data = calculate_transverse_mass(
                            filtered_df[mapped_args['pt']].values,
                            filtered_df[mapped_args['phi']].values,
                            filtered_df[mapped_args['met']].values,
                            filtered_df[mapped_args['met_phi']].values
                         )
                    else:
                        raise ValueError(f"Unsupported calculation function: {func_name}")
                        
                    filtered_df[step['new_column']] = new_col_data
                    analysis_results[f"Calculated: {step['new_column']}"] = f"{step['new_column']} created using {func_name}"
                    
                elif step['type'] == 'apply_filter':
                    col = step['column']
                    min_val = step.get('min')
                    max_val = step.get('max')
                    
                    initial_count = len(filtered_df)
                    if min_val is not None:
                        filtered_df = filtered_df[filtered_df[col] >= min_val]
                    if max_val is not None:
                        filtered_df = filtered_df[filtered_df[col] <= max_val]
                    
                    final_count = len(filtered_df)
                    analysis_results[f"Filter: {col}"] = f"Applied cut: {min_val or '-'} <= {col} <= {max_val or '-'}. Events remaining: {final_count} (lost {initial_count - final_count})"
                    
    except Exception as e:
        return {'success': False, 'error': f"Error during pipeline execution: {str(e)}"}


    # Execute Plotting Steps (Overlapping Plots)
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'plotting' in spec and spec['plotting']:
        try:
            for plot_spec in spec['plotting']:
                data_col = plot_spec['data_col']
                plot_type = plot_spec['plot_type']
                label = plot_spec.get('label', 'Data')
                config = plot_spec.get('config', {})
                filters = plot_spec.get('filters', {})
                
                # Apply temporary filters for this plot layer
                plot_data = filtered_df.copy()
                for col, f_spec in filters.items():
                    if 'min' in f_spec and f_spec['min'] is not None:
                        plot_data = plot_data[plot_data[col] >= f_spec['min']]
                    if 'max' in f_spec and f_spec['max'] is not None:
                        plot_data = plot_data[plot_data[col] <= f_spec['max']]
                
                data = plot_data[data_col].dropna().values
                
                if plot_type == 'histogram' and len(data) > 0:
                    ax.hist(
                        data, 
                        bins=config.get('bins', 50),
                        range=(config['range_min'], config['range_max']) if config.get('range_min') and config.get('range_max') else None,
                        edgecolor='black',
                        alpha=config.get('alpha', 0.7),
                        color=config.get('color', None),
                        histtype=config.get('histtype', 'stepfilled'),
                        label=label
                    )
                elif plot_type == 'scatter':
                    col_y = config['column_y']
                    ax.scatter(data, plot_data[col_y].values, s=5, alpha=config.get('alpha', 0.5), label=label, color=config.get('color', None))
                    ax.set_ylabel(spec['metadata'].get('ylabel', col_y))

            # Apply final plot metadata
            ax.set_title(spec['metadata'].get('title', 'AI Generated Plot'), fontweight='bold')
            ax.set_xlabel(spec['metadata'].get('xlabel', 'X-axis'))
            ax.set_ylabel(spec['metadata'].get('ylabel', 'Events'))
            ax.grid(True, alpha=0.3)
            if len(spec['plotting']) > 1 or any('label' in p for p in spec['plotting']):
                ax.legend()
            
            analysis_results['Plot'] = fig
            
        except Exception as e:
            analysis_results['Plot Error'] = f"Plotting failed: {str(e)}"
            fig = None
    else:
        fig = None

    # Execute Statistical Reporting (Fitting)
    if 'statistics_report' in spec and spec['statistics_report']:
        stats_data = []
        for stat_spec in spec['statistics_report']:
            col = stat_spec['column']
            operation = stat_spec['operation']
            fit_range = stat_spec.get('range')

            if col in filtered_df.columns and operation == 'fit_gaussian':
                data = filtered_df[col].dropna()
                if fit_range:
                    data = data[(data >= fit_range[0]) & (data <= fit_range[1])]

                if len(data) > 20:
                    # Perform histogram and initial parameter estimation
                    try:
                        counts, bin_edges = np.histogram(data, bins=50, range=fit_range)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        mean_guess = data.mean()
                        std_guess = data.std()
                        amplitude_guess = counts.max()
                        
                        # Define Gaussian function
                        def gaussian(x, amplitude, mean, stddev):
                            return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

                        popt, pcov = curve_fit(gaussian, bin_centers, counts, 
                                               p0=[amplitude_guess, mean_guess, std_guess],
                                               maxfev=5000)
                        
                        perr = np.sqrt(np.diag(pcov))
                        
                        stats_data.append({
                            "Statistic": f"Gaussian Fit (Range: {fit_range[0]}-{fit_range[1]} GeV)",
                            "Result": f"Peak Mass: {popt[1]:.3f} ± {perr[1]:.3f} GeV/c²",
                            "Amplitude": f"{popt[0]:.2f} ± {perr[0]:.2f}",
                            "Width (σ)": f"{abs(popt[2]):.3f} ± {perr[2]:.3f} GeV/c²"
                        })
                        
                        # Add fit plot to the figure if it exists
                        if fig:
                            fit_x = np.linspace(bin_edges.min(), bin_edges.max(), 1000)
                            ax.plot(fit_x, gaussian(fit_x, *popt), 'r--', linewidth=2, label='Gaussian Fit')
                            if 'Plot' in analysis_results:
                                # Re-set the figure with the fit added
                                analysis_results['Plot'] = fig
                            
                    except Exception as e:
                        stats_data.append({"Statistic": f"Gaussian Fit Error ({col})", "Result": f"Fit failed: {str(e)}"})
                else:
                    stats_data.append({"Statistic": f"Gaussian Fit Error ({col})", "Result": "Not enough data points for fit."})

        analysis_results['Statistics'] = pd.DataFrame(stats_data)
    

    # --- STEP 3: CONSTRUCT FINAL RESULT ---
    return {
        'success': True,
        'explanation': spec.get('explanation', 'Analysis performed.'),
        'specification': spec,
        'results': analysis_results,
        'metadata': spec.get('metadata', {})
    }
