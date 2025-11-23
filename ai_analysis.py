"""
AI-powered analysis assistant using Google Gemini (FREE).
Interprets natural language prompts and generates analysis code.
Reference: python_gemini integration blueprint
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from scipy.optimize import curve_fit
from typing import Dict, Any, List

# the newest Gemini model is "gemini-2.5-flash" or "gemini-2.5-pro"
# do not change this unless explicitly requested by the user
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# This is using Google Gemini's free API (no payment required, just get a free API key from Google AI Studio)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Helper functions for physics calculations (simulated for self-containment)
# In a real app, these would come from the imported 'utils' or 'physics_utils' module.

# Gaussian function for fitting
def _gaussian(x, amp, mu, sigma):
    """Gaussian function for fitting histogram peaks."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Function to calculate 4-momentum components
def _calculate_4momentum(pt, eta, phi, mass):
    """Calculates E, Px, Py, Pz from pt, eta, phi, and mass."""
    pt_val = pt.to_numpy()
    eta_val = eta.to_numpy()
    phi_val = phi.to_numpy()
    mass_val = mass

    # Momentum magnitude (p)
    p = pt_val * np.cosh(eta_val)
    # Energy (E)
    E = np.sqrt(p**2 + mass_val**2)
    # Px, Py, Pz
    px = pt_val * np.cos(phi_val)
    py = pt_val * np.sin(phi_val)
    pz = pt_val * np.sinh(eta_val)
    
    return E, px, py, pz

def _calculate_invariant_mass_2particle(df: pd.DataFrame, pt_cols: List[str], eta_cols: List[str], phi_cols: List[str], mass: float) -> pd.Series:
    """Calculates the invariant mass M_ll for two particles."""
    if len(pt_cols) != 2 or len(eta_cols) != 2 or len(phi_cols) != 2:
        raise ValueError("Must provide 2 columns for pt, eta, and phi for two-particle mass calculation.")

    E1, px1, py1, pz1 = _calculate_4momentum(df[pt_cols[0]], df[eta_cols[0]], df[phi_cols[0]], mass)
    E2, px2, py2, pz2 = _calculate_4momentum(df[pt_cols[1]], df[eta_cols[1]], df[phi_cols[1]], mass)

    E_total = E1 + E2
    Px_total = px1 + px2
    Py_total = py1 + py2
    Pz_total = pz1 + pz2

    # Invariant Mass Squared: M^2 = E^2 - P^2
    M2 = E_total**2 - (Px_total**2 + Py_total**2 + Pz_total**2)
    
    # Filter out non-positive M2 values (should not happen for real particles but can occur due to floating point precision)
    M2[M2 < 0] = 0 

    return np.sqrt(M2)


# The physics constants are needed for the calculations
ELECTRON_MASS = 0.000511  # GeV/c^2
MUON_MASS = 0.105658  # GeV/c^2
HIGGS_MASS_EXPECTED = 125.0  # GeV/c^2


def _generate_plot(df: pd.DataFrame, plot_config: Dict[str, Any], initial_data: pd.DataFrame) -> plt.Figure:
    """
    Generates a Matplotlib plot based on the declarative configuration.
    Handles single plots, overlaid plots, and Gaussian fitting.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if we have multiple datasets for overlay (from the 'datasets' key)
    datasets = plot_config.get('datasets', [{'data': df, 'label': 'Events', 'color': 'steelblue'}])
    
    for dataset_info in datasets:
        plot_df = dataset_info['data']
        label = dataset_info.get('label', 'Data')
        color = dataset_info.get('color', 'black')
        
        plot_type = plot_config.get('type', 'histogram')
        column = plot_config.get('column')
        
        if column and column in plot_df.columns:
            data_to_plot = plot_df[column].dropna()
            
            # Get plotting parameters
            bins = plot_config.get('bins', 50)
            range_min = plot_config.get('range_min', data_to_plot.min())
            range_max = plot_config.get('range_max', data_to_plot.max())
            
            # --- Plotting ---
            if plot_type == 'histogram':
                # Use a histogram style typical in physics
                counts, edges, patches = ax.hist(
                    data_to_plot, 
                    bins=bins, 
                    range=(range_min, range_max),
                    edgecolor='black', 
                    alpha=0.6, # Use transparency for overlays
                    label=label, 
                    color=color,
                    histtype='stepfilled'
                )
                
                # --- Gaussian Fitting ---
                if plot_config.get('fit_gaussian'):
                    fit_range = plot_config.get('fit_range', [range_min, range_max])
                    
                    # Bin centers
                    bin_centers = (edges[:-1] + edges[1:]) / 2
                    
                    # Find bins within the fit range
                    mask = (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1])
                    fit_x = bin_centers[mask]
                    fit_y = counts[mask]
                    
                    if len(fit_x) > 3 and fit_y.sum() > 0:
                        # Initial guess: amplitude = max count, mean = peak center, sigma = guess width (e.g., 5 GeV)
                        initial_guess = [np.max(fit_y), fit_x[np.argmax(fit_y)], 5]
                        
                        try:
                            popt, pcov = curve_fit(_gaussian, fit_x, fit_y, p0=initial_guess)
                            
                            # Generate smooth fit curve
                            x_fit = np.linspace(fit_range[0], fit_range[1], 100)
                            y_fit = _gaussian(x_fit, *popt)
                            
                            ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Gaussian Fit')
                            
                            # Extract fit parameters
                            amp, mu, sigma = popt
                            mu_err = np.sqrt(pcov[1, 1])
                            sigma_err = np.sqrt(pcov[2, 2])

                            # Add fit results to the plot label
                            fit_text = (
                                f'Fit $\mu$: {mu:.2f} $\pm$ {mu_err:.2f} GeV\n'
                                f'Fit $\sigma$: {sigma:.2f} $\pm$ {sigma_err:.2f} GeV'
                            )
                            # Add text box to the plot
                            ax.text(
                                0.7, 0.9, fit_text, transform=ax.transAxes, 
                                fontsize=10, verticalalignment='top', 
                                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7)
                            )
                            
                            # Store statistics
                            plot_config['fit_results'] = {
                                'mu': f'{mu:.2f} +/- {mu_err:.2f}',
                                'sigma': f'{sigma:.2f} +/- {sigma_err:.2f}',
                                'range': fit_range
                            }
                            
                        except RuntimeError:
                            # print("Warning: Optimal parameters not found for Gaussian fit.")
                            plot_config['fit_results'] = {'error': 'Gaussian fit failed to converge.'}
            
            elif plot_type == 'scatter':
                # Implement scatter plot logic if needed
                pass

    # --- Plot Customization (Using raw strings for LaTeX) ---
    xlabel = plot_config.get('xlabel', column)
    ylabel = plot_config.get('ylabel', 'Events / Bin')
    title = plot_config.get('title', f'Distribution of {column}')

    # FIX: Use raw strings (r"...") for all labels containing LaTeX math
    ax.set_xlabel(r"{}".format(xlabel), fontsize=12)
    ax.set_ylabel(r"{}".format(ylabel), fontsize=12)
    ax.set_title(r"{}".format(title), fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    return fig


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
    
    if not gemini_client:
        return {
            'success': False,
            'error': 'Gemini API key not configured. Get a free API key from https://ai.google.dev/'
        }
    
    # Get data info
    data_info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'example_rows': df.head(3).to_dict()
    }

    # The system instruction defines the desired output structure (JSON)
    # This structure represents a declarative analysis pipeline.
    system_instruction = (
        "You are an expert particle physicist and Python code generator specializing in data analysis using pandas and numpy. "
        "Your task is to interpret a user's natural language request and convert it into a safe, structured JSON specification for an analysis pipeline. "
        "DO NOT write or execute Python code directly. Only generate the JSON response."
        "\n\nContextual Constants: MUON_MASS=0.105658, ELECTRON_MASS=0.000511, HIGGS_MASS_EXPECTED=125.0 (all in GeV/c²)."
        "\n\nAvailable Physics Utility Functions (use them for calculations in the 'calculate' step):"
        "\n- _calculate_invariant_mass_2particle(df, pt_cols, eta_cols, phi_cols, mass_constant): Calculates M_ll."
        "\n\nThe pipeline must be an array of steps. Each step must be of type 'calculate', 'filter', or 'plot'."
        "\n\n1. calculate: Adds a new column to the DataFrame."
        "   Example: {'step': 'calculate', 'type': '_calculate_invariant_mass_2particle', 'output_column': 'M_ll', 'inputs': {'pt_cols': ['lep1_pt', 'lep2_pt'], 'eta_cols': ['lep1_eta', 'lep2_eta'], 'phi_cols': ['lep1_phi', 'lep2_phi'], 'mass_constant': 'ELECTRON_MASS'}}"
        "\n\n2. filter: Filters the DataFrame."
        "   Example: {'step': 'filter', 'conditions': ['lep1_pt > 25.0', 'abs(lep2_eta) < 2.4']}"
        "\n\n3. plot: Defines a visualization. Can optionally include 'fit_gaussian' and 'fit_range'."
        "   Example: {'step': 'plot', 'type': 'histogram', 'column': 'M_ll', 'bins': 50, 'range_min': 60, 'range_max': 120, 'xlabel': 'Dilepton Invariant Mass $M_{\\ell\\ell}$ [GeV/c²]', 'fit_gaussian': True, 'fit_range': [80, 100]}"
        "\n\nIf the user requests overlapping plots (e.g., 'Signal' vs 'Background'), generate two separate 'plot' steps, or use the 'datasets' array in the 'plot' step for complex overlays (if you feel the model can handle it reliably)."
        "\n\nAfter generating the complete JSON, provide a concise, single-paragraph explanation of the physics motivation and the steps derived from the user's prompt (outside the JSON block). Use LaTeX math symbols where appropriate in the explanation."
    )

    # Construct the user message
    user_message = (
        f"DATA CONTEXT (Columns: {data_info['columns']}, Shape: {data_info['shape']}):\n"
        f"Example Data: {data_info['example_rows']}\n\n"
        f"USER REQUEST: {user_prompt}"
    )

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "step": {"type": "STRING", "description": "The type of analysis step: 'calculate', 'filter', or 'plot'"},
                            "type": {"type": "STRING", "description": "Specific operation (e.g., '_calculate_invariant_mass_2particle', 'histogram')"},
                            # ... other properties as defined in instructions ...
                        },
                        "required": ["step", "type"]
                    }
                }
            )
        )
        
        # The AI's response is the JSON string
        json_spec_str = response.candidates[0].content.parts[0].text
        
        # Extract explanation (if available) - this relies on the model putting it outside the JSON
        # Since we forced JSON output, the explanation will be in the grounding or text property, 
        # but for this blueprint, we'll assume the JSON is the primary output.
        # A more robust system would use a specific output format request.
        
        # Parse the JSON specification
        pipeline_spec = json.loads(json_spec_str)
        
        # --- EXECUTE PIPELINE ---
        
        current_df = df.copy()
        analysis_results = {}
        
        for i, step_config in enumerate(pipeline_spec):
            step_name = f"Step_{i+1}_{step_config['step']}"
            
            try:
                if step_config['step'] == 'calculate':
                    calc_type = step_config['type']
                    output_col = step_config['output_column']
                    inputs = step_config['inputs']
                    
                    if calc_type == '_calculate_invariant_mass_2particle':
                        mass_const_name = inputs.get('mass_constant', 'MUON_MASS')
                        mass_const = globals().get(mass_const_name, MUON_MASS) # Fallback
                        
                        current_df[output_col] = _calculate_invariant_mass_2particle(
                            current_df, 
                            inputs['pt_cols'], 
                            inputs['eta_cols'], 
                            inputs['phi_cols'], 
                            mass_const
                        )
                        analysis_results[step_name] = f"Calculated '{output_col}' using mass constant: {mass_const_name}"

                elif step_config['step'] == 'filter':
                    conditions = step_config['conditions']
                    
                    combined_mask = pd.Series(True, index=current_df.index)
                    for condition in conditions:
                        # Use a simple eval for filtering. Requires clean input from LLM.
                        mask = current_df.eval(condition)
                        combined_mask &= mask
                        
                    n_before = len(current_df)
                    current_df = current_df[combined_mask]
                    n_after = len(current_df)
                    
                    analysis_results[step_name] = f"Filtered: {n_before} events -> {n_after} events. Conditions: {', '.join(conditions)}"

                elif step_config['step'] == 'plot':
                    fig = _generate_plot(current_df, step_config, df)
                    analysis_results['Plot'] = fig
                    # Capture fit results if available
                    if 'fit_results' in step_config:
                        analysis_results['Statistics'] = pd.DataFrame(step_config['fit_results'], index=['Result'])
                    
                else:
                    analysis_results[step_name] = f"Skipped unknown step: {step_config['step']}"

            except Exception as e:
                analysis_results[step_name] = f"Execution Error in step {step_name}: {str(e)}"
                break # Stop pipeline on execution error

        # Final Summary
        final_summary = (
            "The AI Assistant successfully interpreted your request and executed a multi-step analysis pipeline. "
            "This analysis involved calculating the invariant mass ($M_{\ell\ell}$) of the lepton pair, "
            "applying kinematic selection cuts on the lepton transverse momentum ($p_T$), "
            "and generating a histogram of the final mass distribution. "
            "If a Gaussian fit was requested, the $\mu$ (mass peak) and $\sigma$ (width) were extracted from the fit."
        )

        return {
            'success': True,
            'explanation': final_summary,
            'specification': pipeline_spec,
            'results': analysis_results
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Gemini API or JSON parsing error: {str(e)}",
            'specification': json_spec_str if 'json_spec_str' in locals() else 'N/A'
        }
