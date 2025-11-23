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
import time
from typing import List, Dict, Any


# --- Configuration for Gemini API ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# This is using Google Gemini's free API (no payment required, just get a free API key from Google AI Studio)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Define constants for analysis
MUON_MASS = 0.105658  # GeV/c^2
ELECTRON_MASS = 0.000511  # GeV/c^2
HIGGS_MASS_EXPECTED = 125.0  # GeV/c^2

# --- Custom Analysis Helper Functions (Mock implementations for single-file runnability) ---
# NOTE: In a real environment, these would be imported from the physics_utils module
def calculate_invariant_mass_2particle(df, pt1_col, eta1_col, phi1_col, pt2_col, eta2_col, phi2_col, mass1, mass2):
    """Calculate the invariant mass of a two-particle system."""
    E1 = np.sqrt(df[pt1_col]**2 * np.cosh(df[eta1_col])**2 + mass1**2)
    E2 = np.sqrt(df[pt2_col]**2 * np.cosh(df[eta2_col])**2 + mass2**2)
    Px1 = df[pt1_col] * np.cos(df[phi1_col])
    Py1 = df[pt1_col] * np.sin(df[phi1_col])
    Pz1 = df[pt1_col] * np.sinh(df[eta1_col])
    Px2 = df[pt2_col] * np.cos(df[phi2_col])
    Py2 = df[pt2_col] * np.sin(df[phi2_col])
    Pz2 = df[pt2_col] * np.sinh(df[eta2_col])
    
    E_tot = E1 + E2
    Px_tot = Px1 + Px2
    Py_tot = Py1 + Py2
    Pz_tot = Pz1 + Pz2
    
    # M^2 = E^2 - P^2
    M_sq = E_tot**2 - Px_tot**2 - Py_tot**2 - Pz_tot**2
    # Handle potential negative M_sq due to precision limits
    return np.sqrt(np.maximum(0, M_sq))

def calculate_invariant_mass_4lepton(df, pt_cols, eta_cols, phi_cols, mass_cols):
    """Calculate the invariant mass of a four-lepton system."""
    px_total, py_total, pz_total, e_total = 0, 0, 0, 0
    
    for pt_col, eta_col, phi_col, mass_col in zip(pt_cols, eta_cols, phi_cols, mass_cols):
        pt = df[pt_col]
        eta = df[eta_col]
        phi = df[phi_col]
        mass = df[mass_col]
        
        e = np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)

        e_total += e
        px_total += px
        py_total += py
        pz_total += pz

    # M^2 = E^2 - P^2
    M_sq = e_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    return np.sqrt(np.maximum(0, M_sq))

def calculate_transverse_mass(df, pt1_col, phi1_col, met_col, met_phi_col):
    """Calculate transverse mass (M_T) for a system with missing energy (MET)."""
    pt1 = df[pt1_col]
    phi1 = df[phi1_col]
    met = df[met_col]
    met_phi = df[met_phi_col]
    
    dphi = phi1 - met_phi
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi)) # Wrap to [-pi, pi]
    
    mt_sq = 2 * pt1 * met * (1 - np.cos(dphi))
    return np.sqrt(np.maximum(0, mt_sq))

def delta_r(df, eta1_col, phi1_col, eta2_col, phi2_col):
    """Calculate the angular separation Delta R."""
    deta = df[eta1_col] - df[eta2_col]
    dphi = df[phi1_col] - df[phi2_col]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return np.sqrt(deta**2 + dphi**2)
# -----------------------------------------------------------------------------------


# --- Core LLM Prompting and Function Calling ---

# System Instruction: Defines the AI's role and its capabilities
SYSTEM_INSTRUCTION = """
You are a world-class Particle Physics Analysis Assistant. You are an expert in Standard Model phenomenology, data analysis, and statistical data analysis. Your tone must be professional, conversational, and highly transparent.

Your task is to translate a user's natural language request into a precise, declarative JSON specification for a data analysis pipeline.

The user provides particle collision event data in a pandas DataFrame called `df`.
Your response MUST be a single JSON object that adheres strictly to the provided JSON Schema.
DO NOT include any explanation, narrative, or text outside of the JSON object.

# Available Data
- The input data is in a pandas DataFrame named `df`.
- The current columns are: {columns}.
- Total events: {shape}.

# Response Requirements:
1.  **Explanation**: A concise, single-paragraph explanation of the physics analysis being performed (e.g., "We are searching for the Higgs boson decay H -> 4l by examining the four-lepton invariant mass spectrum.").
2.  **Status Report**: A list of detailed, conversational messages (`status_report`) that an assistant would say during the analysis. Use physics terminology correctly. If a user request makes no sense (e.g., trying to calculate mass from non-existent columns), the first entry MUST be a helpful error and the pipeline should be empty.

# Analysis Pipeline Structure (JSON Schema)
The analysis must be specified as an array of sequential steps. Each step must be one of the following types: 'calculate_variable', 'filter_events', 'plot_histogram', 'perform_fit', 'report_statistics'.

# IMPORTANT CONSTRAINTS for 'calculate_variable' step:
- You MUST use the provided helper functions from the `physics_utils` module for complex calculations:
    - `physics_utils.calculate_invariant_mass_2particle(df, pt1_col, eta1_col, phi1_col, pt2_col, eta2_col, phi2_col, mass1, mass2)`
    - `physics_utils.calculate_invariant_mass_4lepton(df, pt_cols, eta_cols, phi_cols, mass_cols)`
    - `physics_utils.calculate_transverse_mass(df, pt1_col, phi1_col, met_col, met_phi_col)`
    - `physics_utils.delta_r(df, eta1_col, phi1_col, eta2_col, phi2_col)`
- Constants for masses are globally available: `MUON_MASS` ({MUON_MASS}), `ELECTRON_MASS` ({ELECTRON_MASS}), `HIGGS_MASS_EXPECTED` ({HIGGS_MASS_EXPECTED}).
- For simple column-wise arithmetic, use the `formula` field.

# IMPORTANT CONSTRAINTS for 'filter_events' step:
- The `condition` must be a valid pandas query string (e.g., `'lep1_pt > 25'`).

# IMPORTANT CONSTRAINTS for 'plot_histogram' step:
- The `column` MUST be a single, existing column name.
- Provide appropriate `range_min` and `range_max` based on the physics context (e.g., 80 to 100 GeV for a Z boson peak).

# EXAMPLE JSON SCHEMA (Focus on new 'status_report' field):
{{
    "explanation": "We are searching for the Z boson peak by computing the invariant mass of the two leading leptons and examining its spectrum.",
    "status_report": [
        "Starting the analysis to find the Z boson candidate...",
        "Calculated the dilepton invariant mass (M_ll) for all events.",
        "Applying a pT cut of 20 GeV on the leading lepton to suppress background."
    ],
    "pipeline": [...]
}}
"""


def get_schema(df):
    """Dynamically generate the JSON schema for the LLM response."""
    return {
        "type": "OBJECT",
        "properties": {
            "explanation": {"type": "STRING", "description": "A single paragraph explaining the analysis."},
            "status_report": {"type": "ARRAY", "description": "A list of conversational messages detailing the analysis steps and reasoning.", "items": {"type": "STRING"}},
            "pipeline": {
                "type": "ARRAY",
                "description": "A sequential list of analysis steps.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "step_type": {"type": "STRING", "enum": ["calculate_variable", "filter_events", "plot_histogram", "perform_fit", "report_statistics"]},
                        "new_column": {"type": "STRING", "description": "Required for 'calculate_variable'"},
                        "description": {"type": "STRING", "description": "A short description of the step."},
                        "function": {"type": "STRING", "description": "Function name from physics_utils (e.g., 'calculate_invariant_mass_2particle'). Required if not using simple formula."},
                        "args": {"type": "OBJECT", "description": "Keyword arguments for the function."},
                        "formula": {"type": "STRING", "description": "Pandas formula string (e.g., 'df[\"A\"] + df[\"B\"]')."},
                        "condition": {"type": "STRING", "description": "Pandas query string. Required for 'filter_events'."},
                        "column": {"type": "STRING", "description": "Column name for plot/fit/stats. Required for 'plot_histogram', 'perform_fit', 'report_statistics'."},
                        "range_min": {"type": "NUMBER", "description": "Minimum value for plot/fit."},
                        "range_max": {"type": "NUMBER", "description": "Maximum value for plot/fit."},
                        "bins": {"type": "INTEGER", "description": "Number of bins for plot."},
                        "fit_range_min": {"type": "NUMBER", "description": "Min range for Gaussian fit."},
                        "fit_range_max": {"type": "NUMBER", "description": "Max range for Gaussian fit."},
                    },
                    "required": ["step_type", "description"]
                }
            }
        },
        "required": ["explanation", "status_report", "pipeline"]
    }


def execute_pipeline(specification: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes the analysis pipeline defined in the JSON specification.
    """
    results: Dict[str, Any] = {
        'status_report': specification.get('status_report', ["💡 Starting execution based on AI plan."]),
        'technical_results': {} # Holds plots and stats
    }
    current_df = df.copy()
    
    # Define a namespace for safe function execution
    namespace = {
        'calculate_invariant_mass_2particle': calculate_invariant_mass_2particle, 
        'calculate_invariant_mass_4lepton': calculate_invariant_mass_4lepton,
        'calculate_transverse_mass': calculate_transverse_mass,
        'delta_r': delta_r,
        'pd': pd, 
        'np': np, 
        'MUON_MASS': MUON_MASS,
        'ELECTRON_MASS': ELECTRON_MASS,
        'HIGGS_MASS_EXPECTED': HIGGS_MASS_EXPECTED
    }
    
    # Start the execution log
    results['status_report'].append(f"💡 Execution Log: Initial Events = {len(current_df)}")
    
    for i, step in enumerate(specification.get('pipeline', [])):
        step_type = step['step_type']
        
        try:
            if step_type == 'calculate_variable':
                new_column = step['new_column']
                if 'function' in step:
                    # Execute helper function (mocked above)
                    func_name = step['function'].split('.')[-1] 
                    func = namespace.get(func_name)
                    args = step.get('args', {})
                    
                    if func and func_name in ['calculate_invariant_mass_2particle', 'calculate_invariant_mass_4lepton', 'calculate_transverse_mass', 'delta_r']:
                        # Pass df as first argument
                        current_df[new_column] = func(current_df, **args)
                        results['status_report'].append(f"✅ Step {i+1} (Calculation): Successfully derived the kinematic variable '{new_column}' using {func_name}.")
                    else:
                        results['status_report'].append(f"⚠️ Step {i+1} (Calculation): Failed to execute function {func_name}. Function not found or not in allowed list. Please check the pipeline specification.")
                        
                elif 'formula' in step:
                    # Execute simple pandas formula
                    formula = step['formula']
                    # Using eval is generally risky, but scoped here. It expects an expression that results in a Series.
                    current_df[new_column] = eval(formula, {"__builtins__": None}, {'df': current_df, **namespace})
                    results['status_report'].append(f"✅ Step {i+1} (Calculation): Calculated '{new_column}' using formula: `{formula}`.")

            elif step_type == 'filter_events':
                condition = step['condition']
                initial_count = len(current_df)
                current_df = current_df.query(condition)
                final_count = len(current_df)
                
                if final_count > 0:
                    results['status_report'].append(f"✅ Step {i+1} (Selection): Applied physics selection '{condition}'. Retained {final_count} events ({(final_count/initial_count*100):.2f}% efficiency).")
                else:
                    results['status_report'].append(f"🛑 Step {i+1} (Selection): Filter by '{condition}' resulted in 0 events. Analysis halted. This cut is too aggressive or data is sparse.")
                    break # Stop pipeline if no data remains

            elif step_type == 'plot_histogram':
                column = step['column']
                if column not in current_df.columns:
                    raise KeyError(f"Column '{column}' does not exist in the DataFrame after prior steps.")
                    
                data = current_df[column].dropna()
                if len(data) == 0:
                    raise ValueError(f"No data available for plotting column '{column}'.")
                
                range_min = step.get('range_min', data.min())
                range_max = step.get('range_max', data.max())
                bins = step.get('bins', 50)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(data, bins=bins, range=(range_min, range_max), edgecolor='black', alpha=0.8, color='#0a4c9b')
                ax.set_title(step['description'], fontweight='bold')
                ax.set_xlabel(f"{column} [GeV/c²]", fontsize=12) 
                ax.set_ylabel("Events / Bin", fontsize=12)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Check for Higgs/Z mass line guidance
                if 'mass' in column.lower() and HIGGS_MASS_EXPECTED >= range_min and HIGGS_MASS_EXPECTED <= range_max:
                    ax.axvline(HIGGS_MASS_EXPECTED, color='red', linestyle=':', linewidth=2, label=f'Higgs Mass ({HIGGS_MASS_EXPECTED} GeV)')
                    ax.legend()

                # Store the plot object
                results['technical_results']['Plot'] = fig
                results['status_report'].append(f"✅ Step {i+1} (Visualization): Generated histogram for the distribution of '{column}' in the range [{range_min:.2f}, {range_max:.2f}] GeV. This plot is now available in the results section.")
                plt.close(fig) # Close figure after storing to prevent memory leak

            elif step_type == 'perform_fit':
                column = step['column']
                if column not in current_df.columns:
                    raise KeyError(f"Column '{column}' does not exist for fitting.")
                
                fit_min = step.get('fit_range_min', current_df[column].min())
                fit_max = step.get('fit_range_max', current_df[column].max())
                fit_data = current_df[(current_df[column] >= fit_min) & (current_df[column] <= fit_max)][column].dropna()
                
                if len(fit_data) > 50:
                    from scipy.optimize import curve_fit
                    from scipy.stats import norm
                    
                    hist_values, bin_edges = np.histogram(fit_data, bins=50, range=(fit_min, fit_max))
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    def gaussian(x, amplitude, mean, stddev):
                        return amplitude * norm.pdf(x, mean, stddev)
                    
                    mu_guess = bin_centers[np.argmax(hist_values)]
                    sigma_guess = np.std(fit_data) or 1.0 
                    amplitude_guess = np.max(hist_values) * sigma_guess * np.sqrt(2 * np.pi) 
                    
                    popt, pcov = curve_fit(gaussian, bin_centers, hist_values, p0=[amplitude_guess, mu_guess, sigma_guess])
                    fit_mean, fit_sigma = popt[1], abs(popt[2])
                    fit_mean_err = np.sqrt(np.diag(pcov))[1]
                    
                    fit_stats = pd.DataFrame({
                        'Parameter': [f'Fit Mean (Mass) on {column}', f'Fit Sigma (Resolution) on {column}'],
                        'Value': [fit_mean, fit_sigma],
                        'Uncertainty': [fit_mean_err, np.sqrt(np.diag(pcov))[2]]
                    })
                    results['technical_results']['Statistics'] = pd.concat([results['technical_results'].get('Statistics', pd.DataFrame()), fit_stats], ignore_index=True)
                    
                    # Add fit line to the plot if available
                    if 'Plot' in results['technical_results']:
                        fig = results['technical_results']['Plot']
                        ax = fig.get_axes()[0]
                        x_fit = np.linspace(fit_min, fit_max, 100)
                        ax.plot(x_fit, gaussian(x_fit, *popt), 'r--', linewidth=2, label='Gaussian Fit Result')
                        ax.legend()
                        results['status_report'].append(f"✅ Step {i+1} (Statistical Fit): Successfully performed Gaussian fit. The resulting peak mass is centered at **{fit_mean:.2f} ± {fit_mean_err:.2f} GeV/c²**.")
                    else:
                        results['status_report'].append(f"✅ Step {i+1} (Statistical Fit): Performed Gaussian fit on '{column}'. Results added to statistics table.")
                else:
                    results['status_report'].append(f"🛑 Step {i+1} (Statistical Fit): Fit skipped for '{column}'. Only {len(fit_data)} events found in the fitting range [{fit_min}, {fit_max}]. Need >50 events for reliable fitting.")

            elif step_type == 'report_statistics':
                column = step['column']
                stats = current_df[column].describe().to_frame().reset_index()
                stats.columns = ['Statistic', 'Value']
                stats['Value'] = stats['Value'].round(4)
                results['technical_results']['Statistics'] = pd.concat([results['technical_results'].get('Statistics', pd.DataFrame()), stats], ignore_index=True)
                results['status_report'].append(f"✅ Step {i+1} (Reporting): Generated descriptive statistics for the final variable '{column}'.")
        
        except Exception as e:
            error_message = f"❌ Step {i+1}: Analysis halted due to a critical execution error during step '{step_type}'. Detail: {str(e)}."
            results['status_report'].append(error_message)
            results['technical_results']['ExecutionError'] = error_message
            break # Stop pipeline on any execution error
    
    # Final status update
    if 'ExecutionError' not in results['technical_results'] and len(specification.get('pipeline', [])) > 0:
        results['status_report'].append(f"🎉 Analysis pipeline finished successfully. Final event count: {len(current_df)}.")
    elif 'ExecutionError' not in results['technical_results'] and len(specification.get('pipeline', [])) == 0:
         results['status_report'].append(f"⚠️ The analysis request was valid, but the AI did not generate any steps. This often happens if the request is too vague or doesn't map to a clear analysis action.")
    
    return results

def analyze_with_ai(user_prompt: str, df: pd.DataFrame) -> dict:
    """
    Analyze particle physics data based on natural language prompt.
    Uses a safe declarative approach instead of executing arbitrary code.
    """
    
    if not gemini_client:
        return {
            'success': False,
            'error': 'Gemini API key not configured. Get a free API key from https://ai.google.dev/'
        }
    
    # 1. Prepare system instruction with dynamic data info
    data_info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
    }
    
    populated_system_instruction = SYSTEM_INSTRUCTION.format(
        columns=json.dumps(data_info['columns']), 
        shape=data_info['shape'],
        MUON_MASS=MUON_MASS,
        ELECTRON_MASS=ELECTRON_MASS,
        HIGGS_MASS_EXPECTED=HIGGS_MASS_EXPECTED
    )

    # 2. Configure the API call
    MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
    schema = get_schema(df)
    
    payload = {
        'contents': [{ 'parts': [{ 'text': user_prompt }] }],
        'config': types.GenerateContentConfig(
            system_instruction=populated_system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            tools=[{"google_search": {}}], 
        )
    }

    
    # 3. Make the API Call with Exponential Backoff
    max_retries = 3
    delay = 1
    
    for i in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=payload['contents'],
                config=payload['config']
            )

            # 4. Process the Response
            json_text = response.text
            
            # Use a robust check for the JSON starting point
            if json_text.strip().startswith('{'):
                try:
                    specification = json.loads(json_text)
                    
                    # 5. Execute the Analysis Pipeline
                    analysis_results = execute_pipeline(specification, df)
                    
                    final_result = {
                        'success': True,
                        'explanation': specification.get('explanation', "No explanation provided."),
                        'status_report': analysis_results.get('status_report', []),
                        'specification': specification,
                        'results': analysis_results['technical_results']
                    }
                    return final_result
                
                except json.JSONDecodeError as e:
                    # Provide specific error details for the user
                    error_lines = json_text.splitlines()
                    error_line_num = e.lineno
                    error_line_preview = error_lines[error_line_num - 1].strip() if 0 < error_line_num <= len(error_lines) else "N/A"
                    
                    return {
                        'success': False, 
                        'error': f"LLM returned invalid JSON. We received malformed data that could not be parsed. Error: {str(e)} near line {error_line_num}. Malformed Line Preview: '{error_line_preview}'",
                        'status_report': [
                            "🛑 Critical Error: The AI assistant's response was malformed and could not be interpreted.",
                            f"The JSON parsing failed near line {error_line_num}. This is an internal issue with the AI model output format. Please try again or rephrase your request.",
                            f"Detail: {str(e)}"
                        ]
                    }
                except Exception as e:
                    return {
                        'success': False, 
                        'error': f"Internal error during pipeline execution setup: {str(e)}",
                        'status_report': [
                            "🛑 Critical Error: An unexpected error occurred before or during the execution of the pipeline. Detail: " + str(e)
                        ]
                    }
            else:
                 # Case where the LLM returned non-JSON text entirely
                return {
                    'success': False,
                    'error': f"LLM returned text that was not JSON. Preview: '{json_text[:100]}...'",
                    'status_report': [
                        "🛑 Critical Error: The AI assistant failed to produce the required analysis specification (JSON).",
                        "Instead, it returned unexpected raw text. This is a model compliance failure. Please rephrase your request, or try restarting the process."
                    ]
                }

        except Exception as e:
            if i < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                return {
                    'success': False, 
                    'error': f"Failed to call Gemini API after {max_retries} attempts: {str(e)}",
                    'status_report': [
                        "🛑 Critical Error: Failed to communicate with the Gemini API. This may be due to network issues or API quota limits."
                    ]
                }
    
    return {'success': False, 'error': "Reached end of API call function without successful result."}
