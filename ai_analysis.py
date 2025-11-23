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
import math
from typing import Dict, Any, List

# the newest Gemini model is "gemini-2.5-flash" or "gemini-2.5-pro"
# do not change this unless explicitly requested by the user
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# This is using Google Gemini's free API (no payment required, just get a free API key from Google AI Studio)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- Available Analysis Functions (Internal) ---

def plot_histogram_internal(df: pd.DataFrame, ax: plt.Axes, plot_type: str, **kwargs) -> Dict[str, Any]:
    """Helper to run the plotting logic based on type."""
    
    # Get the column from kwargs or use a default fallback (M for invariant mass)
    col = kwargs.get('column', 'M_4l')
    
    if plot_type == 'invariant_mass':
        
        # Map common names to the actual column name 'M'
        if col in ['M_4l', 'M', 'm4l']:
            col = 'M' # Use the actual column name for 4-lepton mass
        elif col not in df.columns:
            # Fallback for Z-boson masses if M is missing (though M should exist)
            if 'mZ1' in df.columns and col == 'mZ1':
                col = 'mZ1'
            elif 'mZ2' in df.columns and col == 'mZ2':
                col = 'mZ2'
            else:
                # Column not found, raise an error message
                requested_col = kwargs.get('column', 'M_4l')
                ax.text(0.5, 0.5, f"Column '{requested_col}' not found in DataFrame.", transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Analysis Column Not Found')
                return {'result': f"Column '{requested_col}' not found in DataFrame. Please check the available columns."}
        
        
        # Safely determine min/max, handling potential NaNs in small datasets
        data_series = df[col].dropna()
        if data_series.empty:
            ax.text(0.5, 0.5, 'No data available to plot.', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{col} Distribution (Empty)')
            return {'result': 'No data available to plot for this column.'}
            
        # Set default ranges based on the column type if not provided
        if col == 'M':
            # Typical range for Higgs search
            default_min, default_max = 80, 200
        elif col in ['mZ1', 'mZ2']:
            # Typical range for Z boson mass
            default_min, default_max = 50, 120
        else:
            default_min, default_max = data_series.min() if not math.isnan(data_series.min()) else 0, \
                                       data_series.max() if not math.isnan(data_series.max()) else 1000

        range_min = kwargs.get('range_min', default_min)
        range_max = kwargs.get('range_max', default_max)
        bins = kwargs.get('bins', 50)

        # Ensure min <= max
        if range_min > range_max:
            range_min, range_max = range_max, range_min
            
        # Filter data based on the calculated or requested range
        data = data_series[data_series.between(range_min, range_max)]

        # Handle case where data is empty after filtering
        if data.empty:
            ax.text(0.5, 0.5, 'No events found in the specified range.', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{col} Distribution (Empty in Range)')
            return {'result': 'No events found in the specified mass range after filtering.'}
        
        ax.hist(data, bins=bins, range=(range_min, range_max), 
                edgecolor='black', alpha=0.7, color='steelblue')
        
        # Use descriptive label based on the actual column name
        if col == 'M':
             xlabel = 'Four-lepton invariant mass $M_{4\\ell}$ [GeV/c$^2$]'
             title = 'Four-Lepton Invariant Mass Distribution (H→4ℓ)'
        elif col in ['mZ1', 'mZ2']:
             xlabel = f'Dilepton invariant mass ${col}$ [GeV/c$^2$]'
             title = f'Dilepton Invariant Mass Distribution ({col})'
        else:
             xlabel = f'{col} [GeV/c$^2$]'
             title = f'{col} Distribution'
             
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add Higgs mass line ONLY if plotting the 4-lepton mass (M)
        if col == 'M' and range_min <= 125 <= range_max:
            ax.axvline(125, color='red', linestyle='--', linewidth=2, label='Higgs mass (125 GeV)')
            ax.legend()
        
        return {'result': f'Histogram for column {col} plotted from {range_min} to {range_max} in {bins} bins.'}
    
    elif plot_type == 'momentum':
        pt_col = kwargs.get('column', kwargs.get('pt_col', 'pt1'))
        bins = kwargs.get('bins', 50)
        
        if pt_col not in df.columns:
            ax.text(0.5, 0.5, f"Column '{pt_col}' not found.", transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Momentum Column Not Found')
            return {'result': f"Column '{pt_col}' not found in DataFrame."}

        data = df[pt_col].dropna()
        if data.empty:
            ax.text(0.5, 0.5, 'No momentum data available.', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Transverse Momentum Distribution (Empty)')
            return {'result': 'No momentum data available to plot.'}

        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color='forestgreen')
        ax.set_xlabel(f'Transverse momentum $p_T$ ({pt_col}) [GeV/c]', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(f'Transverse Momentum Distribution ({pt_col})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        return {'result': f'Histogram for transverse momentum {pt_col} plotted.'}
    
    elif plot_type == 'eta_phi':
        eta_col = kwargs.get('eta_col', 'eta1')
        phi_col = kwargs.get('phi_col', 'phi1')
        
        if eta_col not in df.columns or phi_col not in df.columns:
            ax.text(0.5, 0.5, 'Missing $\\eta$/$\\phi$ columns.', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Missing Kinematic Columns')
            return {'result': 'Missing required eta or phi columns for 2D plot.'}

        data_eta = df[eta_col].dropna()
        data_phi = df[phi_col].dropna()
        
        if data_eta.empty or data_phi.empty:
            ax.text(0.5, 0.5, 'No $\\eta$/$\\phi$ data available.', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Eta-Phi Distribution (Empty)')
            return {'result': 'No eta or phi data available to plot.'}

        # 2D Histogram/Scatter
        ax.hist2d(data_phi, data_eta, bins=50, cmap=plt.cm.jet)
        ax.set_xlabel(f'Azimuthal Angle $\\phi$ ({phi_col})', fontsize=12)
        ax.set_ylabel(f'Pseudorapidity $\\eta$ ({eta_col})', fontsize=12)
        ax.set_title('Eta-Phi Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        return {'result': f'2D histogram of {eta_col} vs {phi_col} plotted.'}

    return {'result': f'Unknown plot type: {plot_type}'}

def calculate_mass_internal(df: pd.DataFrame, mass_type: str, cols: List[str]) -> pd.DataFrame:
    """Placeholder for mass calculation. Should be implemented by the user in utils."""
    # In a real app, this would call a physics_utils function.
    # For now, we return a simple mock/passthrough for structured output purposes.
    return df

def filter_data_internal(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> pd.DataFrame:
    """Internal function to filter the DataFrame."""
    if column in df.columns:
        filtered_df = df[df[column].between(min_val, max_val)]
        return filtered_df
    return df # Return original if column not found

# --- Main Analysis Function ---

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
        'head': df.head().to_dict('list')
    }
    
    # Context for the model
    context = (
        "You are an expert Particle Physics Data Analyst. Your task is to convert a user's natural language request "
        "into a structured JSON analysis pipeline. The goal is to safely execute data analysis and plotting "
        "on the provided DataFrame. Use the available functions in sequence (pipeline) to achieve the goal.\n"
        "The current dataset has the following information:\n"
        f"DATA COLUMNS: {data_info['columns']}\n"
        f"DATA SHAPE: {data_info['shape']}\n"
        f"DATA HEAD: {data_info['head']}\n"
        "COMMON COLUMNS (In this Dataset): M (four-lepton mass), mZ1/mZ2 (dilepton mass). Use 'M' for the four-lepton mass.\n"
        "Higgs mass is expected at 125 GeV. If asked to plot the Higgs mass, ensure the range includes 125 GeV, and use column 'M'.\n"
        "CRITICAL: The pipeline must contain at least one step if an analysis/plot is requested."
    )
    
    # 2. Define the output schema for structured JSON response
    # This schema definition is critical for guiding the model and preventing the INVALID_ARGUMENT error.
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "explanation": types.Schema(
                type=types.Type.STRING,
                description="A brief explanation of the planned analysis steps.",
            ),
            "is_safe": types.Schema(
                type=types.Type.BOOLEAN,
                description="True if the request can be safely executed with the available tools, False otherwise. Always True for analysis requests.",
            ),
            "pipeline": types.Schema(
                type=types.Type.ARRAY,
                description="A sequential list of analysis steps to be executed. Must use the available functions below.",
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "function": types.Schema(
                            type=types.Type.STRING,
                            description="The analysis function to call. Must be one of: 'plot_invariant_mass', 'plot_momentum', 'plot_eta_phi', 'filter_data', 'calculate_mass', 'get_stats'.",
                        ),
                        # FIX: Ensure 'args' properties are non-empty for OBJECT type
                        "args": types.Schema(
                            type=types.Type.OBJECT,
                            description="Arguments for the function call. Only include arguments relevant to the function.",
                            properties={
                                # General Plotting/Filtering Args
                                "column": types.Schema(type=types.Type.STRING, description="The primary column name for the operation (e.g., 'M', 'mZ1', 'pt1'). REQUIRED for plots/filters/stats. Use 'M' for the four-lepton invariant mass."),
                                "range_min": types.Schema(type=types.Type.NUMBER, description="Minimum value for plot range or filtering cut (GeV)."),
                                "range_max": types.Schema(type=types.Type.NUMBER, description="Maximum value for plot range or filtering cut (GeV)."),
                                "bins": types.Schema(type=types.Type.INTEGER, description="Number of histogram bins (if plotting)."),

                                # Specific Plotting Args
                                "pt_col": types.Schema(type=types.Type.STRING, description="Transverse momentum column name (e.g., 'pt1'). Used for 'plot_momentum'."),
                                "eta_col": types.Schema(type=types.Type.STRING, description="Pseudorapidity column name (e.g., 'eta1'). Used for 'plot_eta_phi'."),
                                "phi_col": types.Schema(type=types.Type.STRING, description="Azimuthal angle column name (e.g., 'phi1'). Used for 'plot_eta_phi'."),
                            }
                        ),
                    },
                    required=["function", "args"],
                ),
            ),
        },
        required=["explanation", "pipeline", "is_safe"],
    )

    # 3. Call the Gemini API
    try:
        # Construct the full user query with data context
        full_query = f"{context}\n\nUSER REQUEST: {user_prompt}\n\nGenerate the JSON analysis pipeline."
        
        # Implement exponential backoff for API calls
        max_retries = 3
        base_delay = 1.0 # seconds

        for attempt in range(max_retries):
            try:
                # CRITICAL: Force the model to generate the required arguments
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=full_query,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are an expert physicist and data analyst. Generate a structured JSON object containing an analysis pipeline to fulfill the user's request. "
                            "CRITICAL: When plotting four-lepton invariant mass, ALWAYS use the function 'plot_invariant_mass' and explicitly include the arguments: "
                            "'column': 'M', 'range_min', 'range_max', and 'bins' with the values requested by the user. "
                            "Do not output any text outside the JSON object. Ensure the JSON is perfectly valid."
                        ),
                        response_mime_type="application/json",
                        response_schema=response_schema
                    ),
                )
                break  # Success! Exit the loop.
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    # Suppress logging retries to console as requested
                    plt.pause(delay) # Use plt.pause for non-blocking delay in Streamlit context
                else:
                    raise e # Re-raise the exception on the last attempt


        # 4. Parse the structured JSON response
        try:
            # The model returns the JSON string in response.text
            analysis_plan = json.loads(response.text)
        except json.JSONDecodeError:
            # Improved error logging to capture the bad JSON
            print(f"--- JSON DECODE ERROR ---")
            print(f"Raw AI Response: {response.text[:500]}...")
            print(f"-------------------------")
            return {'success': False, 'error': f"AI returned malformed JSON: Check console for raw response."}
        
        # 5. Execute the pipeline
        pipeline_results = []
        current_df = df.copy() # Operate on a copy
        
        for step in analysis_plan.get('pipeline', []):
            function_name = step.get('function')
            args = step.get('args', {})
            
            # --- Execution logic for each function ---
            if function_name in ['plot_invariant_mass', 'plot_momentum', 'plot_eta_phi']:
                # Plotting step
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Map the function name to the internal plot_type
                plot_type_map = {
                    'plot_invariant_mass': 'invariant_mass',
                    'plot_momentum': 'momentum',
                    'plot_eta_phi': 'eta_phi',
                }
                
                # IMPORTANT: We need to pass the figure object back for Streamlit to display.
                # The plot_histogram_internal function modifies the 'ax' object in place.
                result = plot_histogram_internal(current_df, ax, plot_type_map[function_name], **args)
                
                pipeline_results.append({
                    'function': function_name,
                    'result': result['result'],
                    'plot': fig # Pass the matplotlib figure for Streamlit to display
                })
                
                plt.close(fig) # Close the figure to free memory
                
            elif function_name == 'filter_data':
                # Filtering step
                column = args.get('column')
                min_val = args.get('range_min')
                max_val = args.get('range_max')
                
                if column and min_val is not None and max_val is not None:
                    count_before = len(current_df)
                    current_df = filter_data_internal(current_df, column, min_val, max_val)
                    count_after = len(current_df)
                    result_msg = f"Filtered {column} between {min_val} and {max_val}. Events reduced from {count_before} to {count_after}."
                else:
                    result_msg = "Filter operation skipped due to missing arguments."
                    
                pipeline_results.append({'function': function_name, 'result': result_msg})
                
            elif function_name == 'calculate_mass':
                 # Mass calculation step (Requires physics_utils integration)
                 # Placeholder for the actual calculation logic
                 result_msg = f"Mass calculation for type {args.get('mass_type')} is planned but not executed in this simplified model."
                 pipeline_results.append({'function': function_name, 'result': result_msg})

            elif function_name == 'get_stats':
                # Basic statistics step
                col = args.get('column')
                if col and col in current_df.columns:
                    stats = current_df[col].describe().to_dict()
                    result_msg = f"Statistics for {col}: {json.dumps(stats, indent=2)}"
                else:
                    result_msg = "Statistics calculation skipped due to missing or invalid column."
                pipeline_results.append({'function': function_name, 'result': result_msg})

            else:
                pipeline_results.append({'function': function_name, 'result': f"Unknown function: {function_name}"})

        
        # 6. Collate the final result
        return {
            'success': True,
            'explanation': analysis_plan.get('explanation', 'Analysis pipeline executed.'),
            'results': pipeline_results,
            'final_df_shape': current_df.shape
        }

    except Exception as e:
        return {'success': False, 'error': f"AI Analysis Failed: {type(e).__name__}: {str(e)}"}

if __name__ == '__main__':
    # This block is for local testing only
    print("AI Analysis Assistant Module Loaded.")
