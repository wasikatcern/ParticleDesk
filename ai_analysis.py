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
    
    # Check if the column exists in the DataFrame
    if col not in df.columns:
        return {'success': False, 'error': f"Plotting failed: Column '{col}' not found in the DataFrame."}
    
    data = df[col].dropna()

    if plot_type == 'invariant_mass':
        # Safely determine min/max for the plot range
        data_min = data.min() if not data.empty else 0
        data_max = data.max() if not data.empty else 200
        
        range_min = kwargs.get('range_min', math.floor(data_min))
        range_max = kwargs.get('range_max', math.ceil(data_max))
        bins = kwargs.get('bins', 50)
        
        # Ensure plot range is sensible
        range_min = max(0, range_min)
        range_max = max(range_max, range_min + 1)
        
        ax.hist(data, bins=bins, range=(range_min, range_max), 
                edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(f'{col} [GeV/c²]', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(f'Invariant Mass Distribution of {col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add Higgs mass line if relevant and in range
        if range_min <= 125 <= range_max:
            ax.axvline(125, color='red', linestyle='--', linewidth=2, label='Higgs mass (125 GeV/c²)')
            ax.legend()
    
    elif plot_type == 'momentum':
        pt_col = kwargs.get('pt_col', 'pt1')
        bins = kwargs.get('bins', 50)
        
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color='forestgreen')
        ax.set_xlabel('Transverse momentum $p_T$ [GeV/c]', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(f'Transverse Momentum Distribution of {pt_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    elif plot_type == 'eta_phi':
        eta_col = kwargs.get('eta_col', 'eta1')
        phi_col = kwargs.get('phi_col', 'phi1')
        
        if eta_col not in df.columns or phi_col not in df.columns:
            return {'success': False, 'error': f"Plotting failed: One of the columns ('{eta_col}' or '{phi_col}') is missing for eta-phi plot."}
            
        # Create a 2D histogram (heatmap)
        hist, xedges, yedges = np.histogram2d(
            df[eta_col].dropna(), 
            df[phi_col].dropna(), 
            bins=kwargs.get('bins', 50),
            range=[[-5, 5], [-math.pi, math.pi]] # Typical ranges for eta and phi
        )
        
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        im = ax.pcolormesh(X, Y, hist.T, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Event Density')
        
        ax.set_xlabel(f'Pseudorapidity ($\eta$) of {eta_col}', fontsize=12)
        ax.set_ylabel(f'Azimuthal Angle ($\phi$) of {phi_col}', fontsize=12)
        ax.set_title(f'Eta-Phi Occupancy for {eta_col}/{phi_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    else:
        # Default histogram for any other numerical column
        bins = kwargs.get('bins', 50)
        
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color='darkblue')
        ax.set_xlabel(f'{col}', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    return {'success': True, 'plot_type': plot_type, 'column': col}

def apply_filter_internal(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Apply a string-based filter to the DataFrame."""
    try:
        # Evaluate the filter condition safely using pd.eval
        # WARNING: This assumes the condition string is safe and only uses df columns.
        # In a real-world scenario, you would need robust sanitization/parsing.
        mask = df.eval(condition)
        return df[mask]
    except Exception as e:
        # Return original DataFrame on failure
        print(f"Filter failed: {e}")
        return df

def get_stats_internal(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Calculate descriptive statistics for a column."""
    if column in df.columns:
        stats = df[column].describe().to_dict()
        return stats
    return {"error": f"Column '{column}' not found for statistics."}


# --- Gemini Interaction and Analysis Execution ---

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
    }
    
    # Define the functions the model can call
    tool_functions = [
        types.Schema(
            name="filter_data",
            description="Filters the DataFrame based on a boolean condition (e.g., 'pt1 > 20' or 'charge1 + charge2 == 0'). This must be the first step for selection/analysis.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "condition": types.Schema(type=types.Type.STRING, description="The Pandas eval-compatible boolean condition string.")
                },
                required=["condition"]
            ),
        ),
        types.Schema(
            name="calculate_invariant_mass",
            description="Calculates the invariant mass of a system (e.g., 4-lepton or 2-lepton) and adds it as a new column to the DataFrame, usually named 'M' or 'M_ll'. This is a conceptual placeholder as the actual calculation needs detailed column names.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "mass_type": types.Schema(type=types.Type.STRING, description="The type of invariant mass ('4l', '2l', etc.)")
                },
                required=["mass_type"]
            ),
        ),
        types.Schema(
            name="plot_histogram",
            description="Generates a histogram plot for visualization.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "plot_type": types.Schema(type=types.Type.STRING, description="The type of plot ('invariant_mass', 'momentum', 'eta_phi', or 'general')."),
                    "column": types.Schema(type=types.Type.STRING, description="The specific column name to plot (e.g., 'M_4l', 'pt1').")
                },
                required=["plot_type", "column"]
            ),
        ),
        types.Schema(
            name="get_stats",
            description="Calculates descriptive statistics (mean, std, min, max, count) for a specific numerical column.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "column": types.Schema(type=types.Type.STRING, description="The column name for which to calculate statistics.")
                },
                required=["column"]
            ),
        ),
    ]

    # --- System Instruction for the Model ---
    system_instruction = f"""
    You are an expert particle physics data analysis assistant. Your task is to interpret a user's request, analyze the provided data structure, and generate a JSON object representing a complete, multi-step analysis plan.

    The user is working with a pandas DataFrame that has the following structure:
    {json.dumps(data_info, indent=2)}

    Your response MUST be a single JSON object that follows this schema:
    {{
        "explanation": "A concise, step-by-step natural language explanation of the analysis plan before execution.",
        "pipeline": [
            {{
                "function": "function_name",
                "args": {{...}}
            }},
            // More function calls if needed
        ]
    }}

    Rules for creating the pipeline:
    1.  Only use the functions defined in the `tool_functions` list.
    2.  The pipeline MUST be ordered logically: `filter_data` always comes before `plot_histogram` or `get_stats` if a cut is requested.
    3.  If a complex calculation like invariant mass is required, include the `calculate_invariant_mass` call, but you must assume the helper function will handle the actual calculation and column creation based on context.
    4.  Provide specific, existing column names in the `args` (e.g., 'M_4l', 'pt1').
    5.  Do not hallucinate function calls or arguments.
    6.  If the user asks for a plot, the final step in the pipeline should be `plot_histogram`.
    7.  Keep the pipeline concise (2-4 steps max).
    """

    # --- Call the Gemini API for tool calling ---
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "explanation": types.Schema(type=types.Type.STRING),
                        "pipeline": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "function": types.Schema(type=types.Type.STRING),
                                    "args": types.Schema(type=types.Type.OBJECT)
                                },
                                required=["function", "args"]
                            )
                        )
                    },
                    required=["explanation", "pipeline"]
                )
            )
        )
        
        # Parse the JSON response from the model
        analysis_plan = json.loads(response.text)
        
    except Exception as e:
        return {'success': False, 'error': f"Gemini API call or JSON parsing failed: {str(e)}"}

    # --- Execute the Analysis Pipeline ---
    current_df = df.copy()
    pipeline_results = []
    plot_info = None
    
    try:
        # 5. Iterate through the generated pipeline
        for step in analysis_plan.get('pipeline', []):
            function_name = step.get('function')
            args = step.get('args', {})
            
            # This is where the model's intent is executed by internal functions
            if function_name == 'filter_data':
                condition = args.get('condition')
                original_shape = current_df.shape
                current_df = apply_filter_internal(current_df, condition)
                
                result_msg = (
                    f"Applied filter: `{condition}`. "
                    f"Shape changed from {original_shape} to {current_df.shape}. "
                    f"Events remaining: {current_df.shape[0]}"
                )
                pipeline_results.append({'function': function_name, 'result': result_msg})

            elif function_name == 'plot_histogram':
                # This function call prepares the final plot and returns its configuration
                # The actual plotting will be done in the main Streamlit app based on `plot_info`
                plot_info = args
                plot_info['df'] = current_df.copy() # Save the filtered DataFrame for plotting
                result_msg = f"Plot configured: {plot_info.get('plot_type')} for column {plot_info.get('column')}. Plot will be rendered next."
                pipeline_results.append({'function': function_name, 'result': result_msg})

            elif function_name == 'calculate_invariant_mass':
                 # Placeholder for the actual calculation logic
                 # NOTE: In a complete system, this would call functions from physics_utils.py
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
            'final_df_shape': current_df.shape,
            'plot_info': plot_info # Include plot configuration for the main app
        }

    except Exception as e:
        return {'success': False, 'error': f"AI Analysis Failed: {type(e).__name__}: {str(e)}"}

if __name__ == '__main__':
    # This block is for local testing only
    print("AI Analysis Assistant Module Loaded.")
