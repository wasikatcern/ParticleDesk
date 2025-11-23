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

# --- Internal Utility for Plotting ---

def plot_histogram_to_base64(df: pd.DataFrame, column: str = 'M_4l', bins: int = 50, title: str = None) -> str:
    """Generates a histogram plot and returns it as a base64 encoded string."""
    try:
        if column not in df.columns:
            return f"Error: Column '{column}' not found in the data."

        # Filter out NaNs for plotting
        data = df[column].dropna()

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Determine the plot type based on the column name for specialized coloring/labels
        plot_type = 'invariant_mass' if 'M_' in column or 'mass' in column.lower() else 'generic'

        if plot_type == 'invariant_mass' and column in df.columns:
            # Special handling for mass plots (e.g., Higgs mass line)
            range_min = data.min()
            range_max = data.max()
            ax.hist(data, bins=bins, range=(range_min, range_max), 
                    edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel(f'{title or column} [GeV/c²]', fontsize=12)
            
            # Add Higgs mass line if in range
            if range_min <= 125 <= range_max:
                ax.axvline(125, color='red', linestyle='--', linewidth=2, label='Higgs mass (125 GeV)')
                ax.legend()
        
        elif plot_type == 'momentum':
            ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color='forestgreen')
            ax.set_xlabel(f'{title or column} [GeV/c]', fontsize=12)

        else: # Generic plot
            ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color='gray')
            ax.set_xlabel(f'{title or column}', fontsize=12)


        ax.set_ylabel('Events', fontsize=12)
        ax.set_title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save to a base64 string
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig) # Close the figure to free memory
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        return f"Plotting Error: {str(e)}"

# --- Function Definitions for Gemini Schema (Structured Output) ---

def get_data_metadata(df: pd.DataFrame = None) -> None:
    """
    Retrieves and summarizes key metadata about the DataFrame, including the number of rows (events), 
    number of columns (variables), and a list of all column names.
    """
    pass # Actual execution happens in analyze_with_ai

def get_stats(column: str, df: pd.DataFrame = None) -> None:
    """
    Calculates descriptive statistics (mean, median, std, min, max, count) for a specified numerical column.
    The analysis should only request this function for numerical columns in the data.
    """
    pass # Actual execution happens in analyze_with_ai

def plot_histogram(column: str, bins: int = 50, title: str = None) -> None:
    """
    Generates a histogram for a numerical column, useful for visualizing distributions like 
    invariant mass, transverse momentum, or energy.
    """
    pass # Actual execution happens in analyze_with_ai

def calculate_invariant_mass(mass_type: str, lepton_cols: List[str] = None) -> None:
    """
    (Placeholder) Plans the calculation of the invariant mass for a system of particles 
    (e.g., 2-lepton, 4-lepton). Supported mass_types: '2l', '4l'.
    The execution is a placeholder in this simplified model.
    """
    pass # Actual execution happens in analyze_with_ai


# --- Define the overall response schema for the analysis plan ---

RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        'explanation': types.Schema(
            type=types.Type.STRING,
            description="A conversational and educational explanation of the user's request and how the analysis plan will address it. This should be the first part of the final output."
        ),
        'pipeline': types.Schema(
            type=types.Type.ARRAY,
            description="A list of analysis steps to execute.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    'function': types.Schema(
                        type=types.Type.STRING,
                        description="The name of the function to call (e.g., 'plot_histogram')."
                    ),
                    'args': types.Schema(
                        type=types.Type.OBJECT,
                        description="A dictionary of arguments for the function."
                    )
                },
                required=['function', 'args']
            )
        )
    },
    required=['explanation', 'pipeline']
)


# --- Core Analysis Function ---

def analyze_with_ai(user_prompt: str, df: pd.DataFrame) -> dict:
    """
    Analyze particle physics data based on natural language prompt.
    Uses a safe declarative approach by forcing a structured JSON plan, which is then executed.
    
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
    
    # 1. Get data info for the model's context
    data_info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
        'head': df.head(2).to_json(orient='records'),
    }

    system_prompt = f"""
    You are an expert Particle Physics Data Analyst. Your task is to analyze user requests
    in the context of the provided data, which is a pandas DataFrame.
    
    The available DataFrame context is:
    {json.dumps(data_info, indent=2)}

    Your response MUST be a single JSON object strictly following the RESPONSE_SCHEMA.
    
    1. **Explanation**: Start with an educational and conversational explanation of the analysis plan. If the user only asks for data properties (like columns or size), you MUST answer in the 'explanation' and use the 'get_data_metadata' function in the pipeline.
    2. **Pipeline**: Generate a 'pipeline' of function calls (e.g., plot_histogram, get_stats) to execute the analysis. If the analysis is purely informational, use 'get_data_metadata'.

    DO NOT include any code or text outside of the JSON block.
    """
    
    try:
        # 2. Call the Gemini API to get the structured analysis plan
        
        # Define the list of available functions for the model to generate the plan
        analysis_functions = [
            get_data_metadata,
            get_stats,
            plot_histogram,
            calculate_invariant_mass
        ]

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-09-2025',
            contents=[
                types.Content(role='user', parts=[types.Part.from_text(user_prompt)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=types.Part.from_text(system_prompt),
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                # Explicitly pass the function definitions so the model can choose them for the plan
                tools=analysis_functions 
            )
        )
        
        # 3. Parse the structured JSON output
        if not response.text:
            return {'success': False, 'error': "AI returned an empty response."}

        # The response.text should be a JSON string conforming to RESPONSE_SCHEMA
        try:
            analysis_plan = json.loads(response.text)
        except json.JSONDecodeError:
            print("Raw response text:", response.text)
            return {'success': False, 'error': "AI returned malformed JSON. Check the console for raw output."}


        # 4. Input Validation (minimal safety check)
        if not isinstance(analysis_plan.get('pipeline'), list):
            return {'success': False, 'error': "AI plan is missing a valid 'pipeline' array."}

        
        # 5. Execute the analysis pipeline defined by the model
        pipeline_results = []
        current_df = df.copy() # Start with a copy of the original dataframe

        for step in analysis_plan.get('pipeline', []):
            function_name = step.get('function')
            args = step.get('args', {})

            if function_name == 'plot_histogram':
                # Plotting step
                plot_data = plot_histogram_to_base64(current_df, **args)
                pipeline_results.append({'function': function_name, 'plot': plot_data, 'args': args})

            elif function_name == 'get_data_metadata':
                # Simple metadata step
                result_msg = f"Data has {current_df.shape[0]} rows (events) and {current_df.shape[1]} columns (variables). Column names are: {', '.join(current_df.columns.tolist())}"
                pipeline_results.append({'function': function_name, 'result': result_msg})

            elif function_name == 'calculate_invariant_mass':
                # Placeholder for the actual calculation logic
                 result_msg = f"Mass calculation for type {args.get('mass_type')} is planned but not executed in this simplified model. Placeholder execution."
                 pipeline_results.append({'function': function_name, 'result': result_msg})

            elif function_name == 'get_stats':
                # Basic statistics step
                col = args.get('column')
                if col and col in current_df.columns:
                    stats = current_df[col].describe().to_dict()
                    result_msg = f"Statistics for **{col}**:\n"
                    # Format stats nicely
                    result_msg += "\n".join([f"- {k.capitalize()}: {v:.4f}" for k, v in stats.items()])
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
            'final_df_shape': current_df.shape
        }

    except Exception as e:
        # Catch and report any generic execution error, including API communication failure
        return {'success': False, 'error': f"AI Analysis Failed: {type(e).__name__}: {str(e)}"}

if __name__ == '__main__':
    # This block is for local testing only
    print("AI Analysis Assistant Module Loaded.")
