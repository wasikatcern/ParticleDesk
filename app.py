"""
Particle Physics Data Analysis Web Application
Analyze CERN Open Data with AI-powered analysis assistance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os

# Import custom modules
# FIX: Ensure all necessary functions are imported from the 'examples' package
from examples import get_all_examples, get_example, get_example_ids, get_example_titles 
from utils import (
    fetch_data_from_url,
    download_cern_dataset,
    load_csv_data,
    open_root_file,
    get_root_file_info,
    read_tree_to_dataframe,
    MUON_MASS,
    ELECTRON_MASS,
    HIGGS_MASS_EXPECTED
)
# Import the AI analysis function
from ai_analysis import analyze_with_ai 

# Page configuration
st.set_page_config(
    page_title="ParticleDesk : Particle Physics Data Analysis Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Streamlit session state variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'plot_info' not in st.session_state:
    st.session_state.plot_info = None


# --- Data Loading Handlers ---

def load_data_from_upload(uploaded_file):
    """Handle Streamlit file upload."""
    if uploaded_file.name.endswith('.csv'):
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            st.session_state.df = df
            st.session_state.filename = uploaded_file.name
            st.success(f"Successfully loaded CSV file: **{uploaded_file.name}** ({len(df):,} events)")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    elif uploaded_file.name.endswith('.root'):
        try:
            # Save the uploaded ROOT file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read the tree from the ROOT file
            root_file_info = get_root_file_info(temp_path)
            trees = [t['name'] for t in root_file_info['trees']]
            
            if not trees:
                st.error("No TTrees found in the ROOT file.")
                return
            
            # Use the first tree found for simplicity
            tree_name = trees[0]
            df = read_tree_to_dataframe(temp_path, tree_name)
            st.session_state.df = df
            st.session_state.filename = uploaded_file.name
            st.success(f"Successfully loaded ROOT file: **{uploaded_file.name}** (Tree: **{tree_name}**, {len(df):,} events)")

        except Exception as e:
            st.error(f"Error loading ROOT file: {e}")
            st.exception(e)
    else:
        st.error("Unsupported file type. Please upload a `.csv` or `.root` file.")

def load_example_data(example_id, file_name):
    """Load data for a selected example."""
    try:
        # Load the specific CSV file for the example (Simplified local load for this context)
        
        # For demonstration, we assume a local data structure or pre-fetched data
        data_url = get_example(example_id)['dataset_url'] 
        st.info(f"Downloading {file_name} from CERN Open Data...")
        filepath = download_cern_dataset(get_example(example_id)['record_id'], file_name)
        
        if filepath.endswith('.csv'):
            df = load_csv_data(filepath)
        elif filepath.endswith('.root'):
            # Load the main tree from the ROOT file
            tree_name = 'events' # Common tree name
            df = read_tree_to_dataframe(filepath, tree_name)
        else:
             st.error("Unsupported file type for example data.")
             return
            
        st.session_state.df = df
        st.session_state.filename = f"Example: {file_name}"
        st.session_state.analysis_results = None
        st.session_state.plot_info = None
        st.success(f"Loaded example dataset: **{file_name}** ({len(df):,} events)")
        
    except Exception as e:
        st.error(f"Failed to load example data: {e}")
        st.exception(e)

# --- Sidebar ---

st.sidebar.markdown("<h1 class='main-header'>ParticleDesk ⚛️</h1>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    **AI-Powered Particle Physics Data Analysis**
    
    Explore and analyze event data from the CMS Experiment at CERN.
    """
)

st.sidebar.markdown("---")


# --- Main Application Layout ---

# Tabbed interface
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Data & Summary", 
    "📤 Upload Data", 
    "📚 Examples Gallery", 
    "🤖 AI Analysis"
])

# --- TAB 1: Data & Summary ---
with tab1:
    st.markdown("## Data and Summary")
    
    if st.session_state.filename:
        st.markdown(f"### Current Dataset: **{st.session_state.filename}**")
    
    if not st.session_state.df.empty:
        df = st.session_state.df
        
        # Data Viewer
        st.markdown("### 📊 Data Viewer")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Basic Statistics
        st.markdown("### 📈 Basic Statistics")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Events", f"{len(df):,}")
        col2.metric("Features (Columns)", f"{len(df.columns)}")
        
        # Identify the main invariant mass column if it exists
        invariant_mass_cols = [c for c in df.columns if 'M' in c or 'm' in c]
        if invariant_mass_cols:
            mass_col = invariant_mass_cols[0]
            col3.metric(f"Mean {mass_col} (GeV/c²)", f"{df[mass_col].mean():.2f}")
        else:
            col3.metric("Status", "Ready for analysis")

        
        # Feature Distribution (Plotting)
        st.markdown("### 📉 Feature Distribution")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            default_cols = [c for c in ['M_4l', 'M', 'pt1'] if c in numerical_cols]
            default_col = default_cols[0] if default_cols else numerical_cols[0]
            
            selected_col = st.selectbox(
                "Select a column to plot distribution:",
                options=numerical_cols,
                index=numerical_cols.index(default_col) if default_col in numerical_cols else 0
            )
            
            # Plot the histogram using Plotly
            fig = px.histogram(
                df, 
                x=selected_col, 
                nbins=50, 
                title=f'Distribution of {selected_col}',
                labels={selected_col: f'{selected_col} [GeV/c²]' if 'M' in selected_col or 'pt' in selected_col else selected_col},
                color_discrete_sequence=['#1f77b4'] # Streamlit blue
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)


        # Correlation matrix
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            st.markdown("### 🔗 Correlation Matrix")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ No data loaded. Please upload data or load an example first.")
        st.markdown("Go to **📤 Upload Data** or **📚 Examples Gallery** to load a dataset.")

# --- TAB 2: Upload Data ---
with tab2:
    st.markdown("## Upload Your Data")
    st.markdown("""
    Upload a data file from your local machine.
    
    **Supported formats:**
    
    * `.csv` (Comma Separated Values)
    * `.root` (CERN ROOT files, requires `uproot` and `awkward`)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or ROOT file",
        type=['csv', 'root'],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        load_data_from_upload(uploaded_file)
        
    st.markdown("---")
    st.markdown("### Data Source Example")
    st.markdown("""
    You can analyze data from CERN Open Data:
    
    * **Higgs $H\\to 4\ell$ Sample:** `https://opendata.cern.ch/record/5200`
    * **Z Boson $Z\\to \ell\ell$ Sample:** `https://opendata.cern.ch/record/5006`
    
    Use the **Examples Gallery** for easy loading.
    """)

# --- TAB 3: Examples Gallery ---
with tab3:
    st.markdown("## Examples Gallery: CERN Open Data")
    
    example_ids = get_example_ids()
    example_titles = get_example_titles()
    
    if example_ids:
        
        # Display examples as radio buttons or a select box
        selected_title = st.selectbox(
            "Select a Particle Physics Analysis Example:",
            options=list(example_titles.values()),
            format_func=lambda x: x 
        )
        
        # Map title back to ID
        selected_id = next(k for k, v in example_titles.items() if v == selected_title)
        example_data = get_example(selected_id)
        
        st.markdown(f"### {example_data['title']}")
        st.markdown(f"*{example_data['description'].strip()}*")
        st.markdown(f"**Source:** [{example_data['source']}]({example_data['source']})")
        st.markdown(f"**Publication:** {example_data.get('publication', 'N/A')}")
        
        # Column information (for reference)
        with st.expander("Show Data Columns"):
            st.json(example_data.get('columns_info', {}))

        
        # Data File Selection (if multiple)
        if example_data['data_files']:
            st.markdown("---")
            st.markdown("#### Select Data File:")
            file_options = {
                f['name']: f for f in example_data['data_files']
            }
            selected_file_name = st.selectbox("File to Load", list(file_options.keys()))
            
            if st.button(f"Load '{selected_file_name}'", key="load_example_btn"):
                load_example_data(selected_id, selected_file_name)
        else:
            st.warning("This example has no specific files defined.")

# --- TAB 4: AI Analysis ---
with tab4:
    st.markdown("## 🤖 AI Analysis Assistant")
    st.markdown("""
    Use natural language to ask questions about your loaded dataset.
    The AI will suggest and execute a data analysis pipeline (filtering, plotting, statistics).
    """)
    
    # Check the state more explicitly for debugging the user's issue
    is_data_loaded = st.session_state.df is not None and not st.session_state.df.empty
    
    if not is_data_loaded:
        st.warning("⚠️ No data loaded. Please load a dataset first to use the AI assistant.")
        st.markdown("---")
        st.markdown("### 🔍 Troubleshooting Data Status:")
        
        # DEBUGGING: Display current internal state to help troubleshoot
        if st.session_state.df is not None:
             st.info(f"The system sees a DataFrame object, but it is empty (Size: {len(st.session_state.df)} events).")
        else:
             st.info("The system cannot find a valid DataFrame object in the session state.")
        st.markdown("Go back to **📂 Data & Summary** to confirm the data status, or try re-loading the example data.")

    else:
        df = st.session_state.df
        
        st.info(f"Analyzing data from: **{st.session_state.filename}** ({len(df):,} events)")
        
        # AI Prompt Input (This is now visible)
        user_prompt = st.text_area(
            "Enter your analysis request (e.g., 'Filter events where pt1 > 20 GeV and plot the M_4l distribution'):",
            height=100
        )
        
        if st.button("Run AI Analysis", key="run_ai_analysis_btn"):
            if user_prompt:
                with st.spinner("AI is generating and executing analysis pipeline..."):
                    
                    # 1. Run the AI to get the analysis plan and results
                    analysis_output = analyze_with_ai(user_prompt, df)
                    
                    if analysis_output['success']:
                        st.session_state.analysis_results = analysis_output
                        st.session_state.plot_info = analysis_output.get('plot_info')
                        
                        st.success("Analysis complete.")
                    else:
                        st.error(f"Analysis Failed: {analysis_output.get('error')}")
            else:
                st.warning("Please enter an analysis prompt.")
        
        # --- Display Results ---
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            st.markdown("### 📝 Analysis Plan")
            st.markdown(f"**Explanation:** {results['explanation']}")
            
            st.markdown("### ⚙️ Execution Steps")
            for step in results['results']:
                st.code(f"[{step['function'].upper()}] {step['result']}", language="text")
                
            st.markdown(f"**Final DataFrame Size:** {results['final_df_shape'][0]} events remaining.")
            
            # --- Render Plot ---
            plot_info = st.session_state.plot_info
            if plot_info and 'df' in plot_info:
                st.markdown("### 📊 Generated Plot")
                
                # Create a figure and axes for the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Import the plotting function dynamically or ensure it's available
                # NOTE: Assuming plot_histogram_internal is accessible from the imported ai_analysis module context
                
                try:
                    from ai_analysis import plot_histogram_internal
                    plot_result = plot_histogram_internal(plot_info['df'], ax, **plot_info)
                    
                    if plot_result['success']:
                        st.pyplot(fig)
                    else:
                        st.error(f"Plotting Error: {plot_result['error']}")
                except ImportError:
                    st.error("Plotting function `plot_histogram_internal` could not be accessed. Please ensure it is correctly defined and imported/accessible.")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [CERN Open Data](https://opendata.cern.ch/)
- [CMS Open Data](https://cms-opendata-guide.web.cern.ch/)
- [GitHub Examples](https://github.com/cms-opendata-analyses)
""")

st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
Built with Streamlit and Python for particle physics data analysis.
""")
