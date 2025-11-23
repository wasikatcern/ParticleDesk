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
# FIX: The file is located at 'examples/examples_data.py'. 
# The import must reference the module within the 'examples' package.
# This assumes the 'examples' directory contains an __init__.py file.
from examples.examples_data import get_all_examples, get_example
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
    .stSelectbox label, .stTextInput label, .stFileUploader label {
        font-weight: bold;
        color: #1f77b4;
    }
    .stButton>button {
        color: white !important;
        background-color: #1f77b4;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1a5e8f;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'filename' not in st.session_state:
    st.session_state.filename = "No Data"
if 'file_type' not in st.session_state:
    st.session_state.file_type = ""
if 'root_info' not in st.session_state:
    st.session_state.root_info = {}
if 'root_filepath' not in st.session_state:
    st.session_state.root_filepath = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'ai_prompt' not in st.session_state:
    st.session_state.ai_prompt = ""
if 'data_source' not in st.session_state:
    st.session_state.data_source = "None"
if 'last_analysis_prompt' not in st.session_state:
    st.session_state.last_analysis_prompt = ""


# --- Data Loading Handlers ---

def load_data(df_or_path, filename, file_type, root_info=None):
    """Updates session state with loaded data."""
    st.session_state.filename = filename
    st.session_state.file_type = file_type
    st.session_state.analysis_results = None # Clear previous analysis

    if file_type == 'csv':
        st.session_state.df = df_or_path
        st.session_state.data_loaded = True
        st.session_state.root_info = {}
        st.session_state.root_filepath = None
        st.success(f"✅ Successfully loaded CSV data: **{filename}** ({len(st.session_state.df):,} events)")
    
    elif file_type == 'root':
        st.session_state.root_filepath = df_or_path # Store path
        st.session_state.root_info = root_info # Store info
        st.session_state.df = pd.DataFrame() # Clear DataFrame until a tree is selected
        st.session_state.data_loaded = False # Set to False until a tree is read
        st.success(f"✅ Successfully opened ROOT file: **{filename}** ({len(root_info.get('trees', []))} TTree(s) found)")


def load_example_data(example_id, file_index=0):
    """Loads a specific example dataset from CERN Open Data."""
    example = get_example(example_id)
    if not example or not example.get('data_files'):
        st.error("❌ Example data not found or no files specified.")
        return

    data_file = example['data_files'][file_index]
    filename = data_file['name']
    url = data_file['url']
    
    try:
        if filename.endswith('.root'):
            # Fetch ROOT file (saved to temp path)
            root_filepath, file_type = fetch_data_from_url(url)
            root_info = get_root_file_info(root_filepath)
            
            # Auto-read the first TTree for immediate use
            if root_info.get('trees'):
                tree_name = root_info['trees'][0]['name']
                df = read_tree_to_dataframe(root_filepath, tree_name)
                load_data(df, filename, 'csv') # Treat as CSV for plotting ease
                st.session_state.data_source = f"CERN Open Data: {example['title']}"
            else:
                 load_data(root_filepath, filename, 'root', root_info)
                 st.session_state.data_source = f"CERN Open Data: {example['title']}"

        elif filename.endswith('.csv'):
            df, file_type = fetch_data_from_url(url)
            load_data(df, filename, file_type)
            st.session_state.data_source = f"CERN Open Data: {example['title']}"

        else:
            st.error(f"❌ Unsupported file type for example: {filename}")
            
    except Exception as e:
        st.error(f"❌ Failed to load example data: {str(e)}")


def run_ai_analysis():
    """Runs the AI analysis and updates results in session state."""
    if not st.session_state.data_loaded:
        st.warning("Please load a dataset first.")
        return

    st.session_state.last_analysis_prompt = st.session_state.ai_prompt # Store prompt
    
    with st.spinner(f"Running AI analysis for: '{st.session_state.ai_prompt}'..."):
        try:
            results = analyze_with_ai(st.session_state.ai_prompt, st.session_state.df)
            st.session_state.analysis_results = results
        except Exception as e:
            st.session_state.analysis_results = {'success': False, 'error': str(e)}


# --- Sidebar Navigation ---

st.sidebar.title("ParticleDesk ⚛️")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation", 
    ["📊 Data Overview", "🤖 AI Analysis", "📚 Examples Gallery", "📤 Upload Data"],
    index=0
)

# --- Main Page Content ---

st.markdown("<p class='main-header'>Particle Physics Data Analysis Platform</p>", unsafe_allow_html=True)

# Display current data status
if st.session_state.data_loaded:
    st.sidebar.success(f"Loaded: **{st.session_state.filename}** ({len(st.session_state.df):,} events)")
    st.sidebar.caption(f"Source: {st.session_state.data_source}")
else:
    st.sidebar.info("No data loaded.")


# --- Page: Examples Gallery ---

if page == "📚 Examples Gallery":
    st.header("📚 Examples Gallery")
    st.markdown("""
    Explore curated datasets from CERN Open Data for quick analysis and learning.
    """)
    
    examples = get_all_examples()
    example_ids = list(examples.keys())
    
    selected_id = st.selectbox("Select Example Dataset", example_ids, format_func=lambda x: examples[x]['title'])

    if selected_id:
        example = examples[selected_id]
        
        st.markdown(f"### {example['title']}")
        st.info(example['description'])
        
        if 'publication' in example:
            st.markdown(f"**Publication:** {example['publication']}")
        
        st.markdown(f"**Data Source:** [{example['dataset_url']}]({example['dataset_url']})")
        
        st.subheader("Data Files")
        for i, file_info in enumerate(example.get('data_files', [])):
            col1, col2 = st.columns([1, 4])
            col2.markdown(f"**{file_info['name']}**")
            col1.button("Load", key=f"load_example_{i}", help=f"Load {file_info['name']}", on_click=load_example_data, args=(selected_id, i))

        st.subheader("Suggested AI Prompts")
        prompt_col1, prompt_col2 = st.columns(2)
        for i, prompt in enumerate(example.get('suggested_prompts', [])):
            with prompt_col1 if i % 2 == 0 else prompt_col2:
                st.code(prompt, language='text')

# --- Page: Upload Data ---

elif page == "📤 Upload Data":
    st.header("📤 Upload Data")
    st.markdown("""
    Upload your own CSV or ROOT files, or fetch a dataset directly from a URL.
    """)
    
    tab_upload, tab_url = st.tabs(["Local Upload", "URL Fetch"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload CSV or ROOT File", type=["csv", "root"])
        
        if uploaded_file is not None:
            filename = uploaded_file.name
            
            if filename.endswith('.csv'):
                try:
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    df = load_csv_data(stringio)
                    load_data(df, filename, 'csv')
                    st.session_state.data_source = "Local Upload"
                except Exception as e:
                    st.error(f"Failed to read CSV: {str(e)}")
                    
            elif filename.endswith('.root'):
                # In a Streamlit environment, handle the uploaded file buffer
                # by saving it temporarily to disk for uproot to access.
                temp_path = f"/tmp/{filename}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    root_info = get_root_file_info(temp_path)
                    load_data(temp_path, filename, 'root', root_info)
                    st.session_state.data_source = "Local Upload"
                except Exception as e:
                    st.error(f"Failed to open ROOT file: {str(e)}")
                    os.remove(temp_path)

    with tab_url:
        url_input = st.text_input("Enter URL for CSV or ROOT file", 
                                  placeholder="e.g., https://opendata.cern.ch/record/5200/files/4mu_2012.csv")
        if st.button("Fetch Data from URL") and url_input:
            with st.spinner(f"Fetching data from {url_input}..."):
                try:
                    result, file_type = fetch_data_from_url(url_input)
                    filename = url_input.split('/')[-1]
                    
                    if file_type == 'csv':
                        load_data(result, filename, 'csv')
                        st.session_state.data_source = "URL Fetch"
                    elif file_type == 'root':
                        root_filepath = result
                        root_info = get_root_file_info(root_filepath)
                        load_data(root_filepath, filename, 'root', root_info)
                        st.session_state.data_source = "URL Fetch"
                    
                except Exception as e:
                    st.error(f"❌ Fetch failed: {str(e)}")


# --- Page: AI Analysis ---

elif page == "🤖 AI Analysis":
    st.header("🤖 AI-Powered Analysis")
    
    if not st.session_state.data_loaded and not st.session_state.root_filepath:
        st.warning("⚠️ No data loaded. Please load a CSV or read a TTree from a ROOT file first.")
    
    else:
        # Step 1: Handle ROOT File Tree Selection (if necessary)
        if st.session_state.file_type == 'root' and st.session_state.root_info and not st.session_state.data_loaded:
            st.info(f"ROOT file **{st.session_state.filename}** loaded. Please select a TTree to load data.")
            
            tree_names = [t['name'] for t in st.session_state.root_info.get('trees', [])]
            selected_tree = st.selectbox("Select TTree to Analyze", tree_names)
            
            if selected_tree and st.button("Load Tree Data"):
                with st.spinner(f"Reading TTree '{selected_tree}'..."):
                    try:
                        # Re-read the full DataFrame from the selected tree
                        df = read_tree_to_dataframe(st.session_state.root_filepath, selected_tree)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.file_type = 'csv' # Treat as flat CSV for analysis
                        st.success(f"✅ Loaded **{selected_tree}** into DataFrame ({len(df):,} events).")
                    except Exception as e:
                        st.error(f"Failed to read tree: {str(e)}")
                        
            st.stop() # Stop further execution until tree is loaded
            
        # Step 2: AI Analysis Interface
        
        st.markdown("""
        Enter your analysis goal using natural language. The AI will generate a safe, multi-step analysis plan
        (filtering, calculating physics quantities, plotting) and execute it.
        """)
        
        # Display data summary for context
        if st.session_state.data_loaded:
            cols_info = st.session_state.df.columns.tolist()
            st.markdown(f"**Current Data Columns ({len(cols_info)}):** `{'`, `'.join(cols_info)}`")
            st.caption("Commonly used columns: `pt`, `eta`, `phi`, `M_4l`.")

        
        st.text_area(
            "Analysis Prompt",
            value=st.session_state.last_analysis_prompt or "Filter events with M_4l between 100 and 150 GeV and plot the invariant mass distribution.",
            key="ai_prompt",
            height=100
        )
        
        st.button("Run Analysis", key="run_ai", on_click=run_ai_analysis)

        # Step 3: Display Results
        
        results = st.session_state.analysis_results
        
        if results:
            st.markdown("---")
            st.subheader("AI Analysis Results")

            if not results['success']:
                st.error(f"Analysis Error: {results.get('error', 'Unknown error.')}")
            else:
                st.markdown(f"### 💡 AI Plan Explanation")
                st.info(results.get('explanation', 'No explanation provided.'))
                
                st.markdown(f"### ⚙️ Execution Details")
                st.code(f"Final Data Shape: {results['final_df_shape']}", language='text')

                
                for step in results.get('results', []):
                    st.markdown(f"**{step['function'].replace('_', ' ').title()}**")
                    st.code(step['result'], language='text')
                
                # Plot Display
                if results.get('plot_base64'):
                    st.markdown("### 📈 Generated Plot")
                    plot_b64 = results['plot_base64']
                    st.image(f"data:image/png;base64,{plot_b64}", use_column_width=True)
                else:
                    st.info("No plot was generated in this analysis run.")


# --- Page: Data Overview ---

elif page == "📊 Data Overview":
    st.header("📊 Data Overview")
    
    df = st.session_state.df
    
    if st.session_state.data_loaded:
        st.markdown(f"**File:** {st.session_state.filename} | **Events:** {len(df):,} | **Columns:** {len(df.columns)}")
        
        st.markdown("### 📋 Data Sample")
        st.dataframe(df.head(), use_container_width=True)
        
        # Display basic statistics for numerical columns
        numeric_df = df.select_dtypes(include=np.number)
        
        if not numeric_df.empty:
            st.markdown("### 🔢 Descriptive Statistics")
            stats = numeric_df.describe().T
            stats['Range'] = stats['max'] - stats['min']
            stats['Median'] = numeric_df.median()
            
            # Reorder for better display
            stats = stats[['count', 'mean', 'std', 'min', 'max', 'Median', 'Range']].rename(columns={'mean': 'Mean', 'std': 'Std Dev', 'min': 'Min', 'max': 'Max', 'count': 'Count'})
            st.dataframe(stats, use_container_width=True)

            # --- Visualizations ---
            
            st.markdown("### 🎨 Visualizations")

            # Column selection for plotting
            plot_col = st.selectbox(
                "Select a column for quick histogram", 
                numeric_df.columns.tolist(),
                key="overview_hist_col"
            )
            
            # Histogram
            if plot_col:
                st.markdown(f"#### Distribution of {plot_col}")
                
                # Auto-determine bins for a smooth look
                num_unique = len(df[plot_col].unique())
                bins = min(50, num_unique // 5 if num_unique > 50 else 20)
                
                fig = px.histogram(
                    df, 
                    x=plot_col, 
                    nbins=bins, 
                    title=f"Histogram of {plot_col}",
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
        # Handle ROOT file info display if only the file is loaded, but not the tree
        if st.session_state.file_type == 'root' and st.session_state.root_info:
            st.warning(f"ROOT file **{st.session_state.filename}** loaded. Please go to **🤖 AI Analysis** to select and load a TTree.")
            st.markdown("### File Contents")
            st.json(st.session_state.root_info)
        else:
            st.warning("⚠️ No data loaded. Please upload data or load an example first.")
            st.markdown("Go to **📤 Upload Data** or **📚 Examples Gallery** to load a dataset.")

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
Built by Dr. Wasikul Islam (https://wasikatcern.github.io), with Streamlit and Python for particle physics data analysis.
""")
