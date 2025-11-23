"""
Particle Physics Data Analysis Web Application
Analyze CERN Open Data with AI-powered analysis assistance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os

# Import custom modules
from examples_data import get_all_examples, get_example, get_example_ids
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
    }
    .stCodeBlock {
        background-color: #e6e6e6;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# --- State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_name' not in st.session_state:
    st.session_state.df_name = "No data loaded"
if 'ai_results' not in st.session_state:
    st.session_state.ai_results = None

# --- Helper Functions ---

def load_data_from_df(df, name):
    """Loads a DataFrame into session state."""
    st.session_state.df = df
    st.session_state.df_name = name
    st.session_state.ai_results = None # Clear old AI results
    st.success(f"Successfully loaded data: **{name}**")

def handle_url_load(url, url_type):
    """Handles loading data from a URL."""
    try:
        if url_type == 'CERN Open Data':
            # Assuming format: https://opendata.cern.ch/record/{id}/files/{filename}
            parts = url.split('/')
            if len(parts) >= 7 and parts[5].isdigit():
                record_id = parts[5]
                filename = parts[7]
                st.info(f"Downloading {filename} from CERN Record {record_id}...")
                filepath = download_cern_dataset(record_id, filename)
                
                if filepath.endswith('.csv'):
                    df = load_csv_data(filepath)
                    load_data_from_df(df, filename)
                elif filepath.endswith('.root'):
                    with open_root_file(filepath) as root_file:
                        tree_names = list_trees(root_file)
                        if not tree_names:
                            st.error("No trees found in ROOT file.")
                            return
                        # For simplicity, pick the first tree
                        tree_name = tree_names[0]
                        st.info(f"Reading tree: {tree_name}")
                        df = read_tree_to_dataframe(root_file, tree_name, max_entries=100000)
                        load_data_from_from(df, filename + f" ({tree_name})")
                else:
                    st.error("Unsupported file type from CERN URL.")
            else:
                st.error("Please provide the full file URL from the CERN Open Data portal.")
        else: # Direct URL or generic fetch
            df, file_type = fetch_data_from_url(url)
            
            if file_type == 'csv':
                load_data_from_df(df, os.path.basename(url))
            elif file_type == 'root':
                # This returns the temp path, now read it
                root_filepath = df
                with open_root_file(root_filepath) as root_file:
                    tree_names = list_trees(root_file)
                    if not tree_names:
                        st.error("No trees found in ROOT file.")
                        return
                    tree_name = tree_names[0]
                    st.info(f"Reading tree: {tree_name}")
                    df_root = read_tree_to_dataframe(root_file, tree_name, max_entries=100000)
                    load_data_from_df(df_root, os.path.basename(url) + f" ({tree_name})")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")

def handle_local_upload(uploaded_file):
    """Handles local file upload."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            load_data_from_df(df, uploaded_file.name)
        
        elif uploaded_file.name.endswith('.root'):
            # Save file to a temporary location for uproot to read
            temp_path = os.path.join("/tmp", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with open_root_file(temp_path) as root_file:
                tree_names = list_trees(root_file)
                if not tree_names:
                    st.error("No trees found in ROOT file.")
                    return
                tree_name = tree_names[0]
                st.info(f"Reading tree: {tree_name}")
                df = read_tree_to_dataframe(root_file, tree_name, max_entries=100000)
                load_data_from_df(df, uploaded_file.name + f" ({tree_name})")
                
            os.remove(temp_path) # Clean up temp file
        
        else:
            st.error("Unsupported file type. Please upload a CSV or ROOT file.")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        
def handle_ai_analysis():
    """Triggers the AI analysis."""
    if st.session_state.df is None:
        st.error("Please load a dataset before running AI analysis.")
        return

    if not st.session_state.ai_prompt:
        st.error("Please enter a prompt for the AI analysis.")
        return

    # Run analysis
    with st.spinner(f"Running AI analysis for: '{st.session_state.ai_prompt}'..."):
        try:
            results = analyze_with_ai(st.session_state.ai_prompt, st.session_state.df)
            st.session_state.ai_results = results
        except Exception as e:
            st.session_state.ai_results = {'success': False, 'error': f"Critical Error in AI System: {e}"}

# --- Sidebar Content (AI Analysis Panel) ---

st.sidebar.markdown(f"## Data Status: {st.session_state.df_name}")

if st.session_state.df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🧠 AI-Powered Analysis")
    
    # Input for AI prompt
    st.sidebar.text_area(
        "Enter your analysis request (e.g., 'Plot invariant mass of 4 leptons between 80 and 150 GeV', 'Calculate Z boson mass')",
        key='ai_prompt',
        height=150
    )
    
    # Button to trigger analysis
    st.sidebar.button("🔬 Run AI Analysis", on_click=handle_ai_analysis, use_container_width=True, type="primary")

# --- Main Application Layout ---

st.title("⚛️ ParticleDesk: Particle Physics Data Analysis")

# Main tabs setup
tab_overview, tab_gallery, tab_upload, tab_dataview, tab_analysis = st.tabs([
    "🏠 Overview", 
    "📚 Examples Gallery", 
    "📤 Upload Data", 
    "📊 Data View", 
    "🔬 Analysis Results"
])

# --- 1. Overview Tab ---
with tab_overview:
    st.markdown("<p class='main-header'>Explore the Subatomic World with AI</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <p>This platform allows you to analyze open data from particle physics experiments like those at CERN's Large Hadron Collider (LHC).</p>
        <p>Use the tabs above to **load data** (via the Examples Gallery or Upload Data), and then navigate to **Data View** to inspect it.
        You can then use the **AI-Powered Analysis** section in the sidebar to ask natural language questions about your data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Quick Start Guide")
    st.markdown("""
    1. **Go to 📚 Examples Gallery** and click **Load** on an example dataset (e.g., Higgs H→4ℓ).
    2. **Go to 📊 Data View** to see the loaded data and its structure.
    3. **Go to the sidebar**, enter a prompt like `Plot the invariant mass column (M) in a histogram` and click **Run AI Analysis**.
    4. **Go to 🔬 Analysis Results** to see the AI's explanation and plot.
    """)
    
    if st.session_state.df is None:
        st.info("No data currently loaded. Please load a dataset to begin analysis.")
    else:
        st.success(f"Data Loaded: **{st.session_state.df_name}** with {len(st.session_state.df):,} rows and {len(st.session_state.df.columns)} columns.")


# --- 2. Examples Gallery Tab ---
with tab_gallery:
    st.header("📚 Examples Gallery: CERN Open Data")
    st.markdown("Load a pre-curated dataset from open access resources to start your analysis immediately.")
    
    examples = get_all_examples()
    
    for example_id in get_example_ids():
        example = examples[example_id]
        
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(example['title'])
            st.markdown(example['description'])
            st.markdown(f"**Source:** [{example['source']}]({example['source']})")
            
            st.markdown("##### Suggested AI Prompts:")
            prompts = " | ".join([f"`{p}`" for p in example['suggested_prompts'][:3]])
            st.markdown(f"* {prompts}")
            
        with col2:
            st.markdown("###") # Spacer
            if st.button(f"⚛️ Load Data for {example_id.replace('_', ' ').title()}", key=f"load_{example_id}", use_container_width=True):
                # For simplicity, we'll auto-load the first data file's URL
                if example['data_files']:
                    first_file = example['data_files'][0]
                    handle_url_load(first_file['url'], 'Direct URL')
                else:
                    st.error("No direct data files available for this example.")


# --- 3. Upload Data Tab ---
with tab_upload:
    st.header("📤 Upload Data")

    st.markdown("You can upload your own particle physics data. **Supported formats: CSV and ROOT (`.root`) files.**")
    
    # Local File Upload
    st.subheader("Upload Local File (CSV or ROOT)")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'root'],
        accept_multiple_files=False,
        help="Upload a CSV file or a ROOT file containing a TTree. We will read the first TTree found."
    )
    if uploaded_file is not None:
        handle_local_upload(uploaded_file)
        
    st.markdown("---")
    
    # Load from URL
    st.subheader("Load Data from URL")
    url_type = st.radio(
        "Select URL Type:",
        ('Direct URL (CSV or ROOT)', 'CERN Open Data'),
        horizontal=True
    )
    
    url_input = st.text_input(
        f"Enter {url_type} URL:",
        placeholder="e.g., https://example.com/data.csv or https://opendata.cern.ch/record/5200/files/4mu_2011.csv",
        key='url_input'
    )
    
    if st.button("⬇️ Fetch Data from URL", use_container_width=True):
        if url_input:
            handle_url_load(url_input, url_type)
        else:
            st.error("Please enter a valid URL.")


# --- 4. Data View Tab ---
with tab_dataview:
    st.header("📊 Data Overview")
    df = st.session_state.df
    
    if df is not None:
        st.markdown(f"### Current Dataset: `{st.session_state.df_name}`")
        
        # Display DataFrame
        st.markdown("#### Raw Data Preview")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Column Information
        st.markdown("#### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Dtype': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Unique Values': df.nunique()
        }).reset_index(drop=True)
        st.dataframe(col_info, use_container_width=True)
        
        # Basic Statistics
        st.markdown("#### Descriptive Statistics (Numerical Columns)")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats = numeric_df.describe().T
            st.dataframe(stats, use_container_width=True)

        # Value Counts for Categorical Data
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            st.markdown("#### Value Counts (Categorical Columns)")
            for col in categorical_cols:
                st.markdown(f"##### Column: `{col}`")
                st.dataframe(df[col].value_counts().reset_index(), use_container_width=True)

        # Correlation matrix
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            st.markdown("### 🔗 Correlation Matrix")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                # FIX: use_column_width replaced with use_container_width
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
        st.markdown("Go to **📚 Examples Gallery** or **📤 Upload Data** to load a dataset.")


# --- 5. Analysis Results Tab ---
with tab_analysis:
    st.header("🔬 Analysis Results")
    st.markdown("View the output from the AI-Powered Analysis here.")
    
    if st.session_state.ai_results is None:
        st.info("Run an analysis using the 'AI-Powered Analysis' panel in the sidebar.")
    elif st.session_state.ai_results['success'] == False:
        st.error("Analysis Failed")
        st.code(st.session_state.ai_results['error'])
    else:
        results = st.session_state.ai_results
        
        st.success("✅ AI Analysis Completed Successfully!")
        
        st.subheader("Explanation")
        st.markdown(results.get('explanation', 'No detailed explanation provided by the AI.'))
        
        st.subheader("Results Pipeline")
        st.dataframe(results.get('results', []), use_container_width=True)
        
        # Check for and display plot
        plot_data = results.get('plot', None)
        if plot_data:
            st.subheader("Generated Plot")
            
            # Matplotlib Figure Display
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # The AI Analysis returns a function name and arguments
            plot_func = plot_data.get('function')
            plot_args = plot_data.get('args', {})
            
            # Import internal plotting helper function locally for execution
            from ai_analysis import plot_histogram_internal
            
            try:
                # Execute the internal plotting function
                plot_histogram_internal(st.session_state.df, ax, plot_func, **plot_args)
                st.pyplot(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error rendering plot from AI results: {e}")
            
        else:
            st.info("The analysis did not generate a plot.")
            
        st.markdown(f"**Final DataFrame Shape after filtering/calculation:** {results.get('final_df_shape')}")


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

# Ensure plt is closed to free memory
plt.close('all')
