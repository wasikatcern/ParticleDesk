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
from examples import get_all_examples, get_example
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
        border: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f2f6;
        color: #333;
        border-radius: 0.5rem 0.5rem 0 0;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = "N/A"
if 'file_type' not in st.session_state:
    st.session_state.file_type = "N/A"


# --- Helper Functions for Data Loading ---

def set_dataframe(df: pd.DataFrame, file_name: str, file_type: str):
    """Set the main DataFrame and file metadata in session state."""
    st.session_state.df = df
    st.session_state.file_name = file_name
    st.session_state.file_type = file_type
    st.success(f"Data loaded successfully: {file_name} ({len(df)} rows)")

def handle_local_upload(uploaded_file):
    """Handle Streamlit file upload."""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        set_dataframe(df, uploaded_file.name, 'CSV')
    else:
        st.error("Unsupported file format. Please upload a .csv file.")

def handle_cern_download(record_id, filename):
    """Handle CERN Open Data download."""
    try:
        filepath = download_cern_dataset(record_id, filename)
        if filename.endswith('.csv'):
            df = load_csv_data(filepath)
            set_dataframe(df, filename, 'CSV')
        elif filename.endswith('.root'):
            # For simplicity, we just show the root file info here, 
            # the user will have to manually select the tree in the analysis tab
            info = get_root_file_info(filepath)
            st.session_state.root_info = info
            st.session_state.filepath = filepath
            st.success(f"ROOT file downloaded: {filename}. Go to the 'ROOT File Analysis' tab to process trees.")
        else:
            st.error("Unsupported file type from CERN.")
    except Exception as e:
        st.error(f"Download Error: {e}")

def handle_example_load(example_id):
    """Load data from a curated example."""
    example = get_example(example_id)
    if example.get("data_files"):
        first_file = example["data_files"][0]
        url = first_file["url"]
        name = first_file["name"]
        
        # Check if file is already downloaded/loaded
        if st.session_state.file_name == name and st.session_state.df is not None:
             st.info(f"Example data '{name}' is already loaded.")
             return
             
        st.info(f"Loading example data: {name} from {url}...")
        try:
            df, file_type = fetch_data_from_url(url)
            set_dataframe(df, name, file_type.upper())
        except Exception as e:
            st.error(f"Failed to load example data: {e}")


# --- Sidebar ---
st.sidebar.title("⚛️ ParticleDesk")
st.sidebar.subheader("Data Management")

# --- Example Gallery Tab (Sidebar) ---
st.sidebar.markdown("### 📚 Examples Gallery")
example_ids = list(get_all_examples().keys())
selected_example = st.sidebar.selectbox("Select a CERN Open Data Example:", 
                                        ['-- Select an Example --'] + example_ids,
                                        format_func=lambda x: get_example(x).get('title', x))
if selected_example != '-- Select an Example --':
    if st.sidebar.button(f"Load '{get_example(selected_example)['title']}'"):
        handle_example_load(selected_example)

st.sidebar.markdown("---")

# --- Upload Data Tab (Sidebar) ---
st.sidebar.markdown("### 📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File:", type=["csv"])
if uploaded_file is not None:
    handle_local_upload(uploaded_file)
    
st.sidebar.markdown("---")

# --- Main App Body ---
st.title("ParticleDesk: AI-Powered Particle Physics Analysis")
st.markdown(f"**Loaded Dataset:** `{st.session_state.file_name}` (`{st.session_state.file_type}`)", unsafe_allow_html=True)
df = st.session_state.df

# --- Tabs ---
tab_eda, tab_ai_analysis = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🤖 AI Analysis Assistant"])


# --- AI Analysis Tab ---
with tab_ai_analysis:
    st.markdown("## 🤖 AI Analysis Assistant")
    st.markdown("""
        Use the power of a large language model to analyze your particle physics data. 
        Describe your analysis goal (e.g., "Plot the invariant mass of the four-lepton system and look for the Higgs boson peak around 125 GeV") and the AI will generate the code, perform the analysis, and explain the results.
    """)

    if df is None:
        st.warning("Please load a dataset first (via 'Upload Data' or 'Examples Gallery') to use the AI assistant.")
    else:
        st.markdown(f"""
            ### Dataset Context: `{st.session_state.file_name}`
            The DataFrame has **{len(df)}** rows and **{len(df.columns)}** columns. 
            The available columns are: `{', '.join(df.columns.tolist())}`
        """)

        user_prompt = st.text_area(
            "Enter your analysis request:",
            height=150,
            value="Plot the histogram of the four-lepton invariant mass (column M_4l) from 100 to 150 GeV in 50 bins. Label the expected Higgs mass at 125 GeV."
        )

        if st.button("🚀 Run AI Analysis", type="primary"):
            if not user_prompt:
                st.error("Please enter an analysis prompt.")
            else:
                with st.spinner("Analyzing data with Gemini..."):
                    try:
                        # Call the AI analysis function
                        analysis_results = analyze_with_ai(user_prompt, df)

                        if analysis_results.get('success', False):
                            st.subheader("✅ Analysis Complete")
                            
                            # 1. Display Explanation
                            st.markdown("### 📝 Explanation of Results")
                            st.info(analysis_results.get('explanation', 'No explanation provided.'))
                            
                            # 2. Display Plot
                            plot_data = analysis_results.get('plot_data')
                            if plot_data and 'fig' in plot_data:
                                st.markdown("### 📈 Generated Plot")
                                st.pyplot(plot_data['fig'])
                            else:
                                st.warning("No plot was generated for this request.")

                            # 3. Display Code
                            st.markdown("### 🐍 Generated Python Code")
                            st.code(analysis_results.get('code', 'No code generated.'), language='python')
                            
                        else:
                            st.error(f"AI Analysis Failed: {analysis_results.get('error', 'Unknown error.')}")

                    except Exception as e:
                        st.exception(f"An unexpected error occurred during AI analysis: {e}")


# --- EDA Tab ---
with tab_eda:
    st.markdown("## 📊 Exploratory Data Analysis")

    if df is not None:
        
        # Display Data Information
        st.markdown("### 📋 Data Overview")
        st.markdown(f"**File Name:** `{st.session_state.file_name}` | **Rows:** `{len(df)}` | **Columns:** `{len(df.columns)}`")
        
        if st.checkbox("Show Raw Data (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)

        # Basic Statistics
        st.markdown("### 🔢 Descriptive Statistics")
        st.dataframe(df.describe(include='all').T, use_container_width=True)
        
        # Column Details
        st.markdown("### 🔍 Column Details")
        col_info = pd.DataFrame({
            'Dtype': df.dtypes,
            'Non-Null Count': df.count(),
            'Unique Values': df.nunique()
        }).sort_values(by='Dtype')
        st.dataframe(col_info, use_container_width=True)

        # Histograms for numerical data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("### 📈 Histograms of Key Variables")
            
            # Select column for quick visualization
            selected_col = st.selectbox("Select a column to plot:", numeric_cols)
            
            if selected_col:
                # Plotly histogram
                fig = px.histogram(
                    df, 
                    x=selected_col, 
                    nbins=50, 
                    title=f"Distribution of {selected_col}",
                    template="plotly_white",
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
        st.markdown("Go to **📚 Examples Gallery** to load a dataset.")

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
