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
import time

# Custom Imports (assuming they are in the parent directory or properly structured)
# Note: For this to work, ensure your local directory structure correctly exposes:
# - ai_analysis.py -> analyze_with_ai
# - root_parser.py -> read_histogram, get_root_file_info, read_tree_to_dataframe
# - physics_utils.py -> MUON_MASS, ELECTRON_MASS, etc.
# - examples_data.py -> get_all_examples, get_example
from ai_analysis import analyze_with_ai
from utils import (
    fetch_data_from_url,
    download_cern_dataset,
    load_csv_data,
    get_root_file_info,
    read_tree_to_dataframe,
    read_histogram, # New import for reading histograms
    MUON_MASS,
    ELECTRON_MASS,
    HIGGS_MASS_EXPECTED
)
from examples_data import get_all_examples, get_example

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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSpinner > div > div {
        color: #1f77b4 !important;
    }
    /* Streamlit components styling */
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for data storage
if 'current_data' not in st.session_state:
    st.session_state.current_data = None  # Stores pandas DataFrame (for event-level analysis)
if 'current_root_object' not in st.session_state:
    st.session_state.current_root_object = None # Stores histogram/other pre-calculated data (for direct plotting)
if 'data_source' not in st.session_state:
    st.session_state.data_source = "None"
if 'current_data_type' not in st.session_state:
    st.session_state.current_data_type = "None"


def load_data_into_state(data, source_name, data_type="DataFrame", root_object=None):
    """Helper function to set session state for loaded data."""
    if data_type == "DataFrame":
        st.session_state.current_data = data
        st.session_state.current_root_object = None
    elif data_type == "ROOT_Object":
        st.session_state.current_data = None
        st.session_state.current_root_object = root_object
    
    st.session_state.data_source = source_name
    st.session_state.current_data_type = data_type
    st.success(f"✅ Data loaded successfully from: **{source_name}**")


# --- Sidebar Navigation ---
st.sidebar.title("⚛️ ParticleDesk")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📚 Examples Gallery", "📤 Upload Data", "🔍 Data Explorer", "🤖 AI Analysis"])
st.sidebar.markdown("---")

# --- Home Page ---
if page == "🏠 Home":
    st.markdown('<div class="main-header">Welcome to ParticleDesk</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>Platform Status:</strong> Ready for Analysis</p>
        <p><strong>Current Data:</strong> <span style="font-weight: bold;">{}</span> (Type: <span style="font-weight: bold;">{}</span>)</p>
        <p>This tool is designed for interactive particle physics data analysis, leveraging Python's scientific stack (Pandas, Numpy, Uproot) and the Gemini LLM for AI-powered assistance. Explore CERN Open Data and run complex analysis pipelines with natural language commands.</p>
    </div>
    """.format(st.session_state.data_source, st.session_state.current_data_type), unsafe_allow_html=True)
    
    st.markdown("### Quick Start")
    st.markdown("""
    1. **📚 Examples Gallery**: Load a curated dataset like the Higgs $H \to 4\ell$ or Z Boson.
    2. **📤 Upload Data**: Upload your own CSV or ROOT file, or fetch a dataset URL.
    3. **🤖 AI Analysis**: Ask natural language questions like: 
       * *"Calculate the $M_{\mu\mu}$ invariant mass for the two highest $p_T$ muons and plot the distribution between 60 and 120 GeV."*
       * *"Apply a $p_T$ cut of 25 GeV on all leptons and show the resulting event count."*
    """)
    
# --- Examples Gallery Page ---
elif page == "📚 Examples Gallery":
    st.markdown('<div class="main-header">📚 Examples Gallery</div>', unsafe_allow_html=True)
    
    examples = get_all_examples()
    example_options = {v['title']: k for k, v in examples.items()}
    
    selected_title = st.selectbox("Choose a Standard Model or Higgs Example:", list(example_options.keys()))
    
    if selected_title:
        example_id = example_options[selected_title]
        example = get_example(example_id)
        
        st.markdown(f"### {example['title']}")
        st.markdown(f"**Source:** [{example['source']}]({example['source']})")
        st.markdown(example['description'])
        
        st.markdown("#### Suggested Data Files")
        data_col1, data_col2 = st.columns(2)
        
        for i, file_info in enumerate(example['data_files']):
            col = data_col1 if i % 2 == 0 else data_col2
            
            with col:
                st.markdown(f"**{file_info['name']}**")
                
                # Use a unique key for each button
                if st.button(f"Load Data: {file_info['name']}", key=f"load_btn_{file_info['name']}"):
                    file_url = file_info['url']
                    file_extension = file_url.split('.')[-1].lower()
                    
                    with st.spinner(f"Fetching {file_info['name']}..."):
                        try:
                            if file_extension in ['csv']:
                                df, _ = fetch_data_from_url(file_url)
                                load_data_into_state(df, f"Example: {file_info['name']}")
                            elif file_extension in ['root']:
                                # This is simplistic and assumes the ROOT file has a predictable TTree name
                                temp_path, _ = fetch_data_from_url(file_url) 
                                root_info = get_root_file_info(temp_path)
                                
                                # Try to load the first TTree if available
                                if root_info['trees']:
                                    tree_name = root_info['trees'][0]['name']
                                    df = read_tree_to_dataframe(temp_path, tree_name)
                                    load_data_into_state(df, f"Example: {file_info['name']} (TTree: {tree_name})")
                                else:
                                    st.warning("Could not automatically find a TTree in the example ROOT file. Please use the 'Upload Data' page for manual TTree/Histogram selection.")
                                    
                        except Exception as e:
                            st.error(f"Failed to load example data: {str(e)}")

        st.markdown("#### Suggested AI Prompts")
        for prompt in example['suggested_prompts']:
            st.code(prompt)


# --- Upload Data Page (Enhanced for ROOT) ---
elif page == "📤 Upload Data":
    st.markdown('<div class="main-header">📤 Upload or Fetch Data</div>', unsafe_allow_html=True)
    st.markdown("Upload your own data files or provide a URL to fetch data from online sources.")
    
    upload_tab, url_tab, cern_tab = st.tabs(["📁 Upload File", "🔗 From URL", "🌐 CERN Dataset"])
    
    with upload_tab:
        st.markdown("### Upload Data File")
        st.markdown("Supported formats: CSV, ROOT")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'root'], 
            help="Upload CSV or ROOT format particle physics data"
        )
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            # Use a time-based unique path to avoid conflicts
            temp_path = f'/tmp/{int(time.time())}_{uploaded_file.name}' 
            
            try:
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    load_data_into_state(df, uploaded_file.name, "DataFrame")
                    st.markdown(f"**Events**: {len(df)}")
                    st.markdown(f"**Columns**: {', '.join(df.columns.tolist())}")
                    st.dataframe(df.head(10))
                    
                elif file_extension == 'root':
                    # Save ROOT file temporarily for uproot processing
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✅ Saved ROOT file temporarily: {uploaded_file.name}")
                    
                    root_info = get_root_file_info(temp_path)
                    
                    st.markdown("### 🔍 ROOT File Contents")
                    trees = [t['name'] for t in root_info['trees']]
                    histograms = [h['name'] for h in root_info['histograms']]
                    st.markdown(f"**TTrees found**: {', '.join(trees) if trees else 'None'}")
                    st.markdown(f"**Histograms found**: {', '.join(histograms) if histograms else 'None'}")
                    
                    
                    # --- Selection Logic for TTree vs. Histogram ---
                    
                    selection_col, button_col = st.columns([3, 1])

                    action = selection_col.radio(
                        "Select an object to load for analysis:",
                        ["Load TTree (for DataFrame Analysis)", "Load Histogram (for Plotting)"],
                        key=f"root_load_action_{uploaded_file.name}"
                    )
                    
                    selected_object = None
                    if action == "Load TTree (for DataFrame Analysis)" and trees:
                        selected_object = selection_col.selectbox("Choose a TTree to load:", trees)
                    elif action == "Load Histogram (for Plotting)" and histograms:
                        selected_object = selection_col.selectbox("Choose a Histogram to load:", histograms)
                    
                    
                    if selected_object and button_col.button(f"Load {action.split()[1]}"):
                        with st.spinner(f"Reading {action.split()[1]} '{selected_object}'..."):
                            if action == "Load TTree (for DataFrame Analysis)":
                                df = read_tree_to_dataframe(temp_path, selected_object)
                                load_data_into_state(df, f"{uploaded_file.name} (TTree: {selected_object})", "DataFrame")
                                st.markdown(f"**Events**: {len(df)}")
                                st.markdown(f"**Columns**: {', '.join(df.columns.tolist())}")
                                st.dataframe(df.head(10))

                            elif action == "Load Histogram (for Plotting)":
                                hist_data = read_histogram(temp_path, selected_object)
                                load_data_into_state(None, f"{uploaded_file.name} (Hist: {selected_object})", "ROOT_Object", hist_data)
                                
                                # Display the loaded histogram
                                st.markdown(f"### Histogram: {hist_data['title']}")
                                if hist_data['type'] == 'TH1':
                                    fig, ax = plt.subplots()
                                    ax.hist(hist_data['edges'][:-1], hist_data['edges'], weights=hist_data['values'], histtype='stepfilled', edgecolor='black', alpha=0.7)
                                    ax.set_title(hist_data['title'])
                                    ax.set_xlabel("Bin Value") 
                                    ax.set_ylabel("Counts")
                                    st.pyplot(fig)
                                    
                                elif hist_data['type'] == 'TH2':
                                    # Plotting 2D histogram data
                                    fig = go.Figure(data=go.Heatmap(
                                        z=np.array(hist_data['values']),
                                        x=np.array(hist_data['x_edges']),
                                        y=np.array(hist_data['y_edges']),
                                        colorscale='Viridis'
                                    ))
                                    fig.update_layout(title=hist_data['title'])
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                
                    else:
                        st.info("Select an object and click 'Load' to analyze.")
                        
            except Exception as e:
                st.error(f"Error during file processing: {str(e)}")
                load_data_into_state(None, "Error", "None")
    
    with url_tab:
        st.markdown("### Fetch Data from URL")
        st.warning("Only use direct links to CSV or ROOT files. Large ROOT files may fail due to size/timeout limits.")
        url_input = st.text_input("Enter URL (e.g., https://example.com/data.csv)")
        
        if st.button("Fetch Data from URL") and url_input:
            with st.spinner(f"Fetching data from {url_input}..."):
                try:
                    data, file_type = fetch_data_from_url(url_input)
                    if file_type == 'csv':
                        load_data_into_state(data, f"URL: {url_input}", "DataFrame")
                    elif file_type == 'root':
                        # Special handling for ROOT fetched by URL (temp_path is returned)
                        root_info = get_root_file_info(data) # 'data' is the temp_path
                        if root_info['trees']:
                            tree_name = root_info['trees'][0]['name']
                            df = read_tree_to_dataframe(data, tree_name)
                            load_data_into_state(df, f"URL: {url_input} (TTree: {tree_name})", "DataFrame")
                        else:
                            st.warning("Fetched ROOT file has no TTree. Use 'Upload Data' to select a Histogram.")
                            
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")

    with cern_tab:
        st.markdown("### Download CERN Open Data")
        st.markdown("Example: Record 5200 (Higgs $\\to 4\ell$) | File: `4mu_2012.csv`")
        record_id = st.text_input("CERN Record ID (e.g., 5200)", value="5200")
        file_name = st.text_input("File Name (e.g., 4mu_2012.csv)", value="4mu_2012.csv")
        
        if st.button("Download CERN File") and record_id and file_name:
            with st.spinner(f"Downloading {file_name} from CERN Record {record_id}..."):
                try:
                    filepath = download_cern_dataset(record_id, file_name)
                    if file_name.endswith('.csv'):
                        df = load_csv_data(filepath)
                        load_data_into_state(df, f"CERN {record_id}: {file_name}", "DataFrame")
                    else:
                        st.warning("Downloaded non-CSV file. Please check 'Upload Data' for manual loading/selection.")
                except Exception as e:
                    st.error(f"Failed to download CERN file: {str(e)}")


# --- Data Explorer Page ---
elif page == "🔍 Data Explorer":
    st.markdown('<div class="main-header">🔍 Data Explorer</div>', unsafe_allow_html=True)
    
    if st.session_state.current_data_type == "DataFrame":
        df = st.session_state.current_data
        st.info(f"Currently analyzing event-level data from: **{st.session_state.data_source}** | Events: **{len(df)}**")
        
        st.markdown("### Dataset Preview")
        st.dataframe(df.head())
        
        st.markdown("### 📈 Basic Statistics")
        stats = {
            "Total Events": len(df),
            "Number of Variables": len(df.columns),
            "Numerical Variables": len(df.select_dtypes(include=[np.number]).columns),
            "Categorical Variables": len(df.select_dtypes(include=['object']).columns)
        }
        st.dataframe(pd.DataFrame(stats.items(), columns=["Statistic", "Value"]))
        
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

    elif st.session_state.current_data_type == "ROOT_Object":
        hist_data = st.session_state.current_root_object
        st.info(f"Currently viewing loaded histogram data from: **{st.session_state.data_source}**")
        
        st.markdown(f"### Histogram Data: {hist_data['title']}")
        if hist_data['type'] == 'TH1':
             fig, ax = plt.subplots(figsize=(10, 6))
             ax.hist(hist_data['edges'][:-1], hist_data['edges'], weights=hist_data['values'], histtype='stepfilled', edgecolor='black', alpha=0.7, color='steelblue')
             ax.set_title(hist_data['title'], fontweight='bold')
             ax.set_xlabel("Bin Value") 
             ax.set_ylabel("Counts")
             st.pyplot(fig)
        elif hist_data['type'] == 'TH2':
            st.markdown("#### 2D Histogram Preview")
            st.json({"Type": "TH2", "Title": hist_data['title'], "X-Bins": len(hist_data['x_edges'])-1, "Y-Bins": len(hist_data['y_edges'])-1})
            # Full 2D plot using Plotly
            fig = go.Figure(data=go.Heatmap(
                z=np.array(hist_data['values']).T, # Transpose for correct visualization
                x=hist_data['x_edges'][:-1],
                y=hist_data['y_edges'][:-1],
                colorscale='Viridis'
            ))
            fig.update_layout(title=hist_data['title'])
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ No data loaded. Please upload data or load an example first.")


# --- AI Analysis Page (Enhanced for Declarative Pipeline) ---
elif page == "🤖 AI Analysis":
    st.markdown('<div class="main-header">🤖 AI-Powered Analysis Assistant</div>', unsafe_allow_html=True)
    
    if not os.getenv('GEMINI_API_KEY'):
        st.warning("⚠️ Gemini API key not configured. Please add your `GEMINI_API_KEY` to your environment variables to use AI analysis features.")

    # Check if event-level data (DataFrame) is loaded
    if st.session_state.current_data_type == "DataFrame":
        df = st.session_state.current_data
        st.info(f"Loaded Data: **{st.session_state.data_source}** | Events: **{len(df)}** | Columns: **{len(df.columns)}**")
        st.markdown("### Analysis Request")
        
        user_prompt = st.text_area(
            "Enter your analysis request in natural language:", 
            value="Calculate the invariant mass M_ll for the two highest $p_T$ leptons (lep1_pt, lep1_eta, lep1_phi, lep2_pt, lep2_eta, lep2_phi) assuming they are electrons (ELECTRON_MASS). Apply a cut requiring lep1_pt > 25 GeV. Then, plot a histogram of M_ll between 0 and 150 GeV, and perform a Gaussian fit in the range 80-100 GeV."
        )
        
        if st.button("Run AI Analysis"):
            if not os.getenv('GEMINI_API_KEY'):
                st.error("Cannot run analysis: GEMINI_API_KEY is not set.")
            else:
                with st.spinner("🧠 AI is generating and executing the analysis pipeline... This may take a moment."):
                    result = analyze_with_ai(user_prompt, df)
                
                if result['success']:
                    st.success("✅ AI Analysis Complete")
                    
                    st.markdown("### 📝 Explanation and Physics Context")
                    st.markdown(result['explanation'])
                    
                    # Display Results
                    st.markdown("### 📊 Results")
                    
                    # Check for plot result
                    if 'Plot' in result['results']:
                        st.pyplot(result['results']['Plot'])
                    
                    # Check for statistics
                    if 'Statistics' in result['results']:
                        st.markdown("#### Statistical Report")
                        st.dataframe(result['results']['Statistics'])
                    
                    # Check for pipeline steps (calculations/filters)
                    st.markdown("#### Analysis Pipeline Steps Summary")
                    # Exclude plot and stats objects for cleaner display
                    pipeline_summary = {k: v for k, v in result['results'].items() if k not in ['Plot', 'Statistics', 'Plot Error']}
                    st.json(pipeline_summary)

                    with st.expander("🔍 View Full Declarative Specification (AI Output)"):
                        st.json(result['specification'])
                else:
                    st.error(f"Analysis Error: {result.get('error', 'Unknown error during AI analysis.')}")

    elif st.session_state.current_data_type == "ROOT_Object":
        st.info(f"A pre-made Histogram (`{st.session_state.current_root_object['name']}`) is currently loaded. AI Analysis requires event-level data (DataFrame) for calculations and filtering.")
        st.warning("Please load a TTree from your ROOT file or a CSV file to use the AI Analysis feature.")

    else:
        st.warning("⚠️ No event data loaded. Please upload data or load an example first.")

    st.markdown("---")
    st.markdown("### Example Prompts for Advanced Analysis")
    st.markdown("""
    - **Overlap Plots**: "Plot the invariant mass M_ll for events with $p_T(lep1)>25$ GeV (label 'Signal') and overlap it with a plot where $p_T(lep1)<10$ GeV (label 'Background'). Use different colors."
    - **Derived Variables**: "Calculate $\Delta R$ between the two leptons (lep1_eta, lep1_phi, lep2_eta, lep2_phi) and plot a histogram of the result."
    - **Transverse Mass**: "Calculate the transverse mass $M_T$ using `lep1_pt`, `lep1_phi`, `met`, and `met_phi`. Plot the distribution up to 200 GeV."
    """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [CERN Open Data](https://opendata.cern.ch/)
- [CMS Open Data](https://cms-opendata-guide.web.cern.ch/)
- [ATLAS Open Data](https://atlas-opendata.web.cern.ch/)
""")

st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
Built by Dr. Wasikul Islam (https://wasikatcern.github.io), with Streamlit and Python for particle physics data analysis and education. Contact: wasikul.islam@cern.ch
""")
