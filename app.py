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
        margin: 1rem 0;
    }
    .citation {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Sidebar navigation
st.sidebar.markdown("# ⚛️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📚 Examples Gallery", "📤 Upload Data", "🤖 AI Analysis", "📊 Data Explorer"]
)

# Main content based on page selection
if page == "🏠 Home":
    st.markdown('<div class="main-header">⚛️ ParticleDesk : Particle Physics Data Analysis Platform</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Particle Physics Data Analysis Platform
    
    This application enables you to analyze particle physics data from CERN Open Data and other sources 
    using modern data science tools and AI-powered analysis assistance.
    
    #### 🎯 Features:
    - **📚 Example Gallery**: Explore curated analyses from CMS experiments including the Higgs discovery
    - **📤 Data Upload**: Upload your own CSV or ROOT files, or fetch data from URLs
    - **🤖 AI Analysis**: Use natural language to request analyses and generate plots
    - **📊 Interactive Visualizations**: Create publication-quality plots with matplotlib and plotly
    - **🔬 Physics Tools**: Built-in calculations for invariant mass, kinematics, and more
    
    #### 📖 Supported Data Sources:
    - **CERN Open Data Portal**: https://opendata.cern.ch/
    - **CMS Open Data Analyses**: https://github.com/cms-opendata-analyses
    - Custom CSV and ROOT files
    - Direct URL links to datasets
    
    #### 🚀 Getting Started:
    1. Visit the **Examples Gallery** to see pre-configured analyses
    2. Try the **Higgs → 4ℓ** example to see the historic Higgs discovery
    3. Upload your own data or provide a URL
    4. Use **AI Analysis** to explore your data with natural language prompts
    
    #### 📚 Example Prompts:
    - "Plot the four-lepton invariant mass distribution"
    - "Show histogram of transverse momentum"
    - "Calculate the Z boson mass peak"
    - "Display the Higgs mass around 125 GeV"
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎓 Educational")
        st.markdown("""
        Perfect for learning about:
        - Particle physics data analysis
        - The Higgs boson discovery
        - Statistical methods
        - Data visualization
        """)
    
    with col2:
        st.markdown("### 🔬 Research")
        st.markdown("""
        Tools for:
        - Exploring CERN datasets
        - Reproducing published analyses
        - Testing new analysis ideas
        - Statistical modeling
        """)
    
    with col3:
        st.markdown("### 🤖 AI-Powered")
        st.markdown("""
        Features:
        - Natural language queries
        - Automated plot generation
        - Analysis suggestions
        - Code generation
        """)

elif page == "📚 Examples Gallery":
    st.markdown('<div class="main-header">📚 Analysis Examples Gallery</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore curated particle physics analyses from CERN Open Data and CMS experiments.
    Each example includes source citations, datasets, and suggested analysis prompts.
    """)
    
    examples = get_all_examples()
    
    # Create tabs for different examples
    example_tabs = st.tabs([ex["title"] for ex in examples.values()])
    
    for idx, (example_id, example_data) in enumerate(examples.items()):
        with example_tabs[idx]:
            st.markdown(f"### {example_data['title']}")
            
            # Description
            st.markdown(example_data['description'])
            
            # Source information
            st.markdown("#### 📖 Sources & Citations")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'source' in example_data:
                    st.markdown(f"**CERN Record**: [{example_data['source']}]({example_data['source']})")
                if 'github' in example_data:
                    st.markdown(f"**GitHub**: [{example_data['github']}]({example_data['github']})")
            
            with col2:
                if 'publication' in example_data:
                    st.markdown(f"**Publication**: {example_data['publication']}")
                if 'dataset_url' in example_data:
                    st.markdown(f"**Dataset**: [{example_data['dataset_url']}]({example_data['dataset_url']})")
            
            # Data files
            if example_data.get('data_files'):
                st.markdown("#### 📁 Available Data Files")
                for file_info in example_data['data_files']:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{file_info['name']}**")
                    with col2:
                        if st.button(f"Load", key=f"load_{example_id}_{file_info['name']}"):
                            with st.spinner(f"Downloading {file_info['name']}..."):
                                try:
                                    # Download the file
                                    filepath = download_cern_dataset("5200", file_info['name'])
                                    df = load_csv_data(filepath)
                                    st.session_state.current_data = df
                                    st.session_state.data_source = file_info['name']
                                    st.success(f"Loaded {file_info['name']} with {len(df)} events!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error loading data: {str(e)}")
            
            # Suggested prompts
            st.markdown("#### 💡 Suggested Analysis Prompts")
            st.markdown("Use these prompts in the **AI Analysis** section:")
            for prompt in example_data['suggested_prompts']:
                st.markdown(f"- {prompt}")
            
            # Column information
            if example_data.get('columns_info'):
                with st.expander("📋 Data Column Information"):
                    for col_name, col_desc in example_data['columns_info'].items():
                        st.markdown(f"**{col_name}**: {col_desc}")

elif page == "📤 Upload Data":
    st.markdown('<div class="main-header">📤 Upload or Fetch Data</div>', unsafe_allow_html=True)
    
    st.markdown("Upload your own data files or provide a URL to fetch data from online sources.")
    
    # Create tabs for different input methods
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
            
            try:
                if file_extension == 'csv':
                    # Load CSV file
                    df = pd.read_csv(uploaded_file)
                    st.session_state.current_data = df
                    st.session_state.data_source = uploaded_file.name
                    
                    st.success(f"✅ Loaded {uploaded_file.name}")
                    st.markdown(f"**Events**: {len(df)}")
                    st.markdown(f"**Columns**: {', '.join(df.columns.tolist())}")
                    
                    # Preview
                    st.markdown("### Data Preview")
                    st.dataframe(df.head(10))
                    
                elif file_extension == 'root':
                    # Save ROOT file temporarily
                    temp_path = f'/tmp/{uploaded_file.name}'
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.read())
                    
                    # Get ROOT file info
                    info = get_root_file_info(temp_path)
                    
                    st.success(f"✅ Opened ROOT file: {uploaded_file.name}")
                    
                    # Display trees
                    if info['trees']:
                        st.markdown("### Available Trees")
                        selected_tree = st.selectbox(
                            "Select a tree to load",
                            options=[tree['name'] for tree in info['trees']]
                        )
                        
                        # Find selected tree info
                        tree_info = next(t for t in info['trees'] if t['name'] == selected_tree)
                        
                        st.markdown(f"**Entries**: {tree_info['num_entries']}")
                        st.markdown(f"**Branches**: {len(tree_info['branches'])}")
                        
                        # Option to select branches
                        with st.expander("Select Branches"):
                            selected_branches = st.multiselect(
                                "Choose branches to load (empty = all)",
                                options=tree_info['branches']
                            )
                        
                        max_entries = st.number_input(
                            "Maximum entries to load",
                            min_value=100,
                            max_value=tree_info['num_entries'],
                            value=min(10000, tree_info['num_entries']),
                            step=1000
                        )
                        
                        if st.button("Load Tree Data"):
                            with st.spinner("Loading ROOT tree..."):
                                try:
                                    root_file = open_root_file(temp_path)
                                    branches = selected_branches if selected_branches else None
                                    df = read_tree_to_dataframe(
                                        root_file,
                                        selected_tree,
                                        branches=branches,
                                        max_entries=int(max_entries)
                                    )
                                    
                                    st.session_state.current_data = df
                                    st.session_state.data_source = f"{uploaded_file.name}:{selected_tree}"
                                    
                                    st.success(f"Loaded {len(df)} entries")
                                    st.dataframe(df.head(10))
                                    
                                except Exception as e:
                                    st.error(f"Error loading tree: {str(e)}")
                    else:
                        st.warning("No TTrees found in this ROOT file")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with url_tab:
        st.markdown("### Fetch Data from URL")
        st.markdown("Provide a direct link to a CSV or ROOT file")
        
        url_input = st.text_input(
            "Enter URL",
            placeholder="https://opendata.cern.ch/record/5200/files/4mu_2012.csv"
        )
        
        if st.button("Fetch Data"):
            if url_input:
                with st.spinner("Fetching data..."):
                    try:
                        result, file_type = fetch_data_from_url(url_input)
                        
                        if file_type == 'csv':
                            st.session_state.current_data = result
                            st.session_state.data_source = url_input
                            
                            st.success(f"✅ Fetched data from URL")
                            st.markdown(f"**Events**: {len(result)}")
                            st.dataframe(result.head(10))
                        
                        elif file_type == 'root':
                            st.info("ROOT file downloaded. Use the file upload interface to process it.")
                            st.session_state.data_source = url_input
                        
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
            else:
                st.warning("Please enter a URL")
    
    with cern_tab:
        st.markdown("### Download from CERN Open Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            record_id = st.text_input("CERN Record ID", value="5200")
        
        with col2:
            filename = st.text_input("Filename", value="4mu_2012.csv")
        
        if st.button("Download from CERN"):
            with st.spinner(f"Downloading {filename}..."):
                try:
                    filepath = download_cern_dataset(record_id, filename)
                    df = load_csv_data(filepath)
                    st.session_state.current_data = df
                    st.session_state.data_source = f"CERN:{record_id}/{filename}"
                    
                    st.success(f"✅ Downloaded and loaded {filename}")
                    st.markdown(f"**Events**: {len(df)}")
                    st.dataframe(df.head(10))
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "🤖 AI Analysis":
    st.markdown('<div class="main-header">🤖 AI-Powered Analysis Assistant</div>', unsafe_allow_html=True)
    
    # Check if Gemini API key is available (using Google Gemini - FREE)
    if not os.getenv('GEMINI_API_KEY'):
        st.warning("⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to use AI analysis features.")
        st.markdown("""
        To use FREE AI analysis with Google Gemini:
        1. Go to https://ai.google.dev/
        2. Click "Get API Key" (requires a Google account, completely FREE - no payment needed)
        3. Create a new API key in Google Cloud
        4. On Render: Go to your project → Environment → Add GEMINI_API_KEY
        5. Paste your API key and deploy
        """)
    else:
        st.markdown("""
        Use natural language to analyze your particle physics data. The AI assistant can:
        - Generate plots and visualizations
        - Calculate physics quantities (invariant mass, momentum, etc.)
        - Suggest analysis strategies
        - Explain physics concepts
        """)
        
        # Check if data is loaded
        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            
            st.info(f"📊 Current Dataset: **{st.session_state.data_source}** ({len(df)} events)")
            
            # Show available columns
            with st.expander("📋 Available Data Columns"):
                cols_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
                    cols_info.append({
                        "Column": col,
                        "Type": dtype,
                        "Sample": sample_val
                    })
                st.dataframe(pd.DataFrame(cols_info))
            
            # AI prompt input
            st.markdown("### 💬 Ask the AI Assistant")
            
            user_prompt = st.text_area(
                "Enter your analysis request:",
                placeholder="Example: Plot the invariant mass distribution between 70 and 200 GeV",
                height=100
            )
            
            if st.button("🚀 Analyze", type="primary"):
                if user_prompt:
                    with st.spinner("AI is analyzing your request..."):
                        # Import AI module
                        from ai_analysis import analyze_with_ai
                        
                        try:
                            result = analyze_with_ai(user_prompt, df)
                            
                            if result['success']:
                                st.success("✅ Analysis complete!")
                                
                                # Display explanation
                                if 'explanation' in result:
                                    st.markdown("#### 📝 Analysis")
                                    st.markdown(result['explanation'])
                                
                                # Display plot if available
                                if 'figure' in result:
                                    st.pyplot(result['figure'])
                                
                                # Display results
                                if 'results' in result and result['results']:
                                    st.markdown("#### 📊 Statistical Results")
                                    st.write(result['results'])
                                
                                # Show analysis specification
                                if 'specification' in result:
                                    with st.expander("🔍 View Analysis Details"):
                                        st.json(result['specification'])
                                
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                else:
                    st.warning("Please enter an analysis request")
            
            # Example prompts
            st.markdown("### 💡 Example Prompts")
            example_prompts = [
                "Plot histogram of M (invariant mass) column",
                "Show the distribution of pt1 (transverse momentum)",
                "Calculate the mean and standard deviation of the invariant mass",
                "Create a scatter plot of eta1 vs phi1",
                "Plot the four-lepton mass with bins between 70-200 GeV"
            ]
            
            for prompt in example_prompts:
                if st.button(f"📝 {prompt}", key=f"prompt_{prompt}"):
                    st.session_state.selected_prompt = prompt
                    st.rerun()
        
        else:
            st.warning("⚠️ No data loaded. Please upload data or load an example first.")
            st.markdown("Go to **📤 Upload Data** or **📚 Examples Gallery** to load a dataset.")

elif page == "📊 Data Explorer":
    st.markdown('<div class="main-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        st.success(f"📊 Dataset: **{st.session_state.data_source}**")
        
        # Dataset summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Events", len(df))
        
        with col2:
            st.metric("Columns", len(df.columns))
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.2f} MB")
        
        st.markdown("---")
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        num_rows = st.slider("Number of rows to display", 5, 100, 10)
        st.dataframe(df.head(num_rows))
        
        # Statistics
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe())
        
        # Column analysis
        st.markdown("### 🔍 Column Analysis")
        
        selected_column = st.selectbox("Select a column to analyze", df.columns)
        
        if selected_column:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df[selected_column].dropna(), bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel(selected_column)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_column}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("#### Statistics")
                stats = {
                    "Count": df[selected_column].count(),
                    "Mean": df[selected_column].mean(),
                    "Std Dev": df[selected_column].std(),
                    "Min": df[selected_column].min(),
                    "25%": df[selected_column].quantile(0.25),
                    "50% (Median)": df[selected_column].median(),
                    "75%": df[selected_column].quantile(0.75),
                    "Max": df[selected_column].max()
                }
                st.dataframe(pd.DataFrame(stats.items(), columns=["Statistic", "Value"]))
        
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
Built by Dr. Wasikul Islam (https://wasikatcern.github.io), with Streamlit and Python for particle physics data analysis and education. Contact: wasikul.islam@cern.ch
""")
