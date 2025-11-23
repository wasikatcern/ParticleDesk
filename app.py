import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os
import json

# Import custom modules
from examples_data import get_all_examples, get_example
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
    /* Style for the conversational chat interface */
    .stChatFloatingInputContainer {
        border-top: 1px solid #ccc;
        padding-top: 10px;
    }
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 15px 15px 5px 15px;
        margin: 5px 0;
        text-align: right;
        display: inline-block;
        max-width: 80%;
        float: right;
    }
    .ai-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 15px 15px 15px 5px;
        margin: 5px 0;
        text-align: left;
        display: inline-block;
        max-width: 80%;
        float: left;
    }
    .st-emotion-cache-1jm694k, .st-emotion-cache-1uj2t3h {
        max-width: none;
    }
</style>
"""
, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'file_info' not in st.session_state:
    st.session_state['file_info'] = "No data loaded."
if 'ai_chat_history' not in st.session_state:
    st.session_state['ai_chat_history'] = []
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = []
if 'current_df' not in st.session_state:
    st.session_state['current_df'] = None


# --- Helper Functions for Data Loading ---

def load_data_from_df(df, info):
    st.session_state['df'] = df
    st.session_state['current_df'] = df.copy()
    st.session_state['file_info'] = info
    st.session_state['ai_chat_history'] = [] # Clear history on new data load
    st.session_state['analysis_results'] = []
    st.success(f"Successfully loaded {len(df)} events.")
    st.rerun()

def load_example_data(example_id):
    example = get_example(example_id)
    if not example or not example.get('data_files'):
        st.error("Example data configuration is incomplete.")
        return

    st.info(f"Loading data for: {example['title']}...")
    
    # For simplicity, load the first file (assuming it's the main data file)
    file_to_load = example['data_files'][0]
    file_url = file_to_load['url']
    
    try:
        if file_url.endswith('.csv'):
            df, _ = fetch_data_from_url(file_url)
            info = f"Loaded CSV data from {file_to_load['name']}"
            load_data_from_df(df, info)
        else:
             st.error("Only CSV loading is supported for direct examples in this version.")
    except Exception as e:
        st.error(f"Error loading example data: {e}")

# --- AI Analysis Tab Logic ---

def handle_ai_prompt(prompt):
    if not st.session_state.get('df') is None:
        
        # 1. Append user message to history
        st.session_state['ai_chat_history'].append({"role": "user", "content": prompt})
        
        # 2. Call the AI analysis function
        with st.spinner("⚛️ Analyzing data with Gemini..."):
            
            # The AI analysis function now returns a more detailed result,
            # including the conversational response from the AI.
            analysis_result = analyze_with_ai(prompt, st.session_state['current_df'])
        
        # 3. Append AI response to history
        st.session_state['ai_chat_history'].append({"role": "ai", "content": analysis_result.get('explanation', 'Error: Could not process request.')})
        
        if analysis_result['success']:
            # 4. Store results and update the current DataFrame if filtering occurred
            st.session_state['analysis_results'].extend(analysis_result['results'])
            
            # Check if a filtering step changed the DataFrame
            temp_df = st.session_state['current_df'].copy()
            for step in analysis_result['results']:
                if step['function'] == 'filter_data' and 'new_df' in step:
                    temp_df = step['new_df']
            
            # Only update the session state if the DF changed its shape or a filtering step occurred
            # This is a bit of a placeholder since `analyze_with_ai` needs to be updated to manage the DF.
            # In the new ai_analysis.py, we manage current_df inside the function.
            st.session_state['current_df'] = temp_df
            
            st.toast(f"Analysis step completed! New shape: {analysis_result['final_df_shape']}")
        else:
            # If AI analysis failed, provide the error message.
            st.error(f"AI Analysis Failure: {analysis_result['error']}")
            st.toast("Analysis failed. Check the error message.")

        st.rerun()
    else:
        st.warning("Please load a dataset before starting the AI analysis.")

# --- Tab Setup ---
tab_titles = ["🏠 Home", "📤 Upload Data", "📚 Examples Gallery", "🧠 AI Analysis", "📊 Data Overview"]
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.markdown("<h1 class='main-header'>ParticleDesk: Interactive HEP Analysis</h1>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to ParticleDesk, your simplified platform for High Energy Physics (HEP) data analysis.
    Load data from CERN Open Data or upload your own, then use the **AI Analysis** tab to run complex analyses using natural language.
    
    ### Current Status
    - **Data Loaded:** {}
    - **Events:** {}
    """.format(
        "Yes" if st.session_state['df'] is not None else "No",
        len(st.session_state['df']) if st.session_state['df'] is not None else 0
    ))
    
    if st.session_state['df'] is None:
        st.info("Start by going to the **📚 Examples Gallery** to load a dataset, or **📤 Upload Data**.")

with tabs[1]:
    st.markdown("## 📤 Upload Data")
    st.info("Upload your CSV or a ROOT file. For ROOT files, only a sample of entries will be read for performance.")
    
    uploaded_file = st.file_uploader("Choose a file (CSV or ROOT)", type=['csv', 'root'])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        
        with st.spinner(f"Loading {file_type} data..."):
            try:
                if file_type == 'csv':
                    df = load_csv_data(uploaded_file)
                    info = f"Loaded CSV file: {uploaded_file.name}"
                    load_data_from_df(df, info)

                elif file_type == 'root':
                    # Temporarily save the uploaded file to disk for uproot to read
                    temp_filepath = os.path.join('/tmp', uploaded_file.name)
                    with open(temp_filepath, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    # Get info and let user select tree/entries
                    file_info = get_root_file_info(temp_filepath)
                    tree_names = [t['name'] for t in file_info['trees']]
                    
                    if not tree_names:
                        st.error("No TTrees found in the ROOT file.")
                    else:
                        selected_tree = st.selectbox("Select TTree to read:", tree_names)
                        max_entries = st.number_input("Max entries to read:", min_value=1, value=50000, step=1000)
                        
                        if st.button("Load Tree"):
                            df = read_tree_to_dataframe(temp_filepath, selected_tree, max_entries=max_entries)
                            info = f"Loaded ROOT tree '{selected_tree}' from {uploaded_file.name} ({len(df)} entries)."
                            load_data_from_df(df, info)

            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")

with tabs[2]:
    st.markdown("## 📚 Examples Gallery")
    st.markdown("Load pre-curated datasets from CERN Open Data to start your analysis instantly.")
    
    examples = get_all_examples()
    example_ids = list(examples.keys())
    
    cols = st.columns(len(example_ids))
    
    for i, example_id in enumerate(example_ids):
        example = examples[example_id]
        with cols[i]:
            st.subheader(example['title'])
            st.caption(f"Source: {example['source']}")
            st.markdown(example['description'])
            
            if st.button(f"Load {example['title']}", key=f"load_ex_{example_id}"):
                load_example_data(example_id)

with tabs[3]:
    st.markdown("## 🧠 AI Analysis Assistant (Conversational)")
    st.info("Type your analysis request (e.g., 'Plot the four-lepton mass between 100 and 150 GeV with 50 bins and mark the Higgs mass at 125 GeV').")
    
    if st.session_state['df'] is None:
        st.warning("Please load a dataset in the 'Upload Data' or 'Examples Gallery' tab to begin AI analysis.")
    else:
        
        # --- Multi-Prompt Chat Interface ---
        
        # Main container for chat history
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        if not st.session_state['ai_chat_history']:
            # Initial greeting
            st.markdown(f"""
            <div class='ai-message'>
                Hello! I'm your Particle Physics AI Analyst. The current dataset has {len(st.session_state['df'])} events and columns like {st.session_state['df'].columns.tolist()[:5]}...
                How can I help you analyze this data?
                Try asking me to plot something, like: "Plot the M column as a histogram from 100 to 150 GeV in 50 bins."
            </div><div style="clear:both;"></div>
            """, unsafe_allow_html=True)
        
        # Display chat history (Jupyter-style cells)
        for i, message in enumerate(st.session_state['ai_chat_history']):
            
            # User Message
            if message["role"] == "user":
                st.markdown(f"""
                <div class='user-message'>
                    **[Input {i+1}]:** {message["content"]}
                </div><div style="clear:both;"></div>
                """, unsafe_allow_html=True)
            
            # AI Response
            elif message["role"] == "ai":
                st.markdown(f"""
                <div class='ai-message'>
                    **[Output {i+1}]:** {message["content"]}
                </div><div style="clear:both;"></div>
                """, unsafe_allow_html=True)
                
                # Check for and display analysis results associated with this output step
                # NOTE: This assumes the history index corresponds roughly to the results index. 
                # A more robust system would link them explicitly, but for now, we rely on sequential appending.
                if i < len(st.session_state['analysis_results']):
                    result = st.session_state['analysis_results'][i]
                    if result.get('plot'):
                        # Display matplotlib figure
                        st.markdown("#### Generated Plot:")
                        st.pyplot(result['plot'])
                    
                    # Display statistics or filtering reports
                    if result.get('function') in ['get_stats', 'filter_data', 'calculate_mass']:
                        st.markdown("#### Analysis Report:")
                        st.code(result.get('result', 'No detailed report available.'))

        st.markdown("</div>", unsafe_allow_html=True)

        # Input box for new prompt
        st.markdown("<div class='stChatFloatingInputContainer'>", unsafe_allow_html=True)
        prompt = st.chat_input("Enter your analysis request (e.g., 'Filter M > 120 GeV'):")
        if prompt:
            handle_ai_prompt(prompt)
        st.markdown("</div>", unsafe_allow_html=True)


with tabs[4]:
    st.markdown("## 📊 Data Overview")
    df = st.session_state['df']
    if df is not None:
        st.markdown(f"### Current Data: {st.session_state['file_info']}")
        st.dataframe(df.head(10))
        
        # Display basic stats
        st.markdown("### 🔢 Descriptive Statistics")
        st.dataframe(df.describe().T)

        # Plot individual column histograms (This is what you mentioned works)
        st.markdown("### 📈 Single Variable Histograms")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column to plot:", numeric_cols, index=numeric_cols.index('M') if 'M' in numeric_cols else 0)
            
            fig = px.histogram(
                df, 
                x=selected_col, 
                title=f"Distribution of {selected_col}", 
                nbins=50,
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
