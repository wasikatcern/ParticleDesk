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
# NOTE: The 'ai_analysis' module is assumed to be available as per the original structure.
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
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    [data-testid="stSidebar"] {
        min-width: 300px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---

if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'data_name' not in st.session_state:
    st.session_state.data_name = "None"
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Overview"

# Initialize 5 chat sessions for the notebook style
if 'analysis_sessions' not in st.session_state:
    st.session_state.analysis_sessions = {}
    for i in range(1, 6):
        st.session_state.analysis_sessions[f'Session {i}'] = {
            'history': [{'role': 'assistant', 'content': 'Hello! I am your AI Particle Data Analyst. Ask me to plot, summarize, or calculate quantities from your loaded dataset.'}],
            'results': [],
            'current_prompt': ''
        }
    st.session_state.current_session_key = 'Session 1'


# --- Sidebar Navigation ---
st.sidebar.image("https://placehold.co/300x80/1f77b4/ffffff?text=ParticleDesk+⚛️", use_column_width=True)

page_options = ["Data Overview", "📤 Upload Data", "📚 Examples Gallery", "🤖 AI-Powered Analysis", "🔬 Advanced Tools"]
selected_page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.current_page))
st.session_state.current_page = selected_page

st.sidebar.markdown(f"**Loaded Dataset:** `{st.session_state.data_name}`")
if st.session_state.data is not None:
    st.sidebar.markdown(f"**Shape:** {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")

# --- Main Page Content ---

if selected_page == "AI-Powered Analysis":
    st.title("🤖 AI-Powered Analysis (Notebook Style)")
    st.markdown("Use the 5 analysis tabs below to run different queries on your loaded dataset. Each tab is a persistent chat session.")
    
    df = st.session_state.data
    
    if df is None:
        st.warning("⚠️ Please load a dataset first using '📤 Upload Data' or '📚 Examples Gallery'.")
    else:
        tab_titles = [f'Session {i}' for i in range(1, 6)]
        tabs = st.tabs(tab_titles)
        
        # Define the current active tab and session
        for i, tab in enumerate(tabs):
            session_key = tab_titles[i]
            with tab:
                st.session_state.current_session_key = session_key
                current_session = st.session_state.analysis_sessions[session_key]

                # Display chat history for the current session
                for message in current_session['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
                # Display results (plots, stats) if available
                for result in current_session['results']:
                    if 'plot' in result:
                        st.image(f"data:image/png;base64,{result['plot']}", caption=f"Plot: {result['function']}", use_column_width=True)
                    elif 'result' in result:
                         st.info(f"**{result['function'].replace('_', ' ').title()}:**\n\n{result['result']}")
                
                # Chat input
                prompt = st.chat_input("Ask for analysis (e.g., 'Plot invariant mass M_4l', 'How many variables in data?')", key=f'prompt_{session_key}')

                if prompt:
                    # Append user message to history
                    current_session['history'].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Start assistant thinking message
                    with st.chat_message("assistant"):
                        with st.spinner(f"Analyzing prompt '{prompt}'..."):
                            analysis_output = analyze_with_ai(prompt, df)

                    # Process AI response
                    if analysis_output['success']:
                        explanation = analysis_output['explanation']
                        results = analysis_output['results']
                        
                        # Add explanation to history
                        current_session['history'].append({"role": "assistant", "content": explanation})
                        
                        # Display explanation immediately
                        with tabs[i].chat_message("assistant"):
                            st.markdown(explanation)
                        
                        # Store results for display
                        current_session['results'] = results
                        
                        # Display new results in the current tab
                        for result in results:
                            if 'plot' in result:
                                # Display image in the chat flow
                                with tabs[i].chat_message("assistant"):
                                    st.image(f"data:image/png;base64,{result['plot']}", caption=f"Plot: {result['function']}", use_column_width=True)
                            elif 'result' in result:
                                # Display result in the chat flow
                                with tabs[i].chat_message("assistant"):
                                    st.info(f"**{result['function'].replace('_', ' ').title()}:**\n\n{result['result']}")
                    else:
                        error_msg = f"AI Analysis Failed. Error: **{analysis_output['error']}**"
                        current_session['history'].append({"role": "assistant", "content": error_msg})
                        with tabs[i].chat_message("assistant"):
                            st.error(error_msg)
                            
                    # Rerun to update history and results display cleanly
                    st.rerun()
                    
elif selected_page == "Data Overview":
    st.title("Data Overview and Exploration")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown(f"### Current Data: **{st.session_state.data_name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='info-box'>Total Events (Rows): <strong>{df.shape[0]}</strong></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='info-box'>Total Variables (Columns): <strong>{df.shape[1]}</strong></div>", unsafe_allow_html=True)
        
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📝 Column Information")
        stats = {}
        for col in df.columns:
            stats[col] = {
                "Type": str(df[col].dtype),
                "Missing": df[col].isnull().sum(),
                "Unique": df[col].nunique(),
            }
        st.dataframe(pd.DataFrame(stats).T)

        st.markdown("### 🔢 Numerical Feature Statistics")
        df_numeric = df.select_dtypes(include=np.number)
        if not df_numeric.empty:
            st.dataframe(df_numeric.describe().T.style.format(precision=4), use_container_width=True)
        else:
            st.info("No numerical features to display statistics for.")

        # Optional: Distribution plots for first few numerical columns
        st.markdown("### 📈 Distributions of Key Variables")
        if not df_numeric.empty:
            num_cols_to_plot = df_numeric.columns[:min(len(df_numeric.columns), 4)]
            
            for col in num_cols_to_plot:
                st.markdown(f"#### {col}")
                try:
                    fig = px.histogram(
                        df_numeric, 
                        x=col, 
                        nbins=50, 
                        title=f'Distribution of {col}',
                        color_discrete_sequence=['#1f77b4'] # Streamlit blue
                    )
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not plot {col}: {e}")


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


# --- Placeholder for other pages (Upload Data, Examples Gallery, Advanced Tools) ---
# NOTE: The actual implementation of these pages is omitted for brevity but should be functional.

elif selected_page == "📤 Upload Data":
    st.title("📤 Upload Data")
    st.markdown("Use this section to upload a local CSV file or fetch data from a public URL.")
    # Placeholder for upload/fetch logic
    # The actual implementation of data loading is kept in the original app.py but omitted here.

elif selected_page == "📚 Examples Gallery":
    st.title("📚 Examples Gallery")
    st.markdown("Load pre-curated datasets from CERN Open Data for quick analysis.")
    # Placeholder for examples logic
    # The actual implementation of data loading is kept in the original app.py but omitted here.

elif selected_page == "🔬 Advanced Tools":
    st.title("🔬 Advanced Tools")
    st.markdown("Future home for tools like cut optimization, mass reconstruction utilities, and advanced plotting.")
    st.info("Coming soon!")


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

st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
Built by Dr. Wasikul Islam (https://wasikatcern.github.io), with Streamlit and Python for particle physics data analysis.
""")
