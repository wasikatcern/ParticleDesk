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
import os

# Import custom modules
from examples import get_all_examples
from utils import (
    fetch_data_from_url,
    download_cern_dataset,
    load_csv_data,
    open_root_file,
    get_root_file_info,
    read_tree_to_dataframe,
    MUON_MASS,
    ELECTRON_MASS,
    HIGGS_MASS_EXPECTED,
)

# Page configuration
st.set_page_config(
    page_title="ParticleDesk : Particle Physics Data Analysis Platform",
    page_icon="⚛️",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .footer {
        font-size: 0.8rem;
        color: #888;
        margin-top: 2rem;
    }
    .highlight {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .dataset-card {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    .dataset-card h4 {
        margin-top: 0;
    }
    .citation {
        font-size: 0.9rem;
        color: #666;
    }
    .note {
        font-size: 0.85rem;
        color: #777;
        font-style: italic;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "datasets" not in st.session_state:
    # name -> DataFrame
    st.session_state.datasets = {}

# Sidebar navigation
st.sidebar.markdown("# ⚛️ Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📚 Examples Gallery",
        "📤 Upload Data",
        "🤖 AI Analysis",
        "📊 Data Explorer",
    ],
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.markdown(
    """
**ParticleDesk** is an interactive platform for analyzing particle physics data using:
- CERN Open Data
- CMS and ATLAS examples
- AI-powered analysis assistance (Google Gemini - FREE)

You can:
- Load example Higgs datasets
- Upload your own CSV/ROOT files
- Use natural language to perform analyses
"""
)

# ------------- HOME PAGE -------------
if page == "🏠 Home":
    st.markdown(
        '<div class="main-header">⚛️ ParticleDesk</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">AI-Assisted Particle Physics Data Analysis Platform</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
Welcome to **ParticleDesk**, an interactive platform designed to help you explore and analyze particle physics data,
especially from CERN Open Data. You can:

- Load Higgs boson and other physics datasets
- Explore data visually and statistically
- Use an AI assistant (powered by Google Gemini, FREE) to ask questions and generate analysis code
"""
    )

    st.markdown("### 🚀 Quick Start")
    st.markdown(
        """
1. Go to **📚 Examples Gallery** to load a Higgs→4ℓ dataset
2. Use the *Higgs Boson → Four Leptons* example to reproduce the discovery-style plot
3. Upload your own data or provide a URL
4. Use **AI Analysis** to explore your data with natural language prompts

#### 📚 Example Prompts:
- "Plot the four-lepton invariant mass distribution"
- "Show histogram of transverse momentum"
- "Calculate the Z boson mass peak"
- "Display the Higgs mass around 125 GeV"
"""
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎓 Educational")
        st.markdown(
            """
Perfect for learning about:
- Particle physics data analysis
- The Higgs boson discovery
- Statistical methods
- Data visualization
"""
        )

    with col2:
        st.markdown("### 🔬 Research")
        st.markdown(
            """
Tools for:
- Exploring CERN datasets
- Reproducing published analyses
- Testing new analysis ideas
- Teaching workshops and tutorials
"""
        )

    with col3:
        st.markdown("### 🤖 AI-Powered")
        st.markdown(
            """
Features:
- Natural language queries
- Automated plot generation
- Analysis suggestions
- Code generation
"""
        )

# ------------- EXAMPLES GALLERY -------------
elif page == "📚 Examples Gallery":
    st.markdown(
        '<div class="main-header">📚 Analysis Examples Gallery</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
Explore curated particle physics analyses from CERN Open Data and CMS/ATLAS.
Each example comes with dataset links, documentation, and suggested prompts.
"""
    )

    examples = get_all_examples()
    example_ids = list(examples.keys())

    tabs = st.tabs([examples[eid]["title"] for eid in example_ids])

    for idx, eid in enumerate(example_ids):
        example_data = examples[eid]
        with tabs[idx]:
            st.markdown(f"### {example_data['title']}")
            if "description" in example_data:
                st.markdown(example_data["description"])

            # Sources
            st.markdown("#### 📖 Sources & Citations")
            col1, col2 = st.columns(2)
            with col1:
                if "source" in example_data:
                    st.markdown(
                        f"**CERN Record**: [{example_data['source']}]({example_data['source']})"
                    )
                if "github" in example_data:
                    st.markdown(
                        f"**GitHub**: [{example_data['github']}]({example_data['github']})"
                    )
            with col2:
                if "publication" in example_data:
                    st.markdown(
                        f"**Reference Paper**: [{example_data['publication']}]({example_data['source']})"
                    )
                if "year" in example_data:
                    st.markdown(f"**Year**: {example_data['year']}")

            # Dataset list (for CMS Higgs 4ℓ example this is the candidate CSV files)
            if "data_files" in example_data:
                st.markdown("#### 📦 Data Files")
                for f in example_data["data_files"]:
                    url = f.get("url", "")
                    st.markdown(
                        f"- `{f['name']}`  "
                        + (f"[[download]]({url})" if url else "")
                    )

            # Special wiring for CMS Higgs→4ℓ example: end-to-end loading & splitting
            if eid == "higgs_4l":
                st.markdown("#### ▶️ Load Ready-to-Use Higgs→4ℓ Datasets")

                st.markdown(
                    """
When you click **Load CMS H→4ℓ example**, ParticleDesk will:

1. Download all Higgs→4ℓ candidate CSV files from CMS Open Data (record 5200).
2. Merge them into a single dataset `CMS_Higgs4l_all_candidates`.
3. Build two derived datasets for significance studies:
   - `CMS_Higgs4l_peak_120_130`  → events with 120 ≤ M ≤ 130 GeV (signal-like)
   - `CMS_Higgs4l_sidebands_70_180_excl_peak` → 70 ≤ M ≤ 180 GeV excluding the peak (background-like)
4. Register all three in the global dataset list so you can select them in **AI Analysis**.
"""
                )

                if st.button("Load CMS H→4ℓ example", key="load_higgs4l"):
                    from utils import download_cern_dataset, load_csv_data

                    with st.spinner(
                        "Downloading CMS Higgs→4ℓ candidate CSV files and preparing datasets..."
                    ):
                        try:
                            record_url = example_data.get("dataset_url", "").rstrip(
                                "/"
                            )
                            record_id = record_url.split("/")[-1] if record_url else "5200"

                            dfs = []
                            for f in example_data.get("data_files", []):
                                local_path = download_cern_dataset(
                                    record_id, f["name"]
                                )
                                df_part = load_csv_data(local_path)
                                dfs.append(df_part)

                            if not dfs:
                                st.error("No data files configured for this example.")
                            else:
                                df_all = pd.concat(dfs, ignore_index=True)
                                # Register combined dataset
                                name_all = "CMS_Higgs4l_all_candidates"
                                st.session_state.datasets[name_all] = df_all
                                st.session_state.current_data = df_all
                                st.session_state.data_source = name_all

                                # If invariant mass column exists, build signal-like / background-like splits
                                if "M" in df_all.columns:
                                    peak_mask = (df_all["M"] >= 120.0) & (
                                        df_all["M"] <= 130.0
                                    )
                                    sideband_mask = (
                                        (df_all["M"] >= 70.0)
                                        & (df_all["M"] <= 180.0)
                                        & ~peak_mask
                                    )

                                    df_peak = df_all[peak_mask].copy()
                                    df_side = df_all[sideband_mask].copy()

                                    name_peak = "CMS_Higgs4l_peak_120_130"
                                    name_side = "CMS_Higgs4l_sidebands_70_180_excl_peak"

                                    st.session_state.datasets[name_peak] = df_peak
                                    st.session_state.datasets[name_side] = df_side

                                    st.success(
                                        f"Loaded {len(df_all)} total events "
                                        f"(peak: {len(df_peak)}, sidebands: {len(df_side)})."
                                    )
                                    st.info(
                                        "You can now go to **🤖 AI Analysis** and select:\n"
                                        "- Primary: `CMS_Higgs4l_peak_120_130` (role: signal)\n"
                                        "- Secondary: `CMS_Higgs4l_sidebands_70_180_excl_peak` (role: background)\n"
                                        "Then use the golden prompts below for significance & overlay studies."
                                    )
                                else:
                                    st.warning(
                                        "Merged dataset has no 'M' column; cannot build peak/sideband splits."
                                    )
                        except Exception as e:
                            st.error(f"Error loading CMS Higgs→4ℓ example: {e}")

                st.markdown("#### 🌟 Golden Prompts for CMS H→4ℓ (use in 🤖 AI Analysis)")
                st.markdown(
                    """
These are hand-crafted prompts that exercise **overlay plots**, **significance**, and
**statistics** for the CMS Higgs→4ℓ example. After loading the example, go to **🤖 AI Analysis**,
set:

- Primary dataset: `CMS_Higgs4l_peak_120_130`  (role: **signal**)
- Secondary dataset: `CMS_Higgs4l_sidebands_70_180_excl_peak`  (role: **background**)

Then try:

1. **Overlay + unit-area normalization**  
   > Overlay the M distribution for the signal and background datasets, normalized to unit area, with about 30 bins between 70 and 180 GeV, and label the axes appropriately.

2. **Significance in a fixed window**  
   > Compute S/sqrt(B) for the Higgs peak using M between 120 and 130 GeV, treating the peak dataset as signal and the sidebands dataset as background. Show the counts S and B and the resulting significance.

3. **Scan of different mass windows**  
   > For the M variable, compare S/sqrt(S+B) for three mass windows centered at 125 GeV with widths of ±2 GeV, ±5 GeV, and ±10 GeV, using the peak dataset as signal and the sidebands as background. Summarize the results in a small table.

4. **Detailed statistics of the Higgs peak**  
   > For the peak dataset, compute the mean, standard deviation, median, minimum, and maximum of M, and overlay a histogram of M with the sideband distribution for comparison.

5. **Kinematic cross-check**  
   > Make a 2D histogram of eta1 vs phi1 for the combined CMS_Higgs4l_all_candidates dataset, and also summarize the mean and standard deviation of pt1 for the same events.

You can also ask general questions like:  
   > Explain in simple terms what the four-lepton invariant mass spectrum tells us about the Higgs boson.
"""
                )

            # Generic suggested prompts for other examples
            if eid != "higgs_4l" and example_data.get("suggested_prompts"):
                st.markdown("#### 💡 Suggested Analysis Prompts")
                for p in example_data["suggested_prompts"]:
                    st.markdown(f"- {p}")

            # Column info if provided
            if example_data.get("columns_info"):
                with st.expander("📋 Data Column Information"):
                    for cname, desc in example_data["columns_info"].items():
                        st.markdown(f"**{cname}**: {desc}")

# ------------- UPLOAD DATA -------------
elif page == "📤 Upload Data":
    st.markdown(
        '<div class="main-header">📤 Upload or Fetch Data</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "Upload your own data files or provide a URL to fetch data from online sources."
    )

    upload_tab, url_tab, cern_tab = st.tabs(
        ["📁 Upload File", "🔗 From URL", "🌐 CERN Dataset"]
    )

    # ---- Upload File tab ----
    with upload_tab:
        st.markdown("### Upload Data File")
        st.markdown("Supported formats: CSV, ROOT")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "root"],
            help="Upload CSV or ROOT format particle physics data",
        )

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1].lower()

            try:
                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                    dataset_name = uploaded_file.name
                    st.session_state.datasets[dataset_name] = df
                    st.session_state.current_data = df
                    st.session_state.data_source = dataset_name

                    st.success(f"✅ Loaded {uploaded_file.name}")
                    st.markdown(f"**Events**: {len(df)}")
                    st.markdown(
                        f"**Columns**: {', '.join(df.columns.astype(str).tolist())}"
                    )

                    st.markdown("### Data Preview")
                    st.dataframe(df.head(10))

                elif file_extension == "root":
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    info = get_root_file_info(temp_path)
                    st.success(f"✅ Opened ROOT file: {uploaded_file.name}")

                    if info["trees"]:
                        st.markdown("### Available Trees")
                        selected_tree = st.selectbox(
                            "Select a tree to load",
                            options=[t["name"] for t in info["trees"]],
                        )
                        tree_info = next(
                            t for t in info["trees"] if t["name"] == selected_tree
                        )

                        st.markdown("### Branch Selection")
                        branches = tree_info["branches"]
                        selected_branches = st.multiselect(
                            "Select branches",
                            options=branches,
                            default=branches[: min(20, len(branches))],
                            help="Select branches (variables) to load into the DataFrame",
                        )

                        max_entries = st.number_input(
                            "Maximum number of entries to load (0 for all)",
                            min_value=0,
                            value=0,
                            step=1000,
                        )

                        if st.button("Load Selected Tree"):
                            with st.spinner("Loading ROOT tree..."):
                                try:
                                    root_file = open_root_file(temp_path)
                                    branches_to_load = (
                                        selected_branches if selected_branches else None
                                    )
                                    df = read_tree_to_dataframe(
                                        root_file,
                                        selected_tree,
                                        branches=branches_to_load,
                                        max_entries=int(max_entries),
                                    )

                                    dataset_name = f"{uploaded_file.name}:{selected_tree}"
                                    st.session_state.datasets[dataset_name] = df
                                    st.session_state.current_data = df
                                    st.session_state.data_source = dataset_name

                                    st.success(f"Loaded {len(df)} entries")
                                    st.dataframe(df.head(10))
                                except Exception as e:
                                    st.error(f"Error loading tree: {e}")
                    else:
                        st.warning("No trees found in this ROOT file")
                else:
                    st.error("Unsupported file format")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # ---- From URL tab ----
    with url_tab:
        st.markdown("### Fetch Data from URL")
        st.markdown("Provide a direct link to a CSV or ROOT file")

        url_input = st.text_input(
            "Enter URL",
            placeholder="https://opendata.cern.ch/record/5200/files/4mu_2012.csv",
        )

        if st.button("Fetch Data"):
            if url_input:
                with st.spinner("Fetching data..."):
                    try:
                        df_or_path, file_type = fetch_data_from_url(url_input)
                        if file_type == "csv":
                            df = df_or_path
                            dataset_name = url_input
                            st.session_state.datasets[dataset_name] = df
                            st.session_state.current_data = df
                            st.session_state.data_source = dataset_name
                            st.success("✅ Fetched data from URL")
                            st.markdown(f"**Events**: {len(df)}")
                            st.dataframe(df.head(10))
                        elif file_type == "root":
                            st.info(
                                "ROOT file downloaded. Use the file upload interface to open and inspect it."
                            )
                            st.session_state.data_source = url_input
                        else:
                            st.error(
                                "Unsupported or unknown file type returned from URL."
                            )
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
            else:
                st.warning("Please enter a URL")

    # ---- CERN Dataset tab ----
    with cern_tab:
        st.markdown("### Download from CERN Open Data")

        col1, col2 = st.columns(2)
        with col1:
            record_id = st.text_input("CERN Record ID", value="5200")
        with col2:
            filename = st.text_input("Filename", value="4mu_2012.csv")

        if st.button("Download from CERN"):
            with st.spinner(f"Downloading {filename} from CERN Open Data..."):
                try:
                    filepath = download_cern_dataset(record_id, filename)
                    df = load_csv_data(filepath)
                    dataset_name = f"CERN:{record_id}/{filename}"
                    st.session_state.datasets[dataset_name] = df
                    st.session_state.current_data = df
                    st.session_state.data_source = dataset_name

                    st.success(f"✅ Downloaded and loaded {filename}")
                    st.markdown(f"**Events**: {len(df)}")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"Error downloading from CERN: {e}")

# ------------- AI ANALYSIS -------------
elif page == "🤖 AI Analysis":
    st.markdown(
        '<div class="main-header">🤖 AI-Powered Analysis Assistant</div>',
        unsafe_allow_html=True,
    )

    if not os.getenv("GEMINI_API_KEY"):
        st.warning(
            "⚠️ Gemini API key not configured. Please add your GEMINI_API_KEY to use AI analysis features."
        )
        st.markdown(
            """
To use FREE AI analysis with Google Gemini:

1. Go to https://ai.google.dev/
2. Click "Get API Key" (requires a Google account, free tier is sufficient)
3. Create a new API key in Google Cloud
4. Set the GEMINI_API_KEY environment variable for your deployment
"""
        )
    else:
        st.markdown(
            """
Use natural language to analyze your particle physics data. The AI assistant can:
- Generate plots and visualizations
- Calculate physics quantities (invariant mass, momentum, etc.)
- Suggest analysis strategies
- Explain physics concepts
"""
        )

        datasets = st.session_state.get("datasets", {})
        current_df = st.session_state.current_data

        def _guess_role(name: str) -> int:
            """Heuristic index for role selectbox: 0=unlabeled,1=signal,2=background."""
            lname = name.lower()
            if any(
                k in lname
                for k in ["sig", "signal", "peak", "higgs4l", "higgs4", "cms_higgs4l"]
            ):
                return 1
            if any(k in lname for k in ["bkg", "background", "sideband", "zz"]):
                return 2
            return 0

        # Case 1: multiple datasets (modern path)
        if datasets:
            dataset_names = list(datasets.keys())
            st.markdown("### 📊 Dataset Selection")

            default_primary = (
                st.session_state.data_source
                if st.session_state.data_source in dataset_names
                else dataset_names[0]
            )
            primary_index = (
                dataset_names.index(default_primary)
                if default_primary in dataset_names
                else 0
            )
            primary_name = st.selectbox(
                "Primary dataset (e.g. signal)", dataset_names, index=primary_index
            )
            primary_df = datasets[primary_name]
            st.session_state.current_data = primary_df
            st.session_state.data_source = primary_name

            secondary_options = ["None"] + [
                n for n in dataset_names if n != primary_name
            ]
            secondary_name = st.selectbox(
                "Secondary dataset (optional, e.g. background)", secondary_options
            )
            secondary_df = datasets[secondary_name] if secondary_name != "None" else None

            role_options = ["unlabeled", "signal", "background"]
            primary_role = st.selectbox(
                "Role of primary dataset",
                role_options,
                index=_guess_role(primary_name),
            )

            secondary_role = None
            if secondary_df is not None:
                secondary_role = st.selectbox(
                    "Role of secondary dataset",
                    role_options,
                    index=_guess_role(secondary_name),
                )

            st.info(
                f"📊 Primary Dataset: **{primary_name}** ({len(primary_df)} events)"
            )
            if secondary_df is not None:
                st.info(
                    f"📊 Secondary Dataset: **{secondary_name}** ({len(secondary_df)} events)"
                )

            # Show columns for primary
            with st.expander("📋 Available Data Columns (Primary)"):
                rows = []
                for col in primary_df.columns:
                    s = primary_df[col]
                    rows.append(
                        {
                            "Column": col,
                            "Type": str(s.dtype),
                            "Sample": s.iloc[0] if len(s) > 0 else "N/A",
                        }
                    )
                st.dataframe(pd.DataFrame(rows))

            st.markdown("### 💬 Ask the AI Assistant")
            user_prompt = st.text_area(
                "Enter your analysis request:",
                placeholder="Example: Overlay signal and background M distributions and compute S/sqrt(B) around 125 GeV",
                height=140,
            )

            if st.button("🚀 Analyze", type="primary"):
                if user_prompt:
                    with st.spinner("AI is analyzing your request..."):
                        from ai_analysis import analyze_with_ai

                        try:
                            meta = {
                                "primary_name": primary_name,
                                "primary_role": primary_role,
                            }
                            if secondary_df is not None:
                                meta["secondary_name"] = secondary_name
                                meta["secondary_role"] = (
                                    secondary_role or "unlabeled"
                                )

                            result = analyze_with_ai(
                                user_prompt,
                                primary_df,
                                secondary_df=secondary_df,
                                meta=meta,
                            )

                            if result.get("success", False):
                                st.success("✅ Analysis complete!")
                                if result.get("explanation"):
                                    st.markdown("#### 📝 Analysis")
                                    st.markdown(result["explanation"])
                                if result.get("figure") is not None:
                                    st.pyplot(result["figure"])
                                if result.get("results"):
                                    st.markdown("#### 📊 Statistical Results")
                                    st.write(result["results"])
                                if result.get("specification"):
                                    with st.expander("🔍 View Analysis Details"):
                                        st.json(result["specification"])
                            else:
                                st.error(result.get("error", "Unknown error"))
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
                else:
                    st.warning("Please enter an analysis request")

            st.markdown("### 💡 Generic Example Prompts")
            generic_prompts = [
                "Plot histogram of M (invariant mass) between 70 and 180 GeV with 30 bins.",
                "Show the distribution of pt1 and compute its mean and standard deviation.",
                "Create a scatter plot of eta1 vs phi1.",
                "Compute S/sqrt(B) for M between 120 and 130 GeV using the chosen signal and background datasets.",
            ]
            for p in generic_prompts:
                st.markdown(f"- {p}")

        # Case 2: single legacy dataset
        elif current_df is not None:
            df = current_df
            st.info(
                f"📊 Current Dataset: **{st.session_state.data_source}** ({len(df)} events)"
            )

            with st.expander("📋 Available Data Columns"):
                rows = []
                for col in df.columns:
                    s = df[col]
                    rows.append(
                        {
                            "Column": col,
                            "Type": str(s.dtype),
                            "Sample": s.iloc[0] if len(s) > 0 else "N/A",
                        }
                    )
                st.dataframe(pd.DataFrame(rows))

            st.markdown("### 💬 Ask the AI Assistant")
            user_prompt = st.text_area(
                "Enter your analysis request:",
                placeholder="Example: Plot the invariant mass distribution between 70 and 200 GeV",
                height=140,
            )

            if st.button("🚀 Analyze", type="primary"):
                if user_prompt:
                    with st.spinner("AI is analyzing your request..."):
                        from ai_analysis import analyze_with_ai

                        try:
                            result = analyze_with_ai(user_prompt, df)
                            if result.get("success", False):
                                st.success("✅ Analysis complete!")
                                if result.get("explanation"):
                                    st.markdown("#### 📝 Analysis")
                                    st.markdown(result["explanation"])
                                if result.get("figure") is not None:
                                    st.pyplot(result["figure"])
                                if result.get("results"):
                                    st.markdown("#### 📊 Statistical Results")
                                    st.write(result["results"])
                                if result.get("specification"):
                                    with st.expander("🔍 View Analysis Details"):
                                        st.json(result["specification"])
                            else:
                                st.error(result.get("error", "Unknown error"))
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
                else:
                    st.warning("Please enter an analysis request")

        # Case 3: chat-only (no data loaded yet)
        else:
            st.info(
                "No dataset loaded yet. You can still chat with the AI assistant about physics concepts, time, date, etc."
            )
            st.markdown("### 💬 Ask the AI Assistant")
            user_prompt = st.text_area(
                "Enter your question or greeting:",
                placeholder="Example: Hi! What is the Higgs boson? Or: What time is it in Geneva?",
                height=120,
            )

            if st.button("💬 Chat", type="primary"):
                if user_prompt:
                    with st.spinner("AI is responding..."):
                        from ai_analysis import analyze_with_ai

                        try:
                            dummy_df = pd.DataFrame({"dummy": [0.0]})
                            result = analyze_with_ai(user_prompt, dummy_df)
                            if result.get("success", False):
                                st.markdown("#### 🧠 Assistant")
                                st.markdown(result.get("explanation", ""))
                            else:
                                st.error(result.get("error", "Unknown error"))
                        except Exception as e:
                            st.error(f"Error during AI chat: {e}")
                else:
                    st.warning("Please enter a message for the AI assistant")

# ------------- DATA EXPLORER -------------
elif page == "📊 Data Explorer":
    st.markdown(
        '<div class="main-header">📊 Data Explorer</div>', unsafe_allow_html=True
    )

    datasets = st.session_state.get("datasets", {})
    df = None
    label = None

    if datasets:
        dataset_names = list(datasets.keys())
        default_name = (
            st.session_state.data_source
            if st.session_state.data_source in dataset_names
            else dataset_names[0]
        )
        default_index = (
            dataset_names.index(default_name) if default_name in dataset_names else 0
        )
        selected_name = st.selectbox(
            "Select dataset to explore", dataset_names, index=default_index
        )
        df = datasets[selected_name]
        st.session_state.current_data = df
        st.session_state.data_source = selected_name
        label = selected_name
    elif st.session_state.current_data is not None:
        df = st.session_state.current_data
        label = st.session_state.data_source or "current"

    if df is not None:
        st.success(f"📊 Dataset: **{label}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.2f} MB")

        st.markdown("---")
        st.markdown("### 📋 Data Preview")
        nrows = st.slider("Number of rows to display", 5, 100, 10)
        st.dataframe(df.head(nrows))

        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe())

        st.markdown("### 🔍 Column Analysis")
        selected_column = st.selectbox("Select a column to analyze", df.columns)
        if selected_column:
            colL, colR = st.columns(2)
            with colL:
                st.markdown("#### Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(
                    df[selected_column].dropna(),
                    bins=50,
                    edgecolor="black",
                    alpha=0.7,
                )
                ax.set_xlabel(selected_column)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {selected_column}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            with colR:
                st.markdown("#### Statistics")
                series = df[selected_column]
                stats = {
                    "Count": series.count(),
                    "Mean": series.mean(),
                    "Std Dev": series.std(),
                    "Min": series.min(),
                    "25%": series.quantile(0.25),
                    "50% (Median)": series.median(),
                    "75%": series.quantile(0.75),
                    "Max": series.max(),
                }
                st.dataframe(
                    pd.DataFrame(
                        list(stats.items()), columns=["Statistic", "Value"]
                    )
                )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("### 🔗 Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                labels=dict(color="Correlation"),
                x=numeric_cols,
                y=numeric_cols,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "⚠️ No data loaded. Please upload data or load an example first."
        )
        st.markdown(
            "Go to **📤 Upload Data** or **📚 Examples Gallery** to load a dataset."
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown(
    """
- [CERN Open Data](https://opendata.cern.ch/)
- [CMS Open Data Guide](https://cms-opendata-guide.web.cern.ch/)
- [GitHub: CMS Open Data Analyses](https://github.com/cms-opendata-analyses)
"""
)

st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown(
    """
Built by Dr. Wasikul Islam (https://wasikatcern.github.io),
with a commitment to particle physics data analysis and education.
Contact: wasikul.islam@cern.ch
"""
)
