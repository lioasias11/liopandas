import os, sys, tempfile
# Add project root to sys.path so we can import liopandas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import liopandas as lp
import shapely.geometry as sg
import numpy as np

# Set page config for a premium look
st.set_page_config(page_title="LioPandas Ultimate Showcase", layout="wide", page_icon="🐼")

# --- CUSTOM CSS FOR CLEAN SOFT LIGHT LOOK ---
st.markdown("""
    <style>
    /* Clean Light Background */
    .stApp {
        background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
        color: #1e293b;
    }
    
    /* Soft Glass Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 25px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(99, 102, 241, 0.1);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Crisp Typography */
    h1 {
        color: #1e293b !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -1px !important;
        border-bottom: 3px solid #6366f1;
        display: inline-block;
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    h2, h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Refined Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px 10px 0 0;
        padding: 5px 15px;
        color: #64748b !important;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-bottom: 3px solid #6366f1 !important;
        color: #6366f1 !important;
        font-weight: bold;
    }
    
    /* Elegant Alerts */
    .stAlert {
        border-radius: 12px !important;
        background: #f8fafc !important;
        border-left: 5px solid #6366f1 !important;
        color: #1e293b !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Table borders */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🐼 LioPandas: The Ultimate Feature Showcase")

# Sidebar
st.sidebar.title("🐼 Navigation")
st.sidebar.info(f"Engine v{lp.__version__}")
st.sidebar.markdown("""
**Core Philosophy:**
- NumPy-optimized
- O(1) Lookup
- Scalable Offline Mapping
- Zero-Dependency Core
""")

# Sample Data
@st.cache_data
def get_main_df():
    return lp.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        "age": [28, 34, 22, 45, 31, 29],
        "salary": [72000, 93000, 55000, 110000, 85000, 68000],
        "dept": ["Eng", "Eng", "Sales", "Sales", "Eng", "HR"],
        "rating": [4.5, 3.8, 4.2, 5.0, 3.2, 4.7]
    })

df_main = get_main_df()

tabs = st.tabs([
    "🚀 Overview", 
    "📊 Series", 
    "🏗️ DataFrame", 
    "🧹 Cleaning", 
    "🔗 Merging", 
    "📈 Analytics", 
    "⚖️ Diff Engine", 
    "🌍 Geospatial",
    "💾 I/O"
])

# 1. OVERVIEW: THE FUNDAMENTALS
with tabs[0]:
    st.header("Built from the Ground Up")
    st.write("""
    LioPandas is a from-scratch implementation of a high-performance data analysis library. 
    It bypasses the overhead of standard libraries by focusing on four core engineering pillars:
    """)
    
    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("⚡ NumPy-First Core")
        st.write("""
        All data is stored in **homogeneous NumPy arrays**. 
        This ensures that arithmetic and boolean operations utilize vectorized CPU instructions for maximum throughput.
        """)
    with p2:
        st.subheader("🔑 O(1) Label Indexing")
        st.write("""
        Our custom `Index` class manages labels using a **hash-map backed lookup system**. 
        Label-to-position translation stays constant regardless of dataset size.
        """)
    with p3:
        st.subheader("🌍 Offline Intelligence")
        st.write("""
        Built with a **zero-cloud dependency** geospatial engine. 
        It extracts raster tiles directly from local MBTiles and renders maps on the CPU.
        """)

    st.divider()
    
    col_arch1, col_arch2 = st.columns([2, 1])
    with col_arch1:
        st.subheader("The Modular Architecture")
        st.write("""
        - **`Index`**: The immutable backbone providing label-based axis control.
        - **`Series`**: A labeled 1D array; the atomic unit of data.
        - **`DataFrame`**: A collection of Series sharing a common Index; stored in a memory-efficient column-oriented format.
        - **`GroupBy`**: A specialized engine for split-apply-combine logic.
        - **`Offline`**: A raster-tile extraction engine for air-gapped mapping.
        """)
    with col_arch2:
        st.info("💡 **Design Goal**: Provide a familiar Pandas-like API while maintaining a footprint small enough for embedded and Edge AI environments.")
        st.metric("Core Components", "5 Modules")
        st.metric("Engine Heritage", "Pure NumPy")

# 2. SERIES
with tabs[1]:
    st.header("One-Dimensional Vector Mastery")
    s = lp.Series([100, 200, 150, 400, 200], index=["t1", "t2", "t3", "t4", "t5"], name="inventory")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Value Counts & Uniqueness")
        st.write("Counts of each unique value:")
        st.write("Applied `.value_counts().to_pandas()`:")
        st.dataframe(s.value_counts().to_pandas())
        st.write("Applied `.unique().tolist()`:")
        st.write(f"Got Unique values: `{s.unique().tolist()}`")
        st.write("Applied `.nunique()`:")
        st.write(f"Got Number of unique values: `{s.nunique()}`")
        
    with col_s2:
        st.subheader("Functional Application")
        st.write("Original Series:")
        st.dataframe(s.to_pandas())
        st.write("Applied `.apply(np.sqrt)`:")
        st.write(s.apply(np.sqrt).to_pandas())
        st.write("Applied `.map({100: 'Out-of-Stock', 400: 'Overstock'})`:")
        st.write(s.map({100: 'Out-of-Stock', 400: 'Overstock'}).to_pandas())

    st.subheader("Statistical Summary")
    stats = s.describe()
    st.dataframe(stats.to_pandas())

# 3. DATAFRAME
with tabs[2]:
    st.header("Tabular Structural Control")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("Index & Shape")
        st.write("Applied `.shape`:")
        st.write(f"DataFrame Shape: `{df_main.shape}`")
        st.write("Data Types (`.dtypes`):")
        st.dataframe(df_main.dtypes.to_pandas())
        
        st.write("Resetting Index:")
        st.write("Applied `.reset_index(drop=True)`:")
        st.dataframe(df_main.reset_index(drop=True).head(3).to_pandas())
        
    with col_d2:
        st.subheader("Dropping & Renaming")
        st.write("Dropping 'rating' and 'salary' columns:")
        st.write("Applied `.drop(columns=['rating', 'salary'])`:")
        st.dataframe(df_main.drop(columns=['rating', 'salary']).to_pandas())
        
        st.write("Renaming columns:")
        st.write("Applied `.rename(columns={'dept': 'organization'})`:")
        st.dataframe(df_main.rename(columns={'dept': 'organization'}).to_pandas())

    st.subheader("Positional vs Label Slicing")
    st.write("`.iloc[1:3, :2]` (Position 1 to 3, First 2 columns):")
    st.dataframe(df_main.iloc[1:3, :2].to_pandas())

# 4. CLEANING
with tabs[3]:
    st.header("Robust Data Sanitation")
    
    dirty_data = {
        "A": [1, 2, np.nan, 4, 4, None],
        "B": ["X", "Y", "Z", "X", "X", "W"],
        "C": [10.5, 20.1, 30.0, 40.0, 40.0, 60.5]
    }
    df_dirty = lp.DataFrame(dirty_data)
    
    st.write("Raw Dirty Data (with NAs and Duplicates):")
    st.dataframe(df_dirty.to_pandas())
    
    cl1, cl2, cl3 = st.columns(3)
    with cl1:
        st.subheader("Handling Missing")
        st.write("`fillna(0.0)`:")
        st.dataframe(df_dirty.fillna(0.0).to_pandas())
        st.write("`dropna()` (rows with any NA):")
        st.dataframe(df_dirty.dropna().to_pandas())
        
    with cl2:
        st.subheader("Deduplication")
        st.write("`drop_duplicates(subset='B')`:")
        st.dataframe(df_dirty.drop_duplicates(subset='B').to_pandas())
        
    with cl3:
        st.subheader("Validation")
        st.write("`isna()` Matrix:")
        st.dataframe(df_dirty.isna().to_pandas())
        st.write("`notna()` Matrix:")
        st.dataframe(df_dirty.notna().to_pandas())

# 5. MERGING
with tabs[4]:
    st.header("Relational Joins & Set Algebra")
    
    df1 = lp.DataFrame({"id": [1, 2, 3], "val1": ["red", "blue", "green"]})
    df2 = lp.DataFrame({"id": [2, 3, 4], "val2": ["circle", "square", "triangle"]})
    
    m_type = st.radio("Join Type", ["inner", "left", "right", "outer"], horizontal=True)
    merged_res = lp.merge(df1, df2, on="id", how=m_type)
    
    cm1, cm2, cm3 = st.columns([1, 1, 2])
    cm1.write("Left DF")
    cm1.dataframe(df1.to_pandas())
    cm2.write("Right DF")
    cm2.dataframe(df2.to_pandas())
    cm3.write(f"Result ({m_type})")
    cm3.dataframe(merged_res.to_pandas())

# 6. ANALYTICS
with tabs[5]:
    st.header("Deep Analytics Pipeline")
    
    st.subheader("GroupBy Aggregations")
    col_agg1, col_agg2 = st.columns([1, 3])
    with col_agg1:
        target_group = st.selectbox("Group By", ["dept", "age"])
        target_func = st.selectbox("Metric", ["mean", "sum", "max", "min", "count", "std"])
    with col_agg2:
        res = df_main.groupby(target_group).agg(target_func)
        st.dataframe(res.to_pandas(), use_container_width=True)
        
    st.divider()
    
    st.subheader("Custom Map/Apply Transformations")
    st.write("Adding a 'Status' column based on Age:")
    status_df = df_main.copy()
    status_df['status'] = status_df['age'].apply(lambda x: 'Senior' if x > 30 else 'Junior')
    st.dataframe(status_df.to_pandas())

# 7. DIFF ENGINE
with tabs[6]:
    st.header("DataFrame Diff / Change Detection")
    st.write("Visual comparison of two datasets.")
    
    df_v1 = df_main.copy()
    df_v2 = df_main.copy()
    
    # Simulate changes
    df_v2.loc[1, 'salary'] = 95000  # Bob raise
    df_v2.loc[4, 'rating'] = 1.0    # Eve bad rating
    
    cd1, cd2 = st.columns(2)
    with cd1:
        st.subheader("Version 1 (Gold)")
        st.dataframe(df_v1.to_pandas(), use_container_width=True)
    with cd2:
        st.subheader("Version 2 (Modified)")
        st.dataframe(df_v2.to_pandas(), use_container_width=True)
        
    st.divider()
    st.subheader("Detected Cell-Level Differences")
    diff_report = df_v1.compare(df_v2)
    st.dataframe(diff_report.to_pandas(), use_container_width=True)

# 8. GEOSPATIAL
with tabs[7]:
    st.header("Static Intelligence Mapping")
    
    # Gaza HQ Example
    gaza_data = {
        'city': ['Gaza Center', 'Al Shati', 'Al Rimal', 'Sheikh Radwan', 'Tel al-Hawa'],
        'population': [590481, 590481, 590481, 590481, 590481],
        'geometry': [
            sg.Point(34.4668, 31.5017), sg.Point(34.4505, 31.5225), 
            sg.Point(34.4750, 31.5150), sg.Point(34.4620, 31.5300), 
            sg.Point(34.4500, 31.4900)
        ]
    }
    gdf_gaza = lp.DataFrame(gaza_data)
    
    gc1, gc2, gc3 = st.columns(3)
    zoom_lvl = gc1.slider("Zoom Level", 6, 13, 10, key="gz_zoom")
    show_ctx = gc2.toggle("Show Regional Side-Map", value=True)
    show_cty = gc3.toggle("Label Cities", value=False)
    
    if st.button("🗺️ Render High-Resolution Offline Map"):
        with st.spinner("Decoding MBTiles sectors..."):
            os.makedirs('plots', exist_ok=True)
            p = 'plots/showcase_map_v2.png'
            gdf_gaza.plot_static(
                output_path=p,
                zoom=zoom_lvl,
                show_context=show_ctx,
                show_city_labels=show_cty
            )
            st.image(p, caption="LioPandas Offline Intelligence Output", use_container_width=True)

# 9. I/O
with tabs[8]:
    st.header("The Connectivity Suite")
    
    st.subheader("NumPy Interface")
    arr = df_main.to_numpy()
    st.write(f"NumPy Array Export (first 2 rows):")
    st.write(arr[:2])
    
    st.divider()
    
    st.subheader("Pandas Interop")
    st.info("Seamless conversion between LioPandas and the standard ecosystem.")
    p_df = df_main.to_pandas()
    st.write(f"Converted to `{type(p_df)}` successfully.")
    
    st.divider()
    
    st.subheader("CSV Serialization")
    if st.button("Export to CSV"):
        t_path = os.path.join(tempfile.gettempdir(), "liopandas_ultimate.csv")
        df_main.to_csv(t_path)
        st.success(f"File stored at: {t_path}")
        st.write("Preview of re-read data:")
        st.dataframe(lp.read_csv(t_path).head(2).to_pandas())
