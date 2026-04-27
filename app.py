import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import folium
from streamlit_folium import st_folium
import os
import geopy
from geopy.geocoders import Nominatim
import requests
import polyline
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Traffic AI Optimization", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* Main Background - Sleek Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #18181b 0%, #27272a 100%) !important;
        font-family: 'Inter', sans-serif;
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }
    /* Streamlit basic text */
    p, span, label, div.stMarkdown p {
        color: #e2e8f0 !important;
    }
    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #f97316, #ea580c) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(234, 88, 12, 0.3) !important;
        transition: all 0.3s ease !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 20px rgba(234, 88, 12, 0.4) !important;
    }
    /* Cards for dataframe/map (Dark Glassmorphism) */
    div[data-testid="stDataFrame"], .folium-map {
        border-radius: 16px !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4) !important;
        overflow: hidden !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(10px) !important;
    }
    /* Warning/Info boxes */
    div.stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(5px) !important;
    }
    /* Make top header match dark theme instead of hiding it */
    header {background-color: transparent !important;}
    /* Custom Header Card */
    .hero-card {
        background: linear-gradient(135deg, #9a3412, #ea580c); 
        padding: 30px; 
        border-radius: 20px; 
        box-shadow: 0 15px 35px rgba(234, 88, 12, 0.3); 
        text-align: center; 
        color: white; 
        margin-bottom: 30px;
        margin-top: -10px;
    }
    .hero-card h1 {
        color: white !important; 
        margin: 0; 
        font-size: 2.8rem; 
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .hero-card p {
        font-size: 1.2rem; 
        opacity: 0.9; 
        margin-top: 10px;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'active_lanes_closed' not in st.session_state:
    st.session_state.active_lanes_closed = 0

# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h2 style="color: #fb923c !important; font-weight: 800; font-size: 2rem; margin: 0;">Trafix AI 🚦</h2>
        <p style="color: #94a3b8 !important; font-size: 0.9rem; margin-top: 0;">Traffic Flow Simulation</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Data Processing", "ML Training", "Dashboard", "Live Route Map"],
        icons=["cloud-upload", "robot", "speedometer2", "map"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#ea580c", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "border-radius":"10px", "font-family": "'Inter', sans-serif", "font-weight": "600"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #f97316, #ea580c)", "color": "white", "box-shadow": "0 4px 10px rgba(234,88,12,0.3)"},
        }
    )
    st.markdown("---")
    st.markdown("**Version:** 2.0.0 Pro")
    st.markdown("**Status:** System Online 🟢")


st.markdown("""
<div class="hero-card">
    <h1>🚦 Trafix AI Optimizer</h1>
    <p>A complete Data Science Pipeline: Upload Data ➡️ Preprocess ➡️ Train ML Model ➡️ Optimize Lane Closures</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PAGE 1: DATA ----------------
if selected == "Data Processing":
    st.header("1. Upload & Clean Dataset")
    
    col_upload, col_gen = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload your traffic dataset (CSV)", type=["csv"])
    with col_gen:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Demo Dataset (If no CSV)"):
            n = 500
            cities = ["Kyrenia, Cyprus", "Nicosia, Cyprus", "Famagusta, Cyprus", "Larnaca, Cyprus"]
            origins = np.random.choice(cities, n)
            destinations = np.random.choice(cities, n)
            for i in range(n):
                while destinations[i] == origins[i]:
                    destinations[i] = np.random.choice(cities)
            
            times = np.random.randint(0, 24, n)
            lanes = np.random.randint(0, 4, n)
            duration = np.random.choice([15, 30, 45, 60, 90, 120], n)
            density = np.random.uniform(10, 100, n)
            delay = (lanes * 15) + (density * 0.5) + (duration * 0.1) + np.random.normal(0, 5, n)
            delay = np.maximum(0, delay) # no negative delay
            
            demo_df = pd.DataFrame({
                'origin': origins,
                'destination': destinations,
                'time': times,
                'lanes_closed': lanes,
                'duration': duration,
                'traffic_density': np.round(density, 2),
                'delay': np.round(delay, 2)
            })
            peak_mask = ((demo_df["time"] >= 7) & (demo_df["time"] <= 9)) | ((demo_df["time"] >= 16) & (demo_df["time"] <= 19))
            demo_df.loc[peak_mask, "delay"] *= 1.5
            demo_df["delay"] += np.random.normal(0, 5, n)
            demo_df["delay"] = np.clip(demo_df["delay"], 0, None).round(2)
            
            # Intentionally add some nulls and duplicates to prove data cleaning works
            demo_df.loc[10:15, "traffic_density"] = np.nan
            demo_df = pd.concat([demo_df, demo_df.iloc[:5]], ignore_index=True)
            
            st.session_state.raw_data = demo_df
            st.success("Demo Dataset Generated! It intentionally includes missing values and duplicates for testing.")

    if uploaded_file is not None:
        st.session_state.raw_data = pd.read_csv(uploaded_file)
        
    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Raw Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        st.markdown("---")
        st.subheader("Data Cleaning Options")
        col_c1, col_c2, col_c3 = st.columns(3)
        drop_na = col_c1.checkbox("Drop Missing Values (NaN)", value=True)
        drop_dup = col_c2.checkbox("Drop Duplicates", value=True)
        
        if st.button("Perform Data Preprocessing", type="primary"):
            clean_df = df.copy()
            if drop_na:
                clean_df = clean_df.dropna()
            if drop_dup:
                clean_df = clean_df.drop_duplicates()
                
            st.session_state.cleaned_data = clean_df
            st.success("Data Preprocessing Complete!")
            
        if st.session_state.cleaned_data is not None:
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)
            reduced_rows = df.shape[0] - st.session_state.cleaned_data.shape[0]
            st.info(f"Cleaned Data Shape: {st.session_state.cleaned_data.shape[0]} rows. Removed {reduced_rows} bad rows.")


# ---------------- PAGE 2: MODEL TRAINING ----------------
if selected == "ML Training":
    st.header("2. Application of Data Science (ML)")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please upload and clean a dataset in the Data Tab first.")
    else:
        df_clean = st.session_state.cleaned_data
        numeric_columns = df_clean.select_dtypes(include=np.number).columns.tolist()
        
        col_f1, col_f2 = st.columns(2)
        target_col = col_f1.selectbox("Select Target Variable (e.g., Delay)", numeric_columns, index=len(numeric_columns)-1)
        
        feature_cols = [c for c in numeric_columns if c != target_col]
        selected_features = col_f2.multiselect("Select Input Features (e.g., Time, Lanes Closed)", feature_cols, default=feature_cols)
        
        st.markdown("---")
        st.subheader("Machine Learning Setup")
        model_choice = st.radio("Select Problem Type:", 
                              ["Regression (Predict continuous values e.g., Delay)", 
                               "Classification (Predict categories/discrete values e.g., Lanes Closed)"])
        
        if st.button("Train AI Model", type="primary"):
            if len(selected_features) == 0:
                st.error("Please select at least one feature.")
            else:
                st.session_state.features = selected_features
                st.session_state.target = target_col
                
                X = df_clean[selected_features]
                y = df_clean[target_col]
                
                # Standard Scaling for preprocessing
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state.scaler = scaler
                
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                if "Classification" in model_choice:
                    if pd.api.types.is_float_dtype(y) or len(y.unique()) > 20:
                        st.error(f"⚠️ Error: You are trying to run a Classification model on '{target_col}', which contains continuous decimal numbers! Classifiers can only predict exact categories (like whole lanes). Please switch the radio button to 'Regression'!")
                        st.stop()
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    st.session_state.model_type = "classification"
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    st.session_state.model_type = "regression"
                    
                model.fit(X_train, y_train)
                st.session_state.model = model
                
                preds = model.predict(X_test)
                if st.session_state.model_type == "classification":
                    score = accuracy_score(y_test, preds)
                    st.success(f"Classification Model Trained Successfully! Accuracy: {score*100:.2f}%")
                else:
                    score = r2_score(y_test, preds)
                    st.success(f"Regression Model Trained Successfully! R² Score: {score:.2f}")


# ---------------- PAGE 3: DASHBOARD ----------------
if selected == "Dashboard":
    st.header("3. Traffic Analytics Dashboard & Optimizer")
    
    if st.session_state.model is None:
        st.warning("Please train the model in the Model Tab first.")
    else:
        df_clean = st.session_state.cleaned_data
        
        # 1. Visualization
        st.subheader("Data Insights")
        if "time" in st.session_state.features and st.session_state.target is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(df_clean["time"], df_clean[st.session_state.target], alpha=0.5, c="red")
            ax.set_xlabel("Time of Day")
            ax.set_ylabel(st.session_state.target.capitalize())
            ax.set_title(f"Time vs {st.session_state.target.capitalize()}")
            st.pyplot(fig)
        else:
            st.info("Scatter plot requires a 'time' feature to be selected.")
            
        st.markdown("---")
        st.subheader("Live Prediction")
        
        st.write("Input current traffic conditions to predict the delay:")
        inputs = {}
        cols = st.columns(len(st.session_state.features))
        for i, feat in enumerate(st.session_state.features):
            is_int = pd.api.types.is_integer_dtype(df_clean[feat])
            min_val = int(df_clean[feat].min()) if is_int else float(df_clean[feat].min())
            max_val = int(df_clean[feat].max()) if is_int else float(df_clean[feat].max())
            def_val = int(df_clean[feat].median()) if is_int else float(df_clean[feat].mean())
            step_val = 1 if is_int else None
            inputs[feat] = cols[i].number_input(f"{feat}", min_value=min_val, max_value=max_val, value=def_val, step=step_val)
            
        if st.button("Predict Delay"):
            input_df = pd.DataFrame([inputs])
            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.model.predict(input_scaled)[0]
            
            if st.session_state.model_type == "classification":
                prediction_formatted = f"{int(prediction)}"
            
            if "lanes_closed" in inputs:
                st.session_state.active_lanes_closed = int(inputs["lanes_closed"])
                
            st.metric(f"Predicted {st.session_state.target.capitalize()}", prediction_formatted)
            
        st.markdown("---")
        st.subheader("Lane Closure Optimization Engine")
        st.markdown("This algorithm searches for the optimal lane closure configuration to **minimize** the delay.")
        
        opt_fixed_feat = st.selectbox("Select a feature to lock (e.g., Traffic Density)", st.session_state.features)
        is_int_opt = pd.api.types.is_integer_dtype(df_clean[opt_fixed_feat])
        def_fixed_val = int(df_clean[opt_fixed_feat].median()) if is_int_opt else float(df_clean[opt_fixed_feat].mean())
        step_fixed_val = 1 if is_int_opt else None
        fixed_val = st.number_input(f"Current {opt_fixed_feat} value:", value=def_fixed_val, step=step_fixed_val)
        
        if st.button("Run Optimizer", type="primary"):
            best_delay = float('inf')
            best_config = None
            
            # Simple grid search over min/max bounds of the other features
            import itertools
            search_space = []
            feat_names = []
            
            for feat in st.session_state.features:
                if feat == opt_fixed_feat:
                    search_space.append([fixed_val])
                else:
                    unique_vals = df_clean[feat].unique()
                    if len(unique_vals) < 10:
                        search_space.append(sorted(unique_vals))
                    else:
                        # 5 steps between min and max
                        if pd.api.types.is_integer_dtype(df_clean[feat]):
                            vals = np.linspace(df_clean[feat].min(), df_clean[feat].max(), 5)
                            search_space.append(np.unique(np.round(vals).astype(int)))
                        else:
                            search_space.append(np.linspace(df_clean[feat].min(), df_clean[feat].max(), 5))
                feat_names.append(feat)
                
            combinations = list(itertools.product(*search_space))
            
            for combo in combinations:
                input_df = pd.DataFrame([combo], columns=feat_names)
                input_scaled = st.session_state.scaler.transform(input_df)
                pred = st.session_state.model.predict(input_scaled)[0]
                
                if pred < best_delay:
                    best_delay = pred
                    best_config = combo
            
            st.success("Optimization Complete!")
            
            if st.session_state.model_type == "classification":
                st.write(f"**Optimal {st.session_state.target.capitalize()} Class Predicted:** {int(best_delay)}")
            else:
                st.write(f"**Lowest Predicted {st.session_state.target.capitalize()}:** {best_delay:.2f}")
                
            st.write("**Optimal Configuration:**")
            for name, val in zip(feat_names, best_config):
                if pd.api.types.is_integer_dtype(df_clean[name]):
                    st.write(f"- {name}: {int(val)}")
                else:
                    st.write(f"- {name}: {val:.2f}")
                    
            if "lanes_closed" in feat_names:
                st.session_state.active_lanes_closed = int(best_config[feat_names.index("lanes_closed")])


# ---------------- PAGE 4: MAP ----------------
if selected == "Live Route Map":
    st.header("🗺️ Intelligent Real-World Routing Map")
    st.markdown("Enter two cities to visualize real road networks. The AI will calculate the optimal route based on your predicted lane closures!")
    
    lanes_closed = st.session_state.active_lanes_closed
    st.info(f"**Current Configuration:** {lanes_closed} Lanes Closed. (This map updates automatically when you run AI Predictions or the Optimizer!)")
    
    # Extract valid dataset cities
    valid_cities = set()
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data
        for col in df.select_dtypes(include=['object', 'string']).columns:
            valid_cities.update(df[col].dropna().unique())
    valid_cities_list = sorted(list(valid_cities)) if valid_cities else ["Kyrenia, Cyprus", "Nicosia, Cyprus"]
            
    col_map1, col_map2 = st.columns(2)
    
    start_idx = 0
    end_idx = 1 if len(valid_cities_list) > 1 else 0
    
    start_city = col_map1.selectbox("Origin City/Location:", options=valid_cities_list, index=start_idx)
    end_city = col_map2.selectbox("Destination City/Location:", options=valid_cities_list, index=end_idx)
    
    with st.spinner("Connecting to GPS Satellites..."):
        try:
            geolocator = Nominatim(user_agent="kyrenia_traffic_sim_ai")
            start_loc = geolocator.geocode(start_city, timeout=10)
            end_loc = geolocator.geocode(end_city, timeout=10)
            if not start_loc or not end_loc:
                st.error("Could not find one of the locations. Please check your spelling and try adding ', Cyprus'.")
            else:
                start_coords = (start_loc.latitude, start_loc.longitude)
                end_coords = (end_loc.latitude, end_loc.longitude)
                
                # Center map on the middle point
                mid_lat = (start_coords[0] + end_coords[0]) / 2
                mid_lon = (start_coords[1] + end_coords[1]) / 2
                m = folium.Map(location=[mid_lat, mid_lon], zoom_start=11)
                
                # Add Markers
                folium.Marker(start_coords, popup="Origin", icon=folium.Icon(color='green', icon='play')).add_to(m)
                folium.Marker(end_coords, popup="Destination", icon=folium.Icon(color='red', icon='stop')).add_to(m)
                
                # Fetch OSRM Route
                # Format: lon,lat;lon,lat
                osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=polyline&alternatives=true"
                res = requests.get(osrm_url)
                data = res.json()
                
                if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
                    routes = data["routes"]
                    
                    # Determine the index of the "Optimized" route
                    # If lanes are closed on the primary route, the bypass (index 1) is optimal
                    optimal_index = 0
                    if lanes_closed > 0 and len(routes) > 1:
                        optimal_index = 1
                        
                    # Loop through ALL possible routes returned by the GPS server
                    for i, route in enumerate(routes):
                        geometry = route["geometry"]
                        coords = polyline.decode(geometry)
                        
                        if i == optimal_index:
                            color = "green"
                            tooltip = "Optimized Route (Best Time)"
                            weight = 6
                            opacity = 0.9
                            dash = None
                        else:
                            color = "lightcoral" # Light Red
                            tooltip = "Sub-optimal Route (Congested or Slower)"
                            weight = 4
                            opacity = 0.6
                            dash = '10'
                            
                        folium.PolyLine(
                            locations=coords,
                            color=color,
                            weight=weight,
                            opacity=opacity,
                            dash_array=dash,
                            tooltip=tooltip
                        ).add_to(m)
                        
                    # Floating Map Legend
                    legend_html = '''
                     <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 250px; height: 100px; 
                     border: 1px solid rgba(255,255,255,0.2); z-index:9999; font-size:14px;
                     background-color:#1e293b; color: #f8fafc; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                     <b>🚦 Live GPS Route Legend</b><br>
                     <i class="fa fa-minus" style="color:green; font-weight:bold;"></i> Optimized Route <br>
                     <i class="fa fa-minus" style="color:lightcoral; font-weight:bold;"></i> Sub-optimal Route <br>
                     </div>
                     '''
                    m.get_root().html.add_child(folium.Element(legend_html))
                    
                    # Render map
                    st_folium(m, width=1200, height=550)
                else:
                    st.error("Could not calculate a real road route between these locations. The server may be unreachable.")
        except Exception as e:
            st.error(f"Error connecting to GPS API: {e}")
