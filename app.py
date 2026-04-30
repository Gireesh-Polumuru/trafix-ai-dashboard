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
import time
from streamlit_option_menu import option_menu
import plotly.express as px
import math
import requests
import polyline
import streamlit.components.v1 as components
import pydeck as pdk

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
if 'model_columns' not in st.session_state:
    st.session_state.model_columns = []
if 'active_incident_lane' not in st.session_state:
    st.session_state.active_incident_lane = 0
if 'active_diversion_lane' not in st.session_state:
    st.session_state.active_diversion_lane = 1
if 'temp_lane_active' not in st.session_state:
    st.session_state.temp_lane_active = False

# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h2 style="color: #fb923c !important; font-weight: 800; font-size: 2rem; margin: 0;">Trafix AI 🚦</h2>
        <p style="color: #94a3b8 !important; font-size: 0.9rem; margin-top: 0;">Intelligent Route Optimizer</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Data Processing", "ML Training", "Dashboard", "Lane Simulation Map"],
        icons=["cloud-upload", "robot", "speedometer2", "layout-three-columns"],
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
    st.markdown("**Version:** 3.0.0 Pro")
    st.markdown("**Status:** System Online 🟢")


st.markdown("""
<div class="hero-card">
    <h1>🚦 Trafix AI: Incident Management System</h1>
    <p>Predicting optimal diversion routes during real-world lane closures (Accidents, Construction, etc.)</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PAGE 1: DATA ----------------
if selected == "Data Processing":
    st.header("1. Upload & Clean Traffic Incident Dataset")
    
    col_upload, col_gen = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload your traffic dataset (CSV)", type=["csv"])
    with col_gen:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Demo Incident Dataset"):
            n = 500
            locations = ["Vijayawada to Mangalagiri (NH16)", "Vijayawada to Mangalagiri (Old Route)"]
            origins = np.random.choice(locations, n)
            
            times = np.random.randint(0, 24, n)
            total_lanes = np.random.choice([4, 6], n)
            density = np.random.uniform(10, 100, n)
            
            incident_types = ["Accident", "Construction", "Maintenance", "Debris Hazard"]
            incidents = np.random.choice(incident_types, n)
            
            incident_lanes = [np.random.randint(0, tl) for tl in total_lanes]
            
            # The optimal diversion lane is the safest open lane adjacent to the incident
            optimal_diversion = []
            for i in range(n):
                inc_lane = incident_lanes[i]
                tl = total_lanes[i]
                if inc_lane == 0:
                    optimal_diversion.append(1)
                elif inc_lane == tl - 1:
                    optimal_diversion.append(tl - 2)
                else:
                    # Randomly pick left or right
                    optimal_diversion.append(inc_lane + np.random.choice([-1, 1]))
                
            demo_df = pd.DataFrame({
                'location': origins,
                'time': times,
                'total_lanes': total_lanes,
                'traffic_density': np.round(density, 2),
                'incident_type': incidents,
                'incident_lane': incident_lanes,
                'optimal_diversion_lane': optimal_diversion
            })
            
            # Intentionally add some nulls and duplicates to prove data cleaning works
            demo_df.loc[10:15, "traffic_density"] = np.nan
            demo_df = pd.concat([demo_df, demo_df.iloc[:5]], ignore_index=True)
            
            st.session_state.raw_data = demo_df
            st.success("Demo Dataset Generated! Ready for analysis.")

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
    st.header("2. AI Diversion Optimization Model")
    st.markdown("Train a Classification AI to predict the optimal lane to route traffic into during an incident.")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please upload and clean a dataset in the Data Tab first.")
    else:
        df_clean = st.session_state.cleaned_data
        # We allow categorical columns now too
        all_columns = df_clean.columns.tolist()
        
        col_f1, col_f2 = st.columns(2)
        # Default to optimal_diversion_lane if available
        default_idx = all_columns.index("optimal_diversion_lane") if "optimal_diversion_lane" in all_columns else len(all_columns)-1
        target_col = col_f1.selectbox("Select Target Variable (e.g., Optimal Diversion Lane)", all_columns, index=default_idx)
        
        feature_cols = [c for c in all_columns if c != target_col and c not in ['location']]
        selected_features = col_f2.multiselect("Select Input Features (e.g., Incident Type, Incident Lane, Time)", feature_cols, default=feature_cols)
        
        if st.button("Train Diversion AI Optimizer", type="primary"):
            if len(selected_features) == 0:
                st.error("Please select at least one feature.")
            else:
                st.session_state.features = selected_features
                st.session_state.target = target_col
                
                X = df_clean[selected_features]
                y = df_clean[target_col]
                
                # Convert categorical features to dummy variables
                X_encoded = pd.get_dummies(X, drop_first=False)
                st.session_state.model_columns = X_encoded.columns.tolist()
                
                # Ensure y is categorical/integer for classification
                y = y.astype(int) if pd.api.types.is_numeric_dtype(y) else y.astype(str)
                
                # Standard Scaling for preprocessing
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
                st.session_state.scaler = scaler
                
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                st.session_state.model_type = "classification"
                    
                model.fit(X_train, y_train)
                st.session_state.model = model
                
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
                st.success(f"Route Optimization AI Trained Successfully! Accuracy: {score*100:.2f}%")


# ---------------- PAGE 3: DASHBOARD ----------------
if selected == "Dashboard":
    st.header("3. Traffic Incident & Route Optimizer")
    st.markdown("Use the AI to determine where to safely divert traffic when a lane closes due to an accident or construction.")
    
    if st.session_state.model is None:
        st.warning("Please train the model in the ML Training Tab first.")
    else:
        df_clean = st.session_state.cleaned_data
        
        st.subheader("Current Traffic Incident Profile")
        inputs = {}
        cols = st.columns(min(len(st.session_state.features), 4))
        for i, feat in enumerate(st.session_state.features):
            col_idx = i % 4
            if pd.api.types.is_numeric_dtype(df_clean[feat]):
                is_int = pd.api.types.is_integer_dtype(df_clean[feat])
                min_val = int(df_clean[feat].min()) if is_int else float(df_clean[feat].min())
                max_val = int(df_clean[feat].max()) if is_int else float(df_clean[feat].max())
                def_val = int(df_clean[feat].median()) if is_int else float(df_clean[feat].mean())
                step_val = 1 if is_int else None
                inputs[feat] = cols[col_idx].number_input(f"{feat}", min_value=min_val, max_value=max_val, value=def_val, step=step_val)
            else:
                unique_vals = df_clean[feat].unique().tolist()
                inputs[feat] = cols[col_idx].selectbox(f"{feat}", unique_vals)
                
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("### 🤖 AI Recommendation")
            st.info("The AI analyzes the incident type, time, and density to pick the safest diversion lane.")
            if st.button("Predict Safest Diversion Lane", type="primary"):
                input_df = pd.DataFrame([inputs])
                input_encoded = pd.get_dummies(input_df)
                
                # Ensure all columns match training data
                for c in st.session_state.model_columns:
                    if c not in input_encoded.columns:
                        input_encoded[c] = 0
                input_encoded = input_encoded[st.session_state.model_columns]
                
                input_scaled = st.session_state.scaler.transform(input_encoded)
                prediction = int(st.session_state.model.predict(input_scaled)[0])
                
                st.success(f"✅ AI Decision: **Divert traffic into Lane {prediction}**")
                # Pre-fill manual override
                st.session_state.active_diversion_lane = prediction
                if "incident_lane" in inputs:
                    st.session_state.active_incident_lane = int(inputs["incident_lane"])

        with col_res2:
            st.markdown("### 🎛️ Manual Route Override")
            st.warning("Directly configure the incident location and the target diversion lane for the 3D map.")
            total_lanes_avail = 4
            if "total_lanes" in inputs:
                total_lanes_avail = int(inputs["total_lanes"])
            elif "total_lanes" in df_clean.columns:
                total_lanes_avail = int(df_clean["total_lanes"].max())
                
            lane_options = [i for i in range(total_lanes_avail)]
            
            c1, c2 = st.columns(2)
            
            # Get current selected from state or default
            current_inc = st.session_state.get('active_incident_lane', 0)
            if current_inc not in lane_options: current_inc = lane_options[0]
            
            current_div = st.session_state.get('active_diversion_lane', 1)
            if current_div not in lane_options: current_div = lane_options[-1]
                
            selected_inc = c1.selectbox("Incident Location (Blocked Lane):", lane_options, index=lane_options.index(current_inc))
            selected_div = c2.selectbox("Diversion Target (Open Lane):", lane_options, index=lane_options.index(current_div))
            
            # Temporary Lane Option
            is_construction = inputs.get("incident_type", "") == "Construction"
            deploy_temp = st.checkbox("🚧 Deploy Temporary Bypass Lane (Construction Only)", 
                                      value=st.session_state.get('temp_lane_active', False), 
                                      disabled=not is_construction)
            
            if st.button("Apply to 3D Map Simulation"):
                st.session_state.active_incident_lane = int(selected_inc)
                st.session_state.active_diversion_lane = int(selected_div)
                st.session_state.temp_lane_active = deploy_temp if is_construction else False
                st.success(f"✅ Simulation Ready! Map will show Incident in Lane {selected_inc} and Diversion to Lane {selected_div}.")


# ---------------- PAGE 4: MAP ----------------
if selected == "Lane Simulation Map":
    st.header("🗺️ 3D Map Visualization: 500m Warning Zone Simulation")
    st.markdown("Visualize real-world incident closures, advanced 500-meter warning zones, and intelligent vehicle diversions in full 3D.")
    
    incident_lane = st.session_state.get('active_incident_lane', 0)
    diversion_lane = st.session_state.get('active_diversion_lane', 1)
    
    # Try to extract total_lanes from session state if available, default to 4
    total_lanes = 4
    if st.session_state.cleaned_data is not None and "total_lanes" in st.session_state.cleaned_data.columns:
        total_lanes = int(st.session_state.cleaned_data["total_lanes"].max())
        
    total_lanes = max(2, min(total_lanes, 8))
    if incident_lane >= total_lanes: incident_lane = 0
    if diversion_lane >= total_lanes: diversion_lane = 1
    
    temp_lane_active = st.session_state.get('temp_lane_active', False)
    
    st.error(f"**⚠️ ACTIVE INCIDENT REPORTED IN LANE {incident_lane}**")
    if temp_lane_active:
        st.success(f"**🚧 TEMPORARY BYPASS:** Routing traffic to newly constructed Shoulder Lane")
        diversion_lane = total_lanes # Set diversion to the extra lane index
    else:
        st.success(f"**🚦 ACTIVE DIVERSION:** Routing traffic to **Lane {diversion_lane}**")
    
    map_type = st.radio("Select Map Engine:", [
        "🛰️ 3D Satellite View (Free)", 
        "🗺️ Google Maps (API Key Required)", 
        "🌑 Mapbox (API Key Required)"
    ], horizontal=True)
    
    api_key = ""
    if "API Key Required" in map_type:
        api_key = st.text_input(f"Enter API Key for {map_type.split(' ')[1]}:", type="password")
        if not api_key:
            st.warning("Please enter your API Key to load this map engine. Falling back to Free View.")
    
    st.markdown("### Interactive 3D PyDeck Simulation")
    
    col_map1, col_map2 = st.columns(2)
    sim_location = col_map1.selectbox("Select Road Segment for Simulation:", ["Vijayawada to Mangalagiri (NH16)"])
    
    if sim_location == "Vijayawada to Mangalagiri (NH16)":
        # Full stretch from Vijayawada Benz Circle to Mangalagiri
        start_pt = (16.4965, 80.6552)
        end_pt = (16.4314, 80.5671)
    else:
        start_pt = (16.4965, 80.6552)
        end_pt = (16.4314, 80.5671)
        
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
    with st.spinner("Fetching Real Road Geometry & Calculating 500m Warning Zones..."):
        try:
            osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_pt[1]},{start_pt[0]};{end_pt[1]},{end_pt[0]}?overview=full&geometries=polyline"
            res = requests.get(osrm_url)
            data = res.json()
            
            if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
                geometry = data["routes"][0]["geometry"]
                coords = polyline.decode(geometry)
                
                # Calculate distances to find the 500m warning zone
                cumulative_distances = [0.0]
                for i in range(1, len(coords)):
                    dist = haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
                    cumulative_distances.append(cumulative_distances[-1] + dist)
                    
                total_distance = cumulative_distances[-1]
                
                # Disruption (Accident) at 70% of the road
                disruption_distance = total_distance * 0.7
                disruption_idx = min(range(len(cumulative_distances)), key=lambda i: abs(cumulative_distances[i]-disruption_distance))
                
                # Diversion warning starts 500m before disruption
                diversion_distance = max(0, disruption_distance - 500)
                diversion_idx = min(range(len(cumulative_distances)), key=lambda i: abs(cumulative_distances[i]-diversion_distance))
                
                path_data = []
                lane_width = 0.0001
                
                for i in range(total_lanes):
                    offset = (i - total_lanes / 2 + 0.5) * lane_width
                    # OSRM gives (lat, lon), pydeck needs (lon, lat)
                    lane_coords = [[c[1] + offset, c[0] - offset] for c in coords]
                    
                    if i == incident_lane:
                        # Split incident lane into 3 segments: Open (Green) -> Warning (Orange) -> Blocked (Red)
                        path_data.append({"path": lane_coords[:diversion_idx+1], "color": [34, 197, 94], "width": 5}) # Green
                        path_data.append({"path": lane_coords[diversion_idx:disruption_idx+1], "color": [249, 115, 22], "width": 5}) # Orange (500m Zone)
                        path_data.append({"path": lane_coords[disruption_idx:], "color": [239, 68, 68], "width": 5}) # Red (Blocked)
                    else:
                        # Normal open lanes
                        path_data.append({"path": lane_coords, "color": [34, 197, 94], "width": 5}) # Green
                        
                # Add temporary bypass lane if active
                if temp_lane_active:
                    offset_temp = (total_lanes - total_lanes / 2 + 0.5) * lane_width
                    temp_coords = [[c[1] + offset_temp, c[0] - offset_temp] for c in coords]
                    path_data.append({"path": temp_coords, "color": [234, 179, 8], "width": 6}) # Distinct Yellow bypass
                        
                # Simulated Vehicle Trajectory (Yellow)
                car_coords = []
                if incident_lane < total_lanes:
                    offset_inc = (incident_lane - total_lanes / 2 + 0.5) * lane_width
                    # If temp_lane_active, diversion_lane is set to total_lanes (the extra lane)
                    target_div_idx = total_lanes if temp_lane_active else diversion_lane
                    offset_div = (target_div_idx - total_lanes / 2 + 0.5) * lane_width
                    
                    inc_path = [[c[1] + offset_inc, c[0] - offset_inc] for c in coords]
                    div_path = [[c[1] + offset_div, c[0] - offset_div] for c in coords]
                    
                    # Travel on incident lane until diversion point
                    car_coords.extend(inc_path[:diversion_idx])
                    
                    # Smooth transition to diversion lane
                    transition = []
                    steps = min(10, len(coords) - diversion_idx)
                    for step in range(steps):
                        idx = diversion_idx + step
                        if idx < len(coords):
                            ratio = step / float(steps)
                            lon = inc_path[idx][0] * (1-ratio) + div_path[idx][0] * ratio
                            lat = inc_path[idx][1] * (1-ratio) + div_path[idx][1] * ratio
                            transition.append([lon, lat])
                            
                    car_coords.extend(transition)
                    
                    # Continue on diversion lane
                    if diversion_idx + steps < len(coords):
                        car_coords.extend(div_path[diversion_idx + steps : disruption_idx + 20])
                    
                    path_data.append({"path": car_coords, "color": [255, 255, 255], "width": 8}) # Bright White/Yellow Trajectory
                
                # Markers (Scatterplot)
                marker_data = []
                if incident_lane < total_lanes:
                    offset_inc = (incident_lane - total_lanes / 2 + 0.5) * lane_width
                    inc_lon = coords[disruption_idx][1] + offset_inc
                    inc_lat = coords[disruption_idx][0] - offset_inc
                    
                    div_lon = coords[diversion_idx][1] + offset_inc
                    div_lat = coords[diversion_idx][0] - offset_inc
                    
                    # Incident marker
                    marker_data.append({"position": [inc_lon, inc_lat], "color": [255, 0, 0], "radius": 40})
                    # 500m Warning Start marker
                    marker_data.append({"position": [div_lon, div_lat], "color": [255, 165, 0], "radius": 25})
                
                df_path = pd.DataFrame(path_data)
                df_markers = pd.DataFrame(marker_data)
                
                center_lat = sum(c[0] for c in coords) / len(coords)
                center_lon = sum(c[1] for c in coords) / len(coords)
                
                layer_lines = pdk.Layer(
                    "PathLayer",
                    df_path,
                    get_path="path",
                    get_color="color",
                    width_scale=5,
                    width_min_pixels=3,
                    get_width="width",
                )
                
                layer_markers = pdk.Layer(
                    "ScatterplotLayer",
                    df_markers,
                    get_position="position",
                    get_fill_color="color",
                    get_radius="radius",
                    pickable=True
                )
                
                view_state = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=12,
                    pitch=50,
                )
                
                if "Google Maps" in map_type and api_key:
                    r = pdk.Deck(
                        map_provider="google_maps",
                        api_keys={"google_maps": api_key},
                        layers=[layer_lines, layer_markers], 
                        initial_view_state=view_state
                    )
                elif "Mapbox" in map_type and api_key:
                    r = pdk.Deck(
                        map_provider="mapbox",
                        map_style=pdk.map_styles.DARK,
                        api_keys={"mapbox": api_key},
                        layers=[layer_lines, layer_markers], 
                        initial_view_state=view_state
                    )
                else:
                    base_layer = pdk.Layer(
                        "TileLayer",
                        data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        min_zoom=0,
                        max_zoom=19,
                    )
                    r = pdk.Deck(layers=[base_layer, layer_lines, layer_markers], initial_view_state=view_state, map_style=None)
                
                st.pydeck_chart(r)
                
                # Custom Legend Overlay
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.8); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2);">
                    <h4 style="margin-top:0;">🚦 Simulation Legend</h4>
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <span style="background-color: #22c55e; width: 20px; height: 10px; display:inline-block; margin-right: 10px; border-radius: 2px;"></span> Open Lane
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <span style="background-color: #ef4444; width: 20px; height: 10px; display:inline-block; margin-right: 10px; border-radius: 2px;"></span> Blocked Lane (Incident Location)
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <span style="background-color: #f97316; width: 20px; height: 10px; display:inline-block; margin-right: 10px; border-radius: 2px;"></span> 500-Meter Warning Zone
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <span style="background-color: #ffffff; width: 20px; height: 5px; display:inline-block; margin-right: 10px; border-radius: 2px;"></span> Vehicle Trajectory
                    </div>
                    <div style="display:flex; align-items:center; margin-bottom:5px;">
                        <span style="background-color: #eab308; width: 20px; height: 10px; display:inline-block; margin-right: 10px; border-radius: 2px;"></span> Temporary Bypass Lane (If active)
                    </div>
                    <div style="display:flex; align-items:center;">
                        <span style="background-color: #ff0000; width: 14px; height: 14px; border-radius: 50%; display:inline-block; margin-right: 13px; margin-left: 3px;"></span> Incident Point
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error("Could not fetch real road geometry from OSRM.")
        except Exception as e:
            st.error(f"Error generating simulation: {e}")
