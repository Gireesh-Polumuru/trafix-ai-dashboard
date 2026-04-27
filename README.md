# 🚦 Trafix AI Optimizer

**Trafix AI Optimizer** is a complete, end-to-end Data Science and Machine Learning web application designed to predict traffic delays and optimize highway lane closures. Built with Python and Streamlit, it combines synthetic traffic data generation, Random Forest Machine Learning, and real-world OpenStreetMap (OSRM) GPS routing into a single, high-performance dashboard.

---

## 🌟 Key Features

1. **Intelligent Data Pipeline:**
   - Upload your own traffic CSV data or instantly generate a synthetic 10,000-row dataset for testing.
   - Built-in data cleaning that automatically handles missing values and standardizes features.

2. **Machine Learning Engine:**
   - Automatically trains a `RandomForestRegressor` (or Classifier) on your dataset.
   - Evaluates the model with real-time R² and Accuracy scores.
   - Identifies the most important factors causing traffic delays (e.g., traffic density, time of day, lane closures).

3. **Optimization Engine:**
   - Input current road conditions to predict exact traffic delays in real-time.
   - A brute-force Optimization Algorithm that iterates through all possible lane-closure configurations to find the absolute minimum delay time for commuters.

4. **Live Real-World GPS Routing Map:**
   - Powered by Folium and the OpenStreetMap (OSRM) API.
   - Enter any two valid cities from your dataset to instantly render real-world road geometries.
   - The AI automatically determines the "Optimal Route" (Green) and visualizes congested "Sub-optimal Routes" (Red) based on the current lane-closure configuration.

5. **Premium UI/UX:**
   - Modern, glassmorphism-style design with a sleek dark theme.
   - Animated sidebar navigation and premium Google Fonts integration.

---

## 💻 Tech Stack
- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Science:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Geospatial & Mapping:** `folium`, `streamlit-folium`, `geopy`, `requests`, `polyline`

---

## 🚀 How to Run Locally

If you want to run this application on your own computer, follow these steps:

1. **Install Python 3.9+**
2. **Install Dependencies:** Open your terminal and run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```
4. **Access the App:** Open your web browser and go to `http://localhost:8501`.

---

## 🌍 How to Deploy to Streamlit Cloud

To share this app with the world:
1. Create a free account on [GitHub](https://github.com/) and upload `app.py` and `requirements.txt` to a new Public Repository.
2. Create a free account on [Streamlit Community Cloud](https://share.streamlit.io/) and link your GitHub.
3. Click **New App**, select your repository, and set the main file path to `app.py`.
4. Click **Deploy**. Your app will be live in 2 minutes!

---
*Developed for Traffic Simulation & Machine Learning Analysis.*
