import sys
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pytz
from timezonefinder import TimezoneFinder

# --- ☀️ CONFIGURATION ☀️ ---
HISTORICAL_SYSTEM_CAPACITY_KW = 5.0 
SOLAR_DATA_CSV = "sample_inverter_data_large.csv"
HISTORICAL_WEATHER_CSV = "weather_data_corrected_2024.csv"
MODEL_FILE_NAME = "solar_model.json"
# --- NEW: Degradation constant ---
ANNUAL_DEGRADATION_RATE = 0.005 # 0.5% per year

# ==============================================================================
# === 🤖 MODEL TRAINING & 🌍 FLASK APP LOGIC (No changes in this part) 🤖 ======
# ==============================================================================
def create_normalized_model():
    # ... (The training logic remains exactly the same as before) ...
    print("--- ⚙️ Starting Model Training Process ⚙️ ---")
    def load_and_clean_solar_data(filepath):
        print(f"Loading and cleaning solar data from '{filepath}'...")
        df = pd.read_csv(filepath); df['timestamp'] = pd.to_datetime(df['timestamp']); df = df.set_index('timestamp')
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df_complete = df.reindex(full_date_range); df_complete['kwh'].fillna(0, inplace=True); return df_complete
    def load_historical_weather_data(filepath):
        print(f"Loading historical weather data from '{filepath}'...")
        df = pd.read_csv(filepath); df['timestamp'] = pd.to_datetime(df['datetime']); df = df.set_index('timestamp')
        df = df[['temp', 'humidity', 'cloudcover', 'solarradiation']]; df = df.rename(columns={'cloudcover': 'cloud_cover'}); return df
    def prepare_data(solar_df, weather_df):
        print("Combining and normalizing data..."); df_combined = solar_df.join(weather_df, how='inner'); df_combined.dropna(inplace=True)
        print(f"Normalizing kWh data using historical system capacity of {HISTORICAL_SYSTEM_CAPACITY_KW} kW...")
        df_combined['kwh_per_kw'] = df_combined['kwh'] / HISTORICAL_SYSTEM_CAPACITY_KW
        df_combined['hour'] = df_combined.index.hour; df_combined['day_of_year'] = df_combined.index.dayofyear; df_combined['month'] = df_combined.index.month
        features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
        target = 'kwh_per_kw'; X = df_combined[features]; y = df_combined[target]; print("✅ Data preparation complete."); return X, y
    def train_and_save_model(X, y):
        print("Training the NORMALIZED prediction model..."); model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, n_jobs=-1)
        model.fit(X, y, eval_set=[(X, y)], verbose=False); model.save_model(MODEL_FILE_NAME); print(f"✅ --- 🎉 Normalized model trained and saved as '{MODEL_FILE_NAME}'! 🎉 ---")
    solar_df = load_and_clean_solar_data(SOLAR_DATA_CSV); weather_df = load_historical_weather_data(HISTORICAL_WEATHER_CSV)
    if solar_df is not None and weather_df is not None:
        X, y = prepare_data(solar_df, weather_df)
        if X is not None: train_and_save_model(X, y)

app = Flask(__name__)
tf = TimezoneFinder()
try:
    model = xgb.XGBRegressor(); model.load_model(MODEL_FILE_NAME)
    print(f"✅ Pre-trained normalized model '{MODEL_FILE_NAME}' loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load model file '{MODEL_FILE_NAME}'."); print(f"❌ --- Please run 'python main_app.py train' first! ---"); model = None

def map_weather_code(code):
    if code == 0: return "Clear sky ☀️"
    if code == 1: return "Mainly clear 🌤️"
    if code == 2: return "Partly cloudy ⛅️"
    if code == 3: return "Overcast ☁️"
    if code in [45, 48]: return "Fog 🌫️"
    if code in [51, 53, 55]: return "Drizzle 🌦️"
    if code in [61, 63, 65]: return "Rain 🌧️"
    if code in [80, 81, 82]: return "Rain showers 🌧️"
    if code == 95: return "Thunderstorm ⛈️"
    return "Unknown"

def get_theme_from_weather_code(code):
    if code in [0, 1]: return "sunny"
    if code in [2, 3]: return "cloudy"
    if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "rainy"
    if code in [95, 96, 99]: return "stormy"
    if code in [45, 48]: return "foggy"
    return "default" # Default dark theme

def get_location_details(city_name):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"; response = requests.get(geo_url); response.raise_for_status(); data = response.json()
    if not data.get('results'): raise ValueError(f"Could not find coordinates for '{city_name}'.")
    location = data['results'][0]; return location['latitude'], location['longitude'], location.get('timezone', 'UTC')

def get_weather_forecast(lat, lon, timezone):
    hourly_params = "temperature_2m,relativehumidity_2m,apparent_temperature,cloudcover,shortwave_radiation,weathercode,windspeed_10m,visibility"; daily_params = "sunrise,sunset"
    forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={hourly_params}&daily={daily_params}&timezone={timezone}"; response = requests.get(forecast_url); response.raise_for_status(); return response.json()

def get_historical_weather(lat, lon, timezone, start_date, end_date):
    hourly_params = "temperature_2m,relativehumidity_2m,cloudcover,shortwave_radiation"
    historical_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={hourly_params}&timezone={timezone}"
    print(f"Fetching historical data from: {historical_url}")
    response = requests.get(historical_url)
    response.raise_for_status()
    return response.json()

# --- NEW: Helper function to get effective capacity ---
def get_effective_capacity(data):
    """
    Calculates the effective capacity based on user settings.
    """
    try:
        base_capacity = float(data['capacity'])
        # Get inverter efficiency, convert from 97% to 0.97
        inverter_efficiency = float(data.get('inverterEfficiency', 97)) / 100.0
        # Get system age
        system_age = int(data.get('systemAge', 0))
        
        # Calculate degradation factor
        degradation_factor = (1 - ANNUAL_DEGRADATION_RATE) ** system_age
        
        effective_capacity = base_capacity * inverter_efficiency * degradation_factor
        
        print(f"Base Capacity: {base_capacity}kW")
        print(f"Inverter Eff: {inverter_efficiency*100}%, Age: {system_age}yrs, Degrad. Factor: {degradation_factor:.4f}")
        print(f"Effective Capacity: {effective_capacity:.2f}kW")
        
        return effective_capacity
    except Exception as e:
        print(f"Error calculating effective capacity: {e}. Falling back to base capacity.")
        return float(data.get('capacity', 1.0))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    try:
        data = request.get_json()
        
        # --- UPDATED: Use effective capacity ---
        capacity = float(data['capacity']) # Still need base capacity for one metric
        effective_capacity = get_effective_capacity(data)

        if 'city' in data and data['city']:
            city_name_for_display = data['city'].title(); lat, lon, timezone = get_location_details(data['city'])
        elif 'latitude' in data and 'longitude' in data:
            lat = float(data['latitude']); lon = float(data['longitude']); timezone = tf.timezone_at(lng=lon, lat=lat)
            if not timezone: raise ValueError("Could not determine timezone for coordinates.")
            city_name_for_display = f"Coords ({lat:.2f}, {lon:.2f})"
        else: raise ValueError("Missing city or coordinate data.")
        
        forecast_response = get_weather_forecast(lat, lon, timezone)
        
        hourly_data = forecast_response['hourly']; daily_data = forecast_response['daily']
        df_forecast = pd.DataFrame(hourly_data); df_forecast['timestamp'] = pd.to_datetime(df_forecast['time']); df_forecast = df_forecast.set_index('timestamp')
        df_forecast = df_forecast.rename(columns={'shortwave_radiation': 'solarradiation', 'temperature_2m': 'temp','cloudcover': 'cloud_cover', 'relativehumidity_2m': 'humidity', 'apparent_temperature': 'feels_like', 'windspeed_10m': 'windspeed', 'weathercode': 'weather_code'})

        df_forecast['hour'] = df_forecast.index.hour; df_forecast['day_of_year'] = df_forecast.index.dayofyear; df_forecast['month'] = df_forecast.index.month
        
        required_features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
        normalized_predictions = model.predict(df_forecast[required_features])
        
        # --- UPDATED: Use effective capacity ---
        df_forecast['predicted_kwh'] = normalized_predictions * effective_capacity
        df_forecast.loc[df_forecast['predicted_kwh'] < 0, 'predicted_kwh'] = 0

        now_in_timezone = datetime.now(pytz.timezone(timezone))
        current_hour_naive = now_in_timezone.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        
        idx = df_forecast.index.get_indexer([current_hour_naive], method='nearest')[0]
        current_weather_data = df_forecast.iloc[idx]
        
        todays_predictions = df_forecast[df_forecast.index.date == now_in_timezone.date()]
        
        total_prediction = todays_predictions['predicted_kwh'].sum(); peak_kwh = todays_predictions['predicted_kwh'].max()
        peak_hour_time = todays_predictions['predicted_kwh'].idxmax(); productive_hours = todays_predictions[todays_predictions['predicted_kwh'] > 0]
        avg_kwh = productive_hours['predicted_kwh'].mean() if not productive_hours.empty else 0
        
        today_daily_index = pd.to_datetime(daily_data['time']).get_loc(now_in_timezone.strftime('%Y-%m-%d'))
        sunrise_iso = daily_data['sunrise'][today_daily_index]; sunset_iso = daily_data['sunset'][today_daily_index]

        theme_name = get_theme_from_weather_code(current_weather_data['weather_code'])

        output = {
            "theme": theme_name,
            "city": city_name_for_display, "total": f"{total_prediction:.2f}",
            "labels": [t.strftime('%I %p').lstrip('0') for t in todays_predictions.index],
            "data": [round(p, 2) for p in todays_predictions['predicted_kwh']],
            "current_weather": {"temperature": f"{current_weather_data['temp']:.1f}", "feels_like": f"{current_weather_data['feels_like']:.1f}", "humidity": int(current_weather_data['humidity']), "windspeed": f"{current_weather_data['windspeed']:.1f}", "description": map_weather_code(current_weather_data['weather_code']), "visibility": f"{current_weather_data['visibility'] / 1000:.1f}"},
            "daily_info": {"sunrise": datetime.fromisoformat(sunrise_iso).strftime('%I:%M %p').lstrip('0'), "sunset": datetime.fromisoformat(sunset_iso).strftime('%I:%M %p').lstrip('0'), "avg_temp": f"{todays_predictions['temp'].mean():.1f}", "avg_cloud": f"{int(todays_predictions['cloud_cover'].mean())}", "avg_radiation": f"{int(productive_hours['solarradiation'].mean() if not productive_hours.empty else 0)}", "avg_humidity": f"{int(todays_predictions['humidity'].mean())}"},
            # --- UPDATED: Efficiency uses base capacity for a true "panel efficiency" metric ---
            "stats": {"peak_hour": peak_hour_time.strftime('%I %p').lstrip('0'), "peak_kwh": f"{peak_kwh:.2f}", "avg_kwh": f"{avg_kwh:.2f}", "efficiency": f"{min(100, (peak_kwh / capacity) * 100):.0f}"}
        }
        return jsonify(output)
    except Exception as e:
        print(f"Error in /predict: {e}"); return jsonify({"error": str(e)}), 400

@app.route('/long_term_predict', methods=['POST'])
def long_term_predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    try:
        data = request.get_json()
        
        # --- UPDATED: Use effective capacity ---
        effective_capacity = get_effective_capacity(data)
        
        if 'city' in data and data['city']:
            lat, lon, timezone_str = get_location_details(data['city'])
        elif 'latitude' in data and 'longitude' in data:
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            timezone_str = tf.timezone_at(lng=lon, lat=lat)
            if not timezone_str: timezone_str = "UTC" # Fallback
        else:
            raise ValueError("Missing city or coordinate data.")
        
        tz = pytz.timezone(timezone_str)
        
        end_date_dt = datetime.now(tz).date() - timedelta(days=1)
        start_date_dt = end_date_dt - timedelta(days=364)
        start_date_str = start_date_dt.strftime('%Y-%m-%d')
        end_date_str = end_date_dt.strftime('%Y-%m-%d')
        
        historical_response = get_historical_weather(lat, lon, timezone_str, start_date_str, end_date_str)
        
        hourly_data = historical_response['hourly']
        df_hist = pd.DataFrame(hourly_data)
        df_hist['timestamp'] = pd.to_datetime(df_hist['time'])
        df_hist = df_hist.set_index('timestamp')
        
        df_hist.ffill(inplace=True) 
        df_hist.fillna(0, inplace=True) 

        df_hist = df_hist.rename(columns={
            'shortwave_radiation': 'solarradiation', 
            'temperature_2m': 'temp',
            'cloudcover': 'cloud_cover', 
            'relativehumidity_2m': 'humidity'
        })
        
        df_hist['hour'] = df_hist.index.hour
        df_hist['day_of_year'] = df_hist.index.dayofyear
        df_hist['month'] = df_hist.index.month
        
        required_features = ['solarradiation', 'temp', 'cloud_cover', 'humidity', 'hour', 'day_of_year', 'month']
        
        normalized_predictions = model.predict(df_hist[required_features])
        
        # --- UPDATED: Use effective capacity ---
        df_hist['predicted_kwh'] = normalized_predictions * effective_capacity
        df_hist.loc[df_hist['predicted_kwh'] < 0, 'predicted_kwh'] = 0

        kwh_series = df_hist['predicted_kwh']
        
        weekly_totals = kwh_series.resample('W-SUN').sum()
        monthly_totals = kwh_series.resample('ME').sum()
        
        total_yearly_kwh = kwh_series.sum()
        avg_daily_kwh = total_yearly_kwh / 365
        best_month_date = monthly_totals.idxmax()
        worst_month_date = monthly_totals.idxmin()

        output = {
            "info": f"Long-term estimate based on data from {start_date_str} to {end_date_str}",
            
            "yearly_totals": {
                "total_kwh": float(round(total_yearly_kwh, 1)),
                "avg_daily_kwh": float(round(avg_daily_kwh, 1)),
                "best_month": best_month_date.strftime('%B'),
                "best_month_kwh": float(round(monthly_totals.max(), 1)),
                "worst_month": worst_month_date.strftime('%B'),
                "worst_month_kwh": float(round(monthly_totals.min(), 1))
            },
            
            "weekly_prediction": {
                "labels": [d.strftime('%Y-%m-%d') for d in weekly_totals.index],
                "data": [float(round(p, 1)) for p in weekly_totals.values]
            },
            "monthly_prediction": {
                "labels": [d.strftime('%B %Y') for d in monthly_totals.index],
                "data": [float(round(p, 1)) for p in monthly_totals.values]
            }
        }
        
        return jsonify(output)

    except Exception as e:
        print(f"Error in /long_term_predict: {e}")
        return jsonify({"error": str(e)}), 400

# ==============================================================================
# === 🚀 SCRIPT RUNNER (No changes needed here) 🚀 ============================
# ==============================================================================
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train': create_normalized_model()
        elif sys.argv[1] == 'run': 
            app.run(debug=True)
        else: print("Invalid command. Use 'train' or 'run'.")
    else: print("Please specify a command: 'train' to create the model, or 'run' to start the web server.")

