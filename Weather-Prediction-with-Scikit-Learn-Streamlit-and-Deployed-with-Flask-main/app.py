import requests
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plost
import reverse_geocoder as rg
import pickle
import datetime
import time

# API endpoint for the Flask app
API = "http://127.0.0.1:5000/"

# File paths
MODEL_PATH = './model/weather_model.pkl'
SCALER_PATH = './model/scaler.pkl'
IMG_SIDEBAR_PATH = "./assets/img.jpg"

# ----------| WEATHER REPORT |----------

class WeatherReport:
    def __init__(self):
        self.apiID = "a42694a8daa8291c88ff863a26c2da62"
        self.units = "imperial"
        self.apiWeather = "https://api.openweathermap.org/data/2.5/weather"

    def get_weather_report(self, location, is_city=True):
        if is_city:
            payload = {'q': location, 'APPID': self.apiID, 'units': self.units}
        else:
            payload = {'zip': location, 'APPID': self.apiID, 'units': self.units}
        
        requestWeather = requests.get(self.apiWeather, params=payload)
        requestStatus = requestWeather.status_code

        if requestStatus == 200:
            weatherData = json.loads(requestWeather.text)
            return weatherData
        else:
            st.error("The city or ZIP code you entered is not valid, or there was a connection error. Please try again.")
            return None

# ----------| WEATHER FORECAST |----------

class WeatherForecast:
    def __init__(self):
        self.apiID = "a42694a8daa8291c88ff863a26c2da62"
        self.units = "imperial"
        self.apiForecast = "https://api.openweathermap.org/data/2.5/forecast"

    def get_weather_forecast(self, location, is_city=True):
        if is_city:
            payload = {'q': location, 'APPID': self.apiID, 'units': self.units}
        else:
            payload = {'zip': location, 'APPID': self.apiID, 'units': self.units}
        
        requestForecast = requests.get(self.apiForecast, params=payload)
        requestStatus = requestForecast.status_code

        if requestStatus == 200:
            forecastData = json.loads(requestForecast.text)
            return forecastData
        else:
            st.error("The city or ZIP code you entered is not valid, or there was a connection error. Please try again.")
            return None

# ----------| MACHINE LEARNING MODEL |----------

def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

def get_clean_data():
    return pd.read_csv("./dataset/weather_dataset.csv")

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['weather'], axis=1)
    scaled_dict = {key: (value - X[key].min()) / (X[key].max() - X[key].min()) for key, value in input_dict.items()}
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Precipitation', 'Max Temperature', 'Min Temperature', 'Wind']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[input_data['precipitation'], input_data['temp_max'], input_data['temp_min'], input_data['wind']],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    return fig

#Receiving Prediction Results from the API
def add_predictions(input_data) :
    input_array = np.array(list(input_data.values())).reshape(1, -1).tolist()

    input_array_scaled = scaler.transform(input_array)
    pred_result = model.predict(input_array_scaled)

    pred_result = int(pred_result[0])
    prob_drizzle = round(model.predict_proba(input_array_scaled)[0][0], 2)
    prob_rain = round(model.predict_proba(input_array_scaled)[0][1], 2)
    prob_sun = round(model.predict_proba(input_array_scaled)[0][2], 2)
    prob_snow = round(model.predict_proba(input_array_scaled)[0][3], 2)
    prob_fog = round(model.predict_proba(input_array_scaled)[0][4], 2)

    #Run first the api.py file and the paste the URL in the API Variable if you want to deploy the Model with Flask and uncomment the next lines
    #data = {'array': input_array}

    #resp = requests.post(API, json=data)
    
    #pred_result = resp.json()["Results"]["result"]
    #prob_drizzle = resp.json()["Results"]["prob_drizzle"]
    #prob_rain = resp.json()["Results"]["prob_rain"]
    #prob_sun = resp.json()["Results"]["prob_sun"]
    #prob_snow = resp.json()["Results"]["prob_snow"]
    #prob_fog = resp.json()["Results"]["prob_fog"]

    st.markdown("### Weather Prediction ✅")
    st.write("<span class='diagnosis-label'>Machine Learning Model Result:</span>",  unsafe_allow_html=True)
    
    if pred_result == 0:
      st.write("<span class='diagnosis drizzle'>Drizzle</span>", unsafe_allow_html=True)
    elif pred_result == 1 :
      st.write("<span class='diagnosis rain'>Rain</span>", unsafe_allow_html=True)
    elif pred_result == 2:
      st.write("<span class='diagnosis sun'>Sun</span>", unsafe_allow_html=True)
    elif pred_result == 3 :
      st.write("<span class='diagnosis snow'>Snow</span>", unsafe_allow_html=True)
    elif pred_result == 4 :
      st.write("<span class='diagnosis fog'>Fog</span>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1 :
        st.metric("Probability:", f"{prob_drizzle}%", "Drizzle")
    
    with col2:
        st.metric("Probability:", f"{prob_rain}%", "Rain")
      
    with col3: 
        st.metric("Probability:", f"{prob_sun}%", "Sun")

    col4, col5 = st.columns([1, 1])
    with col4 :
        st.metric("Probability:", f"{prob_snow}%", "Snow")
    
    with col5:
        st.metric("Probability:", f"{prob_fog}%", "Fog")

    st.write("`This Artificial Intelligence can Assist for any Scientific about the Upcoming Weather, but Should Not be used as a Substitute for a Final Diagnosis and Prediction.`")
    

# ----------| STREAMLIT APP |----------

def add_sidebar():
    st.sidebar.header("DSI Weather Predictor `App ⛈️`")
    image = np.array(Image.open(IMG_SIDEBAR_PATH))
    st.sidebar.image(image)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.write("This Artificial Intelligence App can Predict the Future Weather Given Parameters.")

    # Create a tab menu for the user
    option = st.sidebar.selectbox(
        "Choose the type of information you want:",
        ["Weather Report", "Weather Forecast"]
    )
    
    if option == "Weather Report":
        st.title("Weather Predictor ⛅️")
        st.write("This App predicts using a KNeighborsClassifier Machine Learning Model whether a given parameters the Upcoming Weather is eather Drizzle, Sun, Snow, Fog or Rain. You can also Update the measurements by hand using sliders in the sidebar.")
        st.markdown("<hr/>", unsafe_allow_html=True)

        st.subheader("Weather Report")
        location = st.text_input("Enter city name or ZIP code:")
        is_city = st.radio("Is this a city?", ("Yes", "No")) == "Yes"
        
        if st.button("Get Weather Report"):
            if location:
                wr = WeatherReport()
                weatherData = wr.get_weather_report(location, is_city)
                if weatherData:
                    coordinates = (weatherData['coord']['lat'], weatherData['coord']['lon'])
                    geoLookup = rg.search(coordinates)
                    weatherCity = geoLookup[0]['name']
                    weatherState = geoLookup[0]['admin1']
                    weatherCounty = geoLookup[0]['admin2']
                    st.write(f"City: {weatherCity}")
                    st.write(f"State: {weatherState}")
                    st.write(f"County: {weatherCounty}")
                    st.write(f"Country: {weatherData['sys']['country']}")
                    st.write(f"Conditions: {weatherData['weather'][0]['description'].capitalize()}")
                    st.write(f"Temp: {round(weatherData['main']['temp'])} °F")
                    st.write(f"Humidity: {weatherData['main']['humidity']}%")
                    st.write(f"Pressure: {weatherData['main']['pressure']} hpa")
                    st.write(f"Wind: {weatherData['wind']['speed']} mph")
                    st.write(f"Sunrise: {time.ctime(weatherData['sys']['sunrise'])}")
                    st.write(f"Sunset: {time.ctime(weatherData['sys']['sunset'])}")
                    st.write(f"Latitude: {weatherData['coord']['lat']}")
                    st.write(f"Longitude: {weatherData['coord']['lon']}")
    
    elif option == "Weather Forecast":
        st.title("Weather Predictor ⛅️")
        st.write("This App predicts using a KNeighborsClassifier Machine Learning Model whether a given parameters the Upcoming Weather is eather Drizzle, Sun, Snow, Fog or Rain. You can also Update the measurements by hand using sliders in the sidebar.")
        st.markdown("<hr/>", unsafe_allow_html=True)

        st.subheader("Weather Forecast")
        location = st.text_input("Enter city name or ZIP code:")
        is_city = st.radio("Is this a city?", ("Yes", "No")) == "Yes"
        
        if st.button("Get Weather Forecast"):
            if location:
                wf = WeatherForecast()
                forecastData = wf.get_weather_forecast(location, is_city)
                if forecastData:
                    coordinates = (forecastData['city']['coord']['lat'], forecastData['city']['coord']['lon'])
                    geoLookup = rg.search(coordinates)
                    forecastCity = geoLookup[0]['name']
                    forecastState = geoLookup[0]['admin1']
                    forecastCounty = geoLookup[0]['admin2']
                    
                    st.write(f"City: {forecastCity}")
                    st.write(f"State: {forecastState}")
                    st.write(f"County: {forecastCounty}")
                    st.write(f"Country: {forecastData['city']['country']}")
                    st.write("Five-Day Forecast:")
                    
                    day1 = datetime.date.today() + datetime.timedelta(days=1)
                    day2 = datetime.date.today() + datetime.timedelta(days=2)
                    day3 = datetime.date.today() + datetime.timedelta(days=3)
                    day4 = datetime.date.today() + datetime.timedelta(days=4)
                    day5 = datetime.date.today() + datetime.timedelta(days=5)

                    for i in range(5):
                        index = i * 8 + 4
                        day = [day1, day2, day3, day4, day5][i]
                        temp = round(forecastData['list'][index]['main']['temp'])
                        wind = round(forecastData['list'][index]['wind']['speed'])
                        humidity = forecastData['list'][index]['main']['humidity']
                        description = forecastData['list'][index]['weather'][0]['description'].capitalize()
                        st.write(f"{day.strftime('%b %d')}: Temp: {temp} °F, Wind: {wind} mph, Humidity: {humidity}%, Outlook: {description}")

    

    st.sidebar.subheader('Select the Weather Parameters ✅:')
    data = get_clean_data()
    slider_labels = [
        ("Precipitation", "precipitation"),
        ("Max Temperature", "temp_max"),
        ("Min Temperature", "temp_min"),
        ("Wind", "wind"),
    ]
    input_dict = {key: st.sidebar.slider(label, min_value=float(0), 
                                         max_value=float(data[key].max()), 
                                         value=float(data[key].mean())) 
                  for label, key in slider_labels}
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.markdown('''
    🧑🏻‍💻 Created by [vineet choudhary](https://github.com/mendez-luisjose).
    ''')
    return input_dict

def main():
    st.set_page_config(
        page_title="Weather Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.container() :
        st.markdown("<hr/>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    df = pd.read_csv("./assets/weather_classes.csv")

    with col1:
        st.markdown('### Radar Chart of the Parameters 📊')
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

        st.markdown('### Bar Chart of the Weather Classes 📉')
        st.markdown("---", unsafe_allow_html=True)

        plost.bar_chart(
            data=df,
            bar='Weather',
            value='Number of that Class', 
            legend='bottom',
            use_container_width=True,
            color='Weather'
        )        
        

    with col2:
        st.markdown('### Donut Chart of the Weather Classes 📈')

        plost.donut_chart(
            data=df,
            theta="Number of that Class",
            color='Weather',
            legend='bottom', 
            use_container_width=True)
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        add_predictions(input_data)

if __name__ == "__main__" :
    main()

    print("App Running!")
