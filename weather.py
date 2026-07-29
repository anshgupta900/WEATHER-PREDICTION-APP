# ============================================================
# WEATHER PREDICTION PROJECT 
# ============================================================


# -------------------------------
# IMPORT LIBRARIES
# -------------------------------

import pandas as pd                 
import numpy as np                  
import matplotlib.pyplot as plt     
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score


# -------------------------------
# LOAD DATASET
# -------------------------------
print(" Loading dataset \n")

df = pd.read_csv("weather.csv")     

print("Dataset loaded!\n")
print(df.head(), "\n")              # show first 5 rows


# -------------------------------
# DATA CLEANING
# -------------------------------
print(" Checking missing values \n")

print(df.isnull().sum(), "\n")      # check missing values

df = df.dropna()                   # remove missing values

print("Missing values removed!\n")


# PREPARE DATA

# Convert RainTomorrow Yes/No 
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})

print(df[['RainTomorrow']].head(), "\n")


# ============================================================
# LINEAR REGRESSION (Temperature Prediction)
# ============================================================

print(" Linear Regression \n")

# Input features
X_reg = df[['MinTemp', 'MaxTemp', 'WindGustSpeed', 'Humidity', 'Pressure']]

# Target (Temperature)
y_reg = df['Temp']

print("Features:\n", X_reg.head())
print("Target:\n", y_reg.head(), "\n")


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape, "\n")


# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print(" model is trained!\n")


# Prediction
lr_pred = lr_model.predict(X_test)

print("First 5 Predictions:\n", lr_pred[:5])
print("Actual values:\n", y_test.values[:5], "\n")


# Error calculation
lr_error = mean_absolute_error(y_test, lr_pred)
print("Linear Regression Error:", lr_error, "\n")


# Graph 
plt.figure()

plt.plot(y_test.values[:50], marker='o', label="Actual")
plt.plot(lr_pred[:50], marker='x', label="Predicted")

plt.title("Linear Regression (Temperature)")
plt.xlabel("Data Points")
plt.ylabel("Temperature")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# RANDOM FOREST CLASSIFIER (Rain Prediction)
# ============================================================

print(" Random Forest Classifier \n")

# Input features (same)
X_clf = df[['MinTemp', 'MaxTemp', 'WindGustSpeed', 'Humidity', 'Pressure']]

# Target (RainTomorrow)
y_clf = df['RainTomorrow']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape, "\n")


# Train model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

print(" model is trained!\n")


# Prediction
rf_pred = rf_model.predict(X_test)

print("First 10 Predictions:\n", rf_pred[:10])
print("Actual values:\n", y_test.values[:10], "\n")


# Accuracy
rf_acc = accuracy_score(y_test, rf_pred)
print("Accuracy:", rf_acc, "\n")


#  Graph 
plt.figure()

plt.plot(y_test.values[:50], marker='o', label="Actual")
plt.plot(rf_pred[:50], marker='s', label="Predicted")

plt.title("Random Forest (Rain Prediction)")
plt.xlabel("Data Points")
plt.ylabel("Rain (1=Yes, 0=No)")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# SAMPLE PREDICTION
# ============================================================

print(" Sample Prediction...\n")

sample = [[10, 25, 40, 60, 1015]]

# Temperature prediction
temp_pred = lr_model.predict(sample)

# Rain prediction
rain_pred = rf_model.predict(sample)

print("Predicted Temperature:", temp_pred)

if rain_pred[0] == 1:
    print("Rain Tomorrow: Yes ")
else:
    print("Rain Tomorrow: No ")

# -------------------------------
# Save both models
# -------------------------------

pickle.dump(lr_model, open("temp_model.pkl", "wb"))
pickle.dump(rf_model, open("rain_model.pkl", "wb"))

print("Models is saved ")

# -------------------------------
# Load Models
# -------------------------------
temp_model = pickle.load(open("temp_model.pkl", "rb"))
rain_model = pickle.load(open("rain_model.pkl", "rb"))

# -------------------------------
# UI Design
# -------------------------------
st.title(" Weather Prediction App")

st.write("Enter weather details below:")

# Input Parameters
min_temp = st.number_input("Min Temperature")
max_temp = st.number_input("Max Temperature")
wind_speed = st.number_input("Wind Gust Speed")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")


# Prediction Button
if st.button("Predict"):

    input_data = np.array([[min_temp, max_temp, wind_speed, humidity, pressure]])

    # Predictions
    temp = temp_model.predict(input_data)
    rain = rain_model.predict(input_data)

    # Output
    st.subheader("Results:")

    st.success(f" Predicted Temperature: {temp[0]}")

    if rain[0] == 1:
        st.error("Rain Tomorrow: YES")
    else:
        st.success("Rain Tomorrow: NO")