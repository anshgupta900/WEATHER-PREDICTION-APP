# ============================================================
# WEATHER PREDICTION PROJECT
# ============================================================


# Import all required libraries for data processing,
# machine learning, visualization, model saving, and Streamlit UI.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score


# ------------------------------------------------------------
# STEP 1: Load the weather dataset into a DataFrame.
# ------------------------------------------------------------
df = pd.read_csv("weather.csv")


# ------------------------------------------------------------
# STEP 2: Display the first few rows to understand the dataset.
# ------------------------------------------------------------
print(df.head())


# ------------------------------------------------------------
# STEP 3: Check whether the dataset contains missing values.
# ------------------------------------------------------------
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 4: Remove all rows that contain missing values.
# ------------------------------------------------------------
df = df.dropna()


# ------------------------------------------------------------
# STEP 5: Convert RainTomorrow values from text to numbers.
# Yes = 1
# No = 0
# ------------------------------------------------------------
df["RainTomorrow"] = df["RainTomorrow"].map({
    "Yes":1,
    "No":0
})


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

# ------------------------------------------------------------
# STEP 6: Select input features for temperature prediction.
# ------------------------------------------------------------
X_reg = df[['MinTemp','MaxTemp','WindGustSpeed','Humidity','Pressure']]


# ------------------------------------------------------------
# STEP 7: Select the target column (Temperature).
# ------------------------------------------------------------
y_reg = df['Temp']


# ------------------------------------------------------------
# STEP 8: Split the dataset into training and testing sets.
# 80% for training
# 20% for testing
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------------------
# STEP 9: Create the Linear Regression model.
# ------------------------------------------------------------
lr_model = LinearRegression()


# ------------------------------------------------------------
# STEP 10: Train the model using training data.
# ------------------------------------------------------------
lr_model.fit(X_train, y_train)


# ------------------------------------------------------------
# STEP 11: Predict temperatures using test data.
# ------------------------------------------------------------
lr_pred = lr_model.predict(X_test)


# ------------------------------------------------------------
# STEP 12: Calculate prediction error using MAE.
# ------------------------------------------------------------
lr_error = mean_absolute_error(y_test, lr_pred)


# ------------------------------------------------------------
# STEP 13: Plot Actual vs Predicted Temperature graph.
# ------------------------------------------------------------
plt.figure()

plt.plot(y_test.values[:50], marker='o', label='Actual')
plt.plot(lr_pred[:50], marker='x', label='Predicted')

plt.legend()
plt.show()


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

# ------------------------------------------------------------
# STEP 14: Select input features for rain prediction.
# ------------------------------------------------------------
X_clf = df[['MinTemp','MaxTemp','WindGustSpeed','Humidity','Pressure']]


# ------------------------------------------------------------
# STEP 15: Select RainTomorrow as target variable.
# ------------------------------------------------------------
y_clf = df['RainTomorrow']


# ------------------------------------------------------------
# STEP 16: Split the data into training and testing sets.
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------------------
# STEP 17: Create the Random Forest Classifier.
# ------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100
)


# ------------------------------------------------------------
# STEP 18: Train the Random Forest model.
# ------------------------------------------------------------
rf_model.fit(X_train, y_train)


# ------------------------------------------------------------
# STEP 19: Predict whether it will rain tomorrow.
# ------------------------------------------------------------
rf_pred = rf_model.predict(X_test)


# ------------------------------------------------------------
# STEP 20: Calculate prediction accuracy.
# ------------------------------------------------------------
rf_acc = accuracy_score(y_test, rf_pred)


# ------------------------------------------------------------
# STEP 21: Plot Actual vs Predicted Rain graph.
# ------------------------------------------------------------
plt.figure()

plt.plot(y_test.values[:50], marker='o')
plt.plot(rf_pred[:50], marker='s')

plt.show()


# ============================================================
# SAMPLE PREDICTION
# ============================================================

# ------------------------------------------------------------
# STEP 22: Create a sample weather record for prediction.
# ------------------------------------------------------------
sample = [[10,25,40,60,1015]]


# ------------------------------------------------------------
# STEP 23: Predict tomorrow's temperature.
# ------------------------------------------------------------
temp_pred = lr_model.predict(sample)


# ------------------------------------------------------------
# STEP 24: Predict whether it will rain tomorrow.
# ------------------------------------------------------------
rain_pred = rf_model.predict(sample)


# ============================================================
# SAVE MODELS
# ============================================================

# ------------------------------------------------------------
# STEP 25: Save the trained models as Pickle files.
# ------------------------------------------------------------
pickle.dump(lr_model, open("temp_model.pkl","wb"))
pickle.dump(rf_model, open("rain_model.pkl","wb"))


# ============================================================
# LOAD MODELS
# ============================================================

# ------------------------------------------------------------
# STEP 26: Load the saved models for future predictions.
# ------------------------------------------------------------
temp_model = pickle.load(open("temp_model.pkl","rb"))
rain_model = pickle.load(open("rain_model.pkl","rb"))


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

# ------------------------------------------------------------
# STEP 27: Configure the Streamlit application.
# ------------------------------------------------------------
st.set_page_config(...)


# ------------------------------------------------------------
# STEP 28: Create the application title and header.
# ------------------------------------------------------------
st.title(...)


# ------------------------------------------------------------
# STEP 29: Create sliders for weather input.
# ------------------------------------------------------------
min_temp = st.slider(...)
max_temp = st.slider(...)
wind_speed = st.slider(...)
humidity = st.slider(...)
pressure = st.slider(...)


# ------------------------------------------------------------
# STEP 30: Display the entered weather values.
# ------------------------------------------------------------
st.metric(...)


# ------------------------------------------------------------
# STEP 31: Show weather indicators using progress bars.
# ------------------------------------------------------------
st.progress(...)


# ------------------------------------------------------------
# STEP 32: Predict temperature when the user clicks the button.
# ------------------------------------------------------------
if st.button("🚀 Predict Temperature"):

    prediction = temp_model.predict(
        [[min_temp, max_temp, wind_speed, humidity, pressure]]
    )[0]


# ------------------------------------------------------------
# STEP 33: Display the predicted temperature.
# ------------------------------------------------------------
st.metric(...)


# ------------------------------------------------------------
# STEP 34: Show a weather message based on prediction.
# ------------------------------------------------------------
if prediction > 35:
    st.warning(...)

elif prediction > 25:
    st.success(...)

else:
    st.info(...)


# ------------------------------------------------------------
# STEP 35: Show a celebration animation after prediction.
# ------------------------------------------------------------
st.balloons()
