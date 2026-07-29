# Weather Prediction ML System

This project presents a machine learning-based weather prediction system designed to estimate temperature and predict rainfall for the following day. It integrates data preprocessing, predictive modeling, performance evaluation, and an interactive user interface into a single end-to-end solution.

The system leverages:
- **Linear Regression** for continuous temperature prediction
- **Random Forest Classifier** for binary rainfall prediction (Yes/No)

A Streamlit-based web application is developed to allow users to input weather parameters and receive real-time predictions.

## Key Features

- End-to-end machine learning pipeline
- Data cleaning and preprocessing
- Dual-model architecture for regression and classification tasks
- Model performance evaluation using MAE and accuracy metrics
- Visualization of actual vs predicted values
- Model persistence using Pickle
- Interactive and user-friendly web interface using Streamlit

## Input Parameters

The model uses the following meteorological features:
- Minimum Temperature
- Maximum Temperature
- Wind Gust Speed
- Humidity
- Atmospheric Pressure

## Outputs

- Predicted Temperature
- Rainfall Prediction (Yes/No)

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## Application

The application allows users to input weather conditions and instantly receive predictions, making it suitable for basic weather analysis and educational purposes in machine learning workflows.

## Future Scope

- Integration with real-time weather APIs
- Implementation of advanced models (e.g., Gradient Boosting, Deep Learning)
- Deployment on cloud platforms
- Enhanced feature engineering and hyperparameter tuning
