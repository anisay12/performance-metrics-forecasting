# ARIMA-Based Forecasting for 4G Network Performance Metrics

This repository contains a Python project that utilizes ARIMA models to forecast various performance metrics of a 4G network. The primary objective is to predict the 4G ERAB Drop Rate, but the methods can be extended to other metrics as well.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [ARIMA Forecasting Function](#arima-forecasting-function)
- [Visualization](#visualization)
- [Evaluation Metrics](#evaluation-metrics)

## Introduction

This project demonstrates the application of ARIMA (AutoRegressive Integrated Moving Average) models to predict network performance metrics. The code reads network data from a CSV file, preprocesses it, and then applies ARIMA models to make forecasts. The predicted values are then compared with actual values to evaluate the model's performance using metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Prerequisites

Ensure you have the following libraries installed in your Python environment:

- pandas
- matplotlib
- statsmodels
- scikit-learn
- seaborn
- pmdarima

## Installation

Use the following command to install the required libraries:

```bash
pip install pandas matplotlib statsmodels scikit-learn seaborn pmdarima
```

## Data Preparation

The data is read from a CSV file named `4G.csv`. Ensure that this file is available in the specified path. The data contains columns related to 4G network performance metrics.

```python
ActualData = pd.read_csv('4G.csv', sep=';', encoding="ISO-8859-1")
```

The relevant columns are then selected and preprocessed, including handling null values and converting data types as necessary.

## Usage

1. **Load Data**:
   The data is loaded and initial information is printed to understand its structure.
   
   ```python
   print(ActualData.columns)
   ActualData.info()
   ```

2. **Preprocess Data**:
   Handle null values and convert columns to appropriate data types.
   
   ```python
   ActualData.dropna()
   ActualData["4G_SSR_%"] = pd.to_numeric(ActualData["4G_SSR_%"], downcast="float")
   ```

3. **Visualize Correlations**:
   Use seaborn to visualize the correlation between different variables.
   
   ```python
   sns.heatmap(ActualData.corr())
   ```

4. **Train-Test Split**:
   Split the data into training and testing sets. Typically, 70% of the data is used for training, and 30% for testing.
   
   ```python
   TrainingSize = int(NumberOfElements * 0.8)
   TrainingData = Data_Pico_Basile[0:TrainingSize]
   TestData = Data_Pico_Basile[TrainingSize:NumberOfElements]
   ```

5. **Fit ARIMA Model**:
   Fit the ARIMA model on the training data and make predictions.
   
   ```python
   model = pm.auto_arima(Data_Pico_Basile, start_p=1, start_q=1, max_p=3, max_q=3, m=1, start_P=0, d=1, D=1, trace=True)
   model.fit(TrainingData)
   future_forecast = model.predict(n_periods=255)
   ```

6. **Evaluate Model**:
   Evaluate the model's performance using MSE and MAE.
   
   ```python
   Error = mean_squared_error(TestData, Predictions)
   Error1 = mean_absolute_error(TestData, Predictions)
   print('Test Mean Squared Error (MSE): %.3f' % Error)
   print('Test Mean Absolute Error (MAE): %.3f' % Error1)
   ```

## ARIMA Forecasting Function

The `StartARIMAForecasting` function is used to fit an ARIMA model and make predictions.

```python
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction
```

## Visualization

Visualize the actual and predicted values using matplotlib.

```python
ax = df1.plot(figsize=(15,6), y=["4G_ERAB_Drop_Rate_%"], color='blue')
df.plot(figsize=(15,6), y=["Prediction"], color='red')
```

## Evaluation Metrics

The performance of the model is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

```python
Error = mean_squared_error(TestData, Predictions)
Error1 = mean_absolute_error(TestData, Predictions)
```
