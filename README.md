# Bitcoin Price Forecasting Application

This project is a Flask-based web application that forecasts Bitcoin prices for future dates based on historical price data. The application uses a logistic regression model to analyze price trends and visualize forecasts through an interactive web interface.

---

## Features

- **Forecasting Bitcoin Prices**: Determines whether the Bitcoin price is expected to increase or decrease for a future date.
- **Interactive Web Interface**: Users can input today's Bitcoin price and select a future date for forecasting.
- **Visual Representation**: Generates a dynamic chart displaying historical prices and future forecasts.
- **Feature Engineering**: Incorporates features like `open-close`, `low-high`, and quarter-end information for enhanced forecasting accuracy.

---

## Prerequisites

Before running this application, ensure you have the following installed:

- Python 3.7 or above
- Required Python libraries:
  - Flask
  - Pandas
  - NumPy
  - scikit-learn
  - Matplotlib

---

## Running the Application

1. Open your web browser and navigate to:
   http://127.0.0.1:5000/
---

## Usage

1. **Input Today's Bitcoin Price**: Enter the current Bitcoin price in USD.
2. **Select a Future Date**: Choose a date for which you want to forecast the price trend.
3. **View Results**:
   - The application will display whether the price is expected to increase or decrease on the chosen date.
   - A dynamic chart shows historical prices and the forecast.

---

## Application Workflow

1. **Data Preparation**:
   - Loads historical Bitcoin data and performs feature engineering.
   - Extracts key features: `open-close`, `low-high`, and `is_quarter_end`.

2. **Model Training**:
   - A logistic regression model is trained on the processed data to identify trends in price changes.

3. **Forecasting**:
   - Based on user input (current price and future date), the model predicts the price movement trend.

4. **Visualization**:
   - A chart displays the historical price trends, along with the forecasted trend for the selected future date.

---

## Example Output

- **Forecast Result**: 
  "Price is expected to INCREASE 📈 for 25-12-2025."
  
- **Dynamic Chart**: A plot showcasing historical Bitcoin prices and the forecasted future trend.

---
