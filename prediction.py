from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('bitcoin_data.csv')

# Parse the Timestamp column as datetime for proper comparison
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Feature engineering
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['is_quarter_end'] = np.where(df['Timestamp'].dt.month.isin([3, 6, 9, 12]), 1, 0)
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Features and Target
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Train-Test Split
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Front-end HTML and CSS template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin Price Prediction</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f4f4f9;
        }
        .container {
            text-align: center;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            font-size: 16px;
            margin-bottom: 5px;
            display: inline-block;
        }
        input {
            padding: 8px;
            font-size: 16px;
            width: 80%;
            margin-top: 5px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 20px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
        }
        #priceChart {
            margin-top: 30px;
            width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Bitcoin Price Prediction</h1>
        </header>
        <div class="form-group">
            <label for="currentPrice">Enter Today's Bitcoin Price:</label>
            <input type="number" id="currentPrice" placeholder="e.g., 30000">
        </div>
        <div class="form-group">
            <label for="predictionDate">Enter the Future Date for Prediction (DD-MM-YYYY):</label>
            <input type="date" id="predictionDate">
        </div>
        <button class="predict-btn" onclick="predict()">Predict</button>
        <div id="result" class="result"></div>
        <img id="priceChart" src="" alt="Price Prediction Graph">
    </div>

    <script>
        async function predict() {
            const price = document.getElementById('currentPrice').value;
            const predictionDate = document.getElementById('predictionDate').value;
            if (!price || !predictionDate) {
                alert('Please enter a valid price and date');
                return;
            }

            // Fetch prediction and graph data from the back-end
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ currentPrice: price, predictionDate: predictionDate })
            });

            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            // Display Prediction Result
            const result = data.prediction === 1 ? "Price is expected to INCREASE 📈" : "Price is expected to DECREASE 📉";
            document.getElementById('result').innerText = `Prediction: ${result} for ${data.prediction_date}`;

            // Display the Chart Image
            document.getElementById('priceChart').src = data.chart_url;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    current_price = float(data['currentPrice'])
    prediction_date = pd.to_datetime(data['predictionDate'])

    # Ensure the date is in the future
    historical_date = df['Timestamp'].max()
    if prediction_date <= historical_date:
        return jsonify({'error': 'Please enter a future date'}), 400

    # Use the last known data point to predict for the future
    last_row = df.iloc[-1]
    features = np.array([ 
        last_row['Open'] - current_price,
        last_row['Low'] - last_row['High'],
        1 if last_row['Timestamp'].month % 3 == 0 else 0
    ]).reshape(1, -1)

    # Predict the change in price (increase or decrease)
    prediction = model.predict(features)[0]

    # Create a basic plot for visual representation
    historical = df.tail(30)
    dates = historical['Timestamp'].dt.strftime('%Y-%m-%d').tolist()
    prices = historical['Close'].tolist()
    
    # Add the current price to the plot for reference
    dates.append(prediction_date.strftime('%Y-%m-%d'))
    prices.append(current_price)

    fig, ax = plt.subplots()
    ax.plot(dates, prices, marker='o', color='b', label='Bitcoin Price')
    ax.axvline(x=prediction_date.strftime('%Y-%m-%d'), color='r', linestyle='--', label='Prediction Date')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.set_title('Bitcoin Price Prediction')

    # Save the plot to a BytesIO buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    
    # Convert the image to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # Return the prediction and chart image
    return jsonify({
        'prediction': int(prediction),
        'prediction_date': prediction_date.strftime('%d-%m-%Y'),
        'chart_url': f"data:image/png;base64,{img_base64}"
    })

if __name__ == '__main__':
    app.run(debug=True)
