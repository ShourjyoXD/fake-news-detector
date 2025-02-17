from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    
    return jsonify({"prediction": "REAL" if prediction == 1 else "FAKE"})

if __name__ == '__main__':
    app.run(debug=True)
