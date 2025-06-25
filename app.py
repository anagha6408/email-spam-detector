from flask import Flask, render_template, request
import pickle
from nltk.stem import WordNetLemmatizer
import re
import string

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['email_content']
    cleaned = clean_text(message)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    confidence = model.predict_proba(vect_text)[0]  # Add this

    prediction = model.predict(vect_text)[0]
    confidence = model.predict_proba(vect_text)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    confidence_pct = round(confidence[prediction] * 100, 2)
    result += f" (Confidence: {confidence_pct}%)"
    # Return the result to the template 
    return render_template('index.html', result=result, confidence=confidence_pct, email=message)
if __name__ == '__main__':
    app.run(debug=True)