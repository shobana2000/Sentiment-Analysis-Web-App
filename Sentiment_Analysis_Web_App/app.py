from flask import Flask, render_template, request
from textblob import TextBlob
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

# Download necessary NLTK data
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '')
    file = request.files.get('file')

    if not text and not file:
        return render_template('index.html', error='Please enter text or upload a file.')

    if file:
        text = file.read().decode('utf-8')
        file.seek(0)

    overall_sentiment, positive_percentage, negative_percentage = analyze_sentiment(text)

    # Create and save the bar chart
    plot_url = plot_sentiment_chart(positive_percentage, negative_percentage)

    return render_template('result.html', text=text, overall_sentiment=overall_sentiment,
                           positive_percentage=positive_percentage, negative_percentage=negative_percentage,
                           plot_url=plot_url)


def analyze_sentiment(text):
    blob = TextBlob(text)
    words = [word for word in blob.words if word.lower() not in stop_words]
    filtered_text = ' '.join(words)

    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        overall_sentiment = "Positive"
    elif polarity < -0.2:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    positive_percentage = max(0, min(100, (polarity + 1) * 50))
    negative_percentage = 100 - positive_percentage

    return overall_sentiment, positive_percentage, negative_percentage


def plot_sentiment_chart(positive_percentage, negative_percentage):
    labels = ['Positive', 'Negative']
    percentages = [positive_percentage, negative_percentage]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, percentages, color=['green', 'red'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    plt.ylabel('Percentage')
    plt.title('Sentiment Analysis Results')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the image to base64 for displaying in HTML
    plot_url = base64.b64encode(img.getvalue()).decode()

    return f'data:image/png;base64,{plot_url}'


if __name__ == '__main__':
    app.run(debug=True)
