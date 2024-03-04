from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        if score["compound"] >= 0.05:
            message = "Positive🙂🙂🙂"
        elif score["compound"] <= -0.05:
            message = "Negative🙁🙁🙁"
        else:
            message = "Neutral🙃🙃🙃"
        return render_template('home.html', message=message)
    return render_template('home.html')

# if __name__ == "__main__":
#     app.run(debug=True)
