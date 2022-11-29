from flask import Flask, request
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import json

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    nb_spam_model = open("./models/NB_Spam_Model.pkl", "rb")
    classifier = joblib.load(nb_spam_model)

    vocab_file = open("./models/vocab.pkl", "rb")
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 1), min_df=1, vocabulary=joblib.load(vocab_file)
    )
    message = request.json["message"]
    message_vector = count_vectorizer.transform([message]).toarray()
    my_prediction = classifier.predict(message_vector)

    if my_prediction[0] == "ham":
        return json.dumps({"message": "This is not a spam message."})

    return json.dumps({"message": "This is a spam message."})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
