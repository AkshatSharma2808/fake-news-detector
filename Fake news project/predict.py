import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred

if __name__ == "__main__":
    msg = input("Enter news text: ")
    print("Prediction:", predict_news(msg))
