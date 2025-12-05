import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("=== Fake News Detector ===")
while True:
    news = input("\nEnter news text (or 'quit'): ")
    
    if news.lower() == "quit":
        break

    vec = vectorizer.transform([news])
    output = model.predict(vec)[0]

    print("🟢 REAL NEWS" if output == "real" else "🔴 FAKE NEWS")
