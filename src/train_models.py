import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report

from preprocessing import load_data, clean_text


df = load_data()

df["text"] = df["text"].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)


vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


models = {

    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()

}


best_model = None
best_score = 0


for name, model in models.items():

    model.fit(X_train_vec, y_train)

    score = model.score(X_test_vec, y_test)

    print(name, score)

    if score > best_score:

        best_score = score
        best_model = model


predictions = best_model.predict(X_test_vec)

print(classification_report(y_test, predictions))


joblib.dump(best_model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully!")