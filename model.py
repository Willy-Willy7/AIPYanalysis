from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class SentimentModel:
    def __init__(self):
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, text):
        return self.model.predict([text])
