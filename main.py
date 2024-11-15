from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model import SentimentModel

def main():
    data = load_data('data/sample_reviews.csv')
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    model = SentimentModel()
    model.train(X_train, y_train)

    # Example prediction
    review = "I love this product!"
    sentiment = model.predict(review)
    print(f"The sentiment of the review is: {sentiment[0]}")

if __name__ == "__main__":
    main()
