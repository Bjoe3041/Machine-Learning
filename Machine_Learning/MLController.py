import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from Vectorizer import Vectorizer


class MLController:
    def trainmodel(self, datapath):
        data = pd.read_excel(datapath)
        from Preprocesser import Preprocesser
        preprocesser = Preprocesser()
        data_preprocessed = preprocesser.preprocess_excel_column(data, "Title")
        # /\-- Preprocess data
        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))
        vectorizer = Vectorizer()
        results = vectorizer.TFIDF_Vectorize(data_preprocessed, tfidf_vectorizer)
        X_new = results[0]
        y_new = results[1]
        # /\-- Vectorize data

        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.01)

        # Train the logistic regression model on the new training data
        model_new = LogisticRegression(max_iter=1000)
        model_new.fit(X_train_new, y_train_new)

        return model_new, vectorizer

        pass

    def savemodel(self, model):
        pass

    def loadmodel(self):
        pass

    def evaluate(self, vectorizer, model, inputstring):
        pass
