import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from Model.Vectorizer import Vectorizer
from Model.Preprocesser import Preprocesser


class MLController:
    @staticmethod
    def trainmodel(datapath):
        data = pd.read_excel(datapath)
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

        return model_new, tfidf_vectorizer

    @staticmethod
    def savemodel(model, vectorizer, modelname, vectorizername):
        save_directory = "Machine_Learning/Saves/"
        model_filename = modelname + '.pkl'
        joblib.dump(model, save_directory+model_filename)

        # Save the vectorizer to disk
        vectorizer_filename = vectorizername + '.pkl'
        joblib.dump(vectorizer, save_directory+vectorizer_filename)

    @staticmethod
    def loadmodel(model_filename, vectorizer_filename):
        # Load the trained model from disk
        save_directory = "Machine_Learning/Saves/"
        loaded_model = joblib.load(save_directory+model_filename+'.pkl')

        # Load the vectorizer from disk
        loaded_vectorizer = joblib.load(save_directory+vectorizer_filename+'.pkl')

        return loaded_model, loaded_vectorizer

    @staticmethod
    def evaluate(model, vectorizer, inputstring):
        return model.predict_proba(vectorizer.transform([inputstring]))[0][1]
