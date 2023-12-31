import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import ApiAccess.ApiAccess
from Model.Vectorizer import Vectorizer
from Model.Preprocesser import Preprocesser


class MLController:
    @staticmethod
    def trainmodel_excel(datapath):
        data = pd.read_excel(datapath)

        data_preprocessed = MLController.preprocess(data, "Title")
        # results, tfidf_vectorizer = MLController.vectorize(data_preprocessed)
        results, tfidf_vectorizer = MLController.vectorize(data_preprocessed)
        X_new = results[0]
        y_new = results[1]
        # /\-- Vectorize data

        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.01)

        # Train the logistic regression model on the new training data
        model_new = LogisticRegression(max_iter=1000)
        model_new.fit(X_train_new, y_train_new)

        return model_new, tfidf_vectorizer

    @staticmethod
    def trainmodel_database():
        articles = ApiAccess.ApiAccess.get_articles()
        db_data = pd.DataFrame(articles)
        data = db_data[['id', 'title', 'doi', 'title_is_preferred']].copy()
        data.columns = ['Id', 'title', 'doi', 'Title_value']

        data_preprocessed = MLController.preprocess(data, "title")
        results, tfidf_vectorizer = MLController.vectorize(data_preprocessed)
        X_new = results[0]
        y_new = results[1]

        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.01)

        # Train the logistic regression model on the new training data
        model_new = LogisticRegression(max_iter=1000)
        model_new.fit(X_train_new, y_train_new)

        return model_new, tfidf_vectorizer

    @staticmethod
    def preprocess(data, preprocessed_column_name):
        preprocesser = Preprocesser()
        data_preprocessed = preprocesser.preprocess_modular_column(data, preprocessed_column_name)
        return data_preprocessed

    @staticmethod
    def vectorize(data_preprocessed):
        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))
        vectorizer = Vectorizer()
        results = vectorizer.TFIDF_Vectorize(data_preprocessed, tfidf_vectorizer)
        return results, tfidf_vectorizer

    @staticmethod
    def savemodel(model, vectorizer, modelname, vectorizername):
        illegal_characters_and_names = [
            '<', '>', ':', '"', '/', '\\', '|', '?', '*',  # Illegal characters for Windows
            'CON', 'PRN', 'AUX', 'NUL',  # Reserved filenames
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5',  # More reserved filenames
            'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5',
            'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]

        def filtername(name):
            tempname = name
            for char in illegal_characters_and_names:  # Remove illegal symbols from the filename
                if char in tempname:
                    tempname = tempname.replace(char, "-")

            for reserved_name in illegal_characters_and_names[
                                 8:]:  # This slices the list to only include reserved names
                if tempname.upper().startswith(reserved_name):  # They are always caps btw
                    tempname = tempname.replace(reserved_name,
                                                reserved_name + "_")  # just add something to it, I suppose
            return tempname

        modelname = filtername(modelname)
        vectorizername = filtername(vectorizername)

        save_directory = "Machine_Learning/Saves/"
        model_filename = modelname + '.pkl'
        joblib.dump(model, save_directory + model_filename)

        # Save the vectorizer to disk
        vectorizer_filename = vectorizername + '.pkl'
        joblib.dump(vectorizer, save_directory + vectorizer_filename)

    @staticmethod
    def loadmodel(model_filename, vectorizer_filename):
        # Load the trained model from disk
        save_directory = "Machine_Learning/Saves/"
        loaded_model = joblib.load(save_directory + model_filename)

        # Load the vectorizer from disk
        loaded_vectorizer = joblib.load(save_directory + vectorizer_filename)

        return loaded_model, loaded_vectorizer

    @staticmethod
    def evaluate(inputstring):
        name = MLController.getchosenmodelpath()
        modelname = "model_" + name
        vectorizername = "vector_" + name
        loadedmodel, loadedvectorizer = MLController.loadmodel(modelname, vectorizername)
        return loadedmodel.predict_proba(loadedvectorizer.transform([inputstring]))[0][1]

    @staticmethod
    def getchosenmodelpath():
        with open('Machine_Learning/modelpath.txt', 'r') as file:
            content = file.read()
        return content

    @staticmethod
    def setchosenmodelpath(content):
        with open('Machine_Learning/modelpath.txt', 'w') as file:
            file.write(str(content))

    @staticmethod
    def has_modelpath():
        with open('Machine_Learning/modelpath.txt', 'r') as file:
            return (file.read() != "")

    # Todo methods: modelpath_matches_save
    # Todo methods: has_saves

    @staticmethod
    def evaluatetitle(inputtitle):
        retvalue = MLController.evaluate(inputtitle)

        return retvalue
