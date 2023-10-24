import os

import numpy as np


class PartedFile:
    def run_all(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction.text import TfidfVectorizer
        from Vectorizer import  Vectorizer
        import os

        data = pd.read_excel("Machine_Learning/Corrected_2_Updated_Preferred_titles.xlsx")

        from Preprocesser import Preprocesser
        preprocesser = Preprocesser()
        data_preprocessed = preprocesser.preprocess_excel_column(data, "Title")
        #print(data_preprocessed)

        #_______ vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))
        vectorizer = Vectorizer()
        results = vectorizer.TFIDF_Vectorize(data_preprocessed,tfidf_vectorizer)
        X_new = results[0]
        y_new = results[1]


        # # MODEL TRAINING
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.01)

        # Train the logistic regression model on the new training data
        model_new = LogisticRegression(max_iter=1000)
        model_new.fit(X_train_new, y_train_new)
        #X_train_new.head()

        # Predict on the new validation set     MODEL ACCURACY
        y_pred_new = model_new.predict(X_val_new)

        accuracy_new = accuracy_score(y_val_new, y_pred_new)
        print(str((accuracy_new) * 100) + " %")


        while(True):
            print("inputs: title to evaluate, 'q' to quit.")
            s = input()
            print("input: ", s)
            if(s.lower() == "q"):
                break
            #if(s.lower() == "eval"):
            vectored = tfidf_vectorizer.transform([s])
            value = model_new.predict_proba(vectored)

            os.system('cls')
            RED = '\033[91m'
            GREEN = '\033[92m'
            RESET = '\033[0m'  # Reset to default color
            the_value = value[0][1]
            header = ""
            if(the_value > 0.5):
                header = GREEN
            else:
                header = RED
            print(header + "input: ", s)
            print("{:.3f}".format(value[0][1])+RESET)


        #model_new.predict_proba(tfidf_vectorizer.transform(["Test"]))
        #
        # accuracies = []
        # for i in range(50):
        #     # Split the new data into training and validation sets
        #     X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.04)
        #
        #     # Train the logistic regression model on the new training data
        #     model_new = LogisticRegression(max_iter=1000)
        #     model_new.fit(X_train_new, y_train_new)
        #     X_train_new.head()
        #
        #     # Predict on the new validation set
        #     y_pred_new = model_new.predict(X_val_new)
        #
        #     accuracy_new = accuracy_score(y_val_new, y_pred_new)
        #     accuracies.append(accuracy_new)
        #
        # print(str((sum(accuracies) / len(accuracies)) * 100) + " %")