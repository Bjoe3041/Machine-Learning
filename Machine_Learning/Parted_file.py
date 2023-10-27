import os
import sys

import numpy as np


class PartedFile:
    def run_all(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction.text import TfidfVectorizer
        from Vectorizer import Vectorizer
        import os

        data = pd.read_excel("./Corrected_2_Updated_Preferred_titles.xlsx")

        from Preprocesser import Preprocesser
        preprocesser = Preprocesser()
        data_preprocessed = preprocesser.preprocess_excel_column(data, "Title")
        print("data_preprocessed")
        print(data_preprocessed.head(10).to_string())

        # _______ vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))  # TODO agree on ngram range
        vectorizer = Vectorizer()
        results = vectorizer.TFIDF_Vectorize(data_preprocessed, tfidf_vectorizer)
        X_new = results[0]  # a dataframe
        y_new = results[1]  # a list

        print("combined feature matrices with values - training set")
        print(X_new)  # can be sorted by ".sort_values(by=['youth'])"
        # print("combined feature matrices with values - test set")
        # print(y_new)  # just a list of 0 and 1's

        # # TODO Visualize TF-IDF for Preferred Titles
        # plt.figure(figsize=(400, 6))
        #     # lots of titles, need long figure, smaller makes words unreadable, larger makes ticks clutter
        # plt.bar(tfidf_pref_df_cleaned.columns, tfidf_pref_df_cleaned.mean(), label='Preferred Titles', alpha=0.7)
        #     # alpha sets opaqueness
        # plt.bar(tfidf_non_pref_df_cleaned.columns, tfidf_non_pref_df_cleaned.mean(), label='Non-Preferred Titles',
        #   alpha=0.7)
        # plt.xlabel('Terms')
        # plt.ylabel('TF-IDF Mean Value')
        # plt.xticks(rotation=90)
        # plt.legend()
        # plt.show()  # takes about 30 sec

        # # MODEL TRAINING
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.2)
        # X is the training set, 80% - y is the test set/the correct values, 20%.
        # X is further split into 80% training data and 20% validation data,
        # the same goes for y although these are used to validate X

        # Train the logistic regression model on the new training data
        model_new = LogisticRegression(max_iter=1000)
        model_new.fit(X_train_new, y_train_new)

        # Predict on the new validation set     MODEL ACCURACY
        y_pred_new = model_new.predict(X_val_new)

        accuracy_new = accuracy_score(y_val_new, y_pred_new)
        print(str((accuracy_new) * 100) + " % - result of single run prediction")

        model_new.predict_proba(tfidf_vectorizer.transform(["Test"]))

        accuracies = []
        for i in range(50):
            # Split the new data into training and validation sets
            X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.2)

            # Train the logistic regression model on the new training data
            model_new = LogisticRegression(max_iter=1000)
            model_new.fit(X_train_new, y_train_new)
            X_train_new.head()

            # Predict on the new validation set
            y_pred_new = model_new.predict(X_val_new)

            accuracy_new = accuracy_score(y_val_new, y_pred_new)
            accuracies.append(accuracy_new)

            sys.stdout.write('.')  # a loading bar while waiting for the while loop to finish
            sys.stdout.flush()

        print("\n" + str((sum(accuracies) / len(accuracies)) * 100) + " % - avg of multiple run test")

        while (True):
            print("inputs: title to evaluate, 'q' to quit.")
            s = input()
            print("input: ", s)
            if (s.lower() == "q"):
                break
            # if(s.lower() == "eval"):
            vectored = tfidf_vectorizer.transform([s])
            value = model_new.predict_proba(vectored)

            os.system('cls')
            RED = '\033[91m'
            GREEN = '\033[92m'
            RESET = '\033[0m'  # Reset to default color
            the_value = value[0][1]
            header = ""
            if (the_value > 0.5):
                header = GREEN
            else:
                header = RED
            print(header + "input: ", s)
            print("{:.3f}".format(value[0][1]) + RESET)