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

        # # sort data for visualizer
        # data_sorted = data_preprocessed.sort_values(by=['doi', 'Title_value'], ascending=[True, False])
        #     # we are sorting by doi to get all duplicates listed together
        #     # may be an unnecessary step
        # pref_titles = data_sorted[data_sorted['Title_value'] == 1].reset_index(drop=True)
        # nonpref_titles = data_sorted[data_sorted['Title_value'] == 0].reset_index(drop=True)
        #     # seperates pref and non pref
        # paired_titles = pd.concat([pref_titles, nonpref_titles], axis=1, ignore_index=True)
        # paired_titles.columns = ['id_pref', 'title_pref', 'doi_pref', 'value_pref',
        #                          'id_nonpref', 'title_nonpref', 'doi_nonpref', 'value_nonpref']
        # paired_titles_clean = paired_titles.dropna().reset_index(drop=True);
        #     # creates new table, reaplies column names, cleans nan-values
        #
        # print(paired_titles_clean.to_string())

        # _______ vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))  # TODO agree on ngram range
        vectorizer = Vectorizer()
        results = vectorizer.TFIDF_Vectorize(data_preprocessed, tfidf_vectorizer)
            # takes dataframe and vectorizer and spits out df and list
            # does the same as "sort data for visualizer?
        X_new = results[0]  # a dataframe
        y_new = results[1]  # a list

        print("combined feature matrices with values - training set")
        print(X_new)  # can be sorted by ".sort_values(by=['youth'])"

        # Visualize TF-IDF for Preferred Titles
        # TODO sort the data so the 0-values wont be shown
        # TODO improve the visuals so it's easier to see individual values? fx make a grid/color pref/nonpref
        sample = 200
        print('> Creating barchart')
        plt.figure(figsize=(sample, 15))  # Adjust the figure size as needed
        plt.bar(X_new.columns[:sample], X_new.mean().head(sample), 0.3, alpha=0.7)
        plt.xlabel('Terms')
        plt.ylabel('TF-IDF Mean Value')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.title('TF-IDF Mean Values for Terms')
        print('> Just a moment...')
        plt.show()  # takes about 30 seconds

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

        model_new.predict_proba(tfidf_vectorizer.transform(["Test"]))  # returns predicted and rest value?
        # print() TODO print this ^

        accuracies = []
        for i in range(5):
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

            # TODO scale with range, fx if range > 50 only print for each x number
            sys.stdout.write('.')  # a loading bar while waiting for the while loop to finish
            sys.stdout.flush()

        print("\n" + str((sum(accuracies) / len(accuracies)) * 100) + " % - avg of multiple run test")

        # while (True):
        #     print("inputs: title to evaluate, 'q' to quit.")
        #     s = input()
        #     print("input: ", s)
        #     if (s.lower() == "q"):
        #         break
        #     # if(s.lower() == "eval"):
        #     vectored = tfidf_vectorizer.transform([s])
        #     value = model_new.predict_proba(vectored)
        #
        #     os.system('cls')
        #     RED = '\033[91m'
        #     GREEN = '\033[92m'
        #     RESET = '\033[0m'  # Reset to default color
        #     the_value = value[0][1]
        #     header = ""
        #     if (the_value > 0.5):
        #         header = GREEN
        #     else:
        #         header = RED
        #     print(header + "input: ", s)
        #     print("{:.3f}".format(value[0][1]) + RESET)
