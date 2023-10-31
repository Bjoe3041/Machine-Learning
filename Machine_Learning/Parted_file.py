import itertools
import os
import sys

import numpy as np


class PartedFile:
    def run_all(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction.text import TfidfVectorizer
        from Vectorizer import Vectorizer

        data = pd.read_excel("./Corrected_2_Updated_Preferred_titles.xlsx")

        from Preprocesser import Preprocesser
        preprocesser = Preprocesser()
        data_preprocessed = preprocesser.preprocess_excel_column(data, "Title")
        print("data_preprocessed")
        print(data_preprocessed.head(10).to_string())

        # sort data for visualizer
        data_sorted = data_preprocessed.sort_values(by=['doi', 'Title_value'], ascending=[True, False])
            # we are sorting by doi to get all duplicates listed together
            # may be an unnecessary step
        pref_titles = data_sorted[data_sorted['Title_value'] == 1].reset_index(drop=True)
        nonpref_titles = data_sorted[data_sorted['Title_value'] == 0].reset_index(drop=True)
            # seperates pref and non pref
        paired_titles = pd.concat([pref_titles, nonpref_titles], axis=1, ignore_index=True)
        paired_titles.columns = ['id_pref', 'title_pref', 'doi_pref', 'value_pref',
                                 'id_nonpref', 'title_nonpref', 'doi_nonpref', 'value_nonpref']
        paired_titles_clean = paired_titles.dropna().reset_index(drop=True);
            # creates new table, reaplies column names, cleans nan-values
        combined_titles_cleaned = (paired_titles_clean['title_pref'].tolist() +
                                   paired_titles_clean['title_nonpref'].tolist())
            # assigns new list with pref/nonpref
        # continues past next code block
        print(paired_titles_clean.to_string())  # df
        print(combined_titles_cleaned)  # list

        # _______ vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))
        vectorizer = Vectorizer()
        results = vectorizer.TFIDF_Vectorize(data_preprocessed, tfidf_vectorizer)
        # takes dataframe and vectorizer and spits out df and list
        # does the same as "sort data for visualizer?
        X_new = results[0]  # a dataframe
        y_new = results[1]  # a list

        # sort data for visualizer continued
        tfidf_matrix_cleaned = tfidf_vectorizer.fit_transform(combined_titles_cleaned)
            # creates a vectorizer
        tfidf_pref = tfidf_matrix_cleaned[:len(paired_titles_clean)]
        tfidf_nonpref = tfidf_matrix_cleaned[len(paired_titles_clean):]
            # splits the vector into pref and non pref
        tfidf_pref_df = pd.DataFrame(tfidf_pref.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_nonpref_df = pd.DataFrame(tfidf_nonpref.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
            # converts sparse matrices into dataframes
        print(tfidf_pref_df.head(), tfidf_nonpref_df.head())
        X = tfidf_pref_df - tfidf_nonpref_df
        X_new_detailed = pd.concat([X, -X], ignore_index=True)
            # creates a new feature matrix
        y_new_detailed = ([1] * len(X)) + ([0] * len(X))
            # creates a new target vector
        print("combined feature matrices with values - training set")
        print(X_new_detailed)  # can be sorted by ".sort_values(by=['youth'])"

        # Visualize TF-IDF for Preferred Titles
        # TODO sort the data so the 0-values wont be shown
        tfidf_pref_df_nz = tfidf_pref_df[tfidf_pref_df != 0.0]
        tfidf_nonpref_df_nz = tfidf_nonpref_df[tfidf_nonpref_df != 0.0]
        # TODO improve the visuals so it's easier to see individual values?
        sizer = (len(tfidf_pref_df_nz)*2) - len(tfidf_nonpref_df_nz) - 179  # hardcoded ftw
        print('term list/sample size: ' + str(sizer))
        print('> Creating barchart')
        plt.figure(figsize=(sizer, 15))
        plt.bar(tfidf_pref_df_nz.columns, tfidf_pref_df_nz.mean(), label='Preferred Titles',
                alpha=0.7)  # alpha sets opaqueness
        plt.bar(tfidf_nonpref_df_nz.columns, tfidf_nonpref_df_nz.mean(), label='Non-Preferred Titles',
                alpha=0.7)
        plt.axhline()
        plt.grid()
        plt.xlabel('Terms')
        plt.ylabel('TFIDF Mean Value')
        plt.xticks(rotation=90)  # rotates labels for readability
        plt.title('TFIDF Mean Values for Terms')
        print('> Just a moment...')
        plt.show()

        # # OLD
        # sample = 200
        # X_new_sortbyval = X_new[(X_new.sum(axis=1) > 0.0) | (X_new.sum(axis=1) < 0.0)]
        # print('> Creating barchart')
        # plt.figure(figsize=(sample, 15))  # Adjust the figure size as needed
        # plt.bar(X_new_sortbyval.columns[:sample], X_new_sortbyval.mean().head(sample), 0.3, alpha=0.7)
        # plt.axhline()  # add a line at zero
        # plt.grid()  # add grid
        # plt.xlabel('Terms')
        # plt.ylabel('TF-IDF Mean Value')
        # plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        # plt.title('TF-IDF Mean Values for Terms')
        # print('> Just a moment...')
        # plt.show()  # takes about 30 seconds

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
        # print() TODO print this by assigning ^

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
