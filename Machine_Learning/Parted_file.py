import sys

class PartedFile:
    def run_all(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction.text import TfidfVectorizer
        from Model.Vectorizer import Vectorizer

        data = pd.read_excel("./Corrected_2_Updated_Preferred_titles.xlsx")

        from Model.Preprocesser import Preprocesser
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
        paired_titles_clean = paired_titles.dropna().reset_index(drop=True)
        # creates new table, reaplies column names, cleans nan-values
        combined_titles_cleaned = (paired_titles_clean['title_pref'].tolist() +
                                   paired_titles_clean['title_nonpref'].tolist())
        # assigns new list with pref/nonpref
        # continues past next code block
        print(paired_titles_clean.to_string())  # df
        print(combined_titles_cleaned)  # list

        # _______ vectorizer TODO needs a rework or to be phased out
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
        tfidf_pref_df_nz = tfidf_pref_df[tfidf_pref_df != 0.0]  # removes zero-values
        tfidf_nonpref_df_nz = tfidf_nonpref_df[tfidf_nonpref_df != 0.0]
        print('term list/sample size: ' + str(len(tfidf_pref_df_nz)))  # assuming nonpref and pref has the same length

        print('> Creating barchart')
        sample = 50
        plt.figure(figsize=(25, sample))  # width, height - height of figure scales with samplesize

        # plt.bar(ticks on y-axis = up until samplesize, width of bars = scales with mean value,
        # label for legend, height (width) of bars, alpha = bar opaqueness)
        plt.barh(tfidf_pref_df_nz.columns[:sample], tfidf_pref_df_nz.mean().head(sample),
                 label='Preferred Titles', height=0.3, alpha=0.7)
        plt.barh(tfidf_nonpref_df_nz.columns[:sample], tfidf_nonpref_df_nz.mean().head(sample),
                 label='Non-Preferred Titles', height=0.3, alpha=0.7)

        plt.grid()  # makes grid visible, for better visualization
        plt.ylabel('Terms')
        plt.xlabel('TFIDF Mean Value')
        plt.title('TFIDF Mean Values for Terms')
        plt.legend()  # adds bar label/color legend
        print('> Just a moment...')
        plt.show()

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

        prob_est = model_new.predict_proba(tfidf_vectorizer.transform(["Test"]))  # returns predicted and rest value?
        print('probability estimation: ' + str(prob_est))

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

            sys.stdout.write('.')  # a loading bar while waiting for the while loop to finish, does not scale well
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
