import sys


class UnpartedFile:
    def run_all(self):
        import pandas as pd
        import nltk as nlp
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.stem import WordNetLemmatizer
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction.text import TfidfVectorizer

        nlp.download('punkt')
        nlp.download('stopwords')
        nlp.download('wordnet')

        data = pd.read_excel("Corrected_2_Updated_Preferred_titles.xlsx")

        # data = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/Elsevier projekt/Preferred_titles.xlsx")
        # data = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/Elsevier projekt/Skralde_data.xlsx")

        def preprocess_excel_column(dataframe, column_name):
            # Extract text data from the specified column
            text_data = dataframe[column_name].astype(str)

            # Text preprocessing: lowercasing, stopwords removal, and stemming
            stop_words = set(stopwords.words('english'))
            ps = PorterStemmer()
            processed_text_data = []

            for text in text_data:
                # Lowercasing
                text_lower = text.lower()

                # Removing stopwords and stemming
                # words = [ps.stem(word) for word in text_lower.split() if word.isalnum() and word not in stop_words]

                # Removing stopwords and lemmatization
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word) for word in text_lower.split() if
                         word.isalnum() and word not in stop_words]

                # Joining words back to sentences
                processed_text = ' '.join(words)
                processed_text_data.append(processed_text)

            # Create a new DataFrame with the preprocessed text data and keep other columns intact
            processed_df = dataframe.copy()
            processed_df[column_name] = processed_text_data

            print("processed dataframe head 10 to string")
            print(processed_df.head(10).to_string())

            return processed_df

        # Refator TF-IDF ned i seperat metode som vi lettere kan udskife vuderingskriterier
        def td_idf(list):
            # Perform TF-IDF transformation on the cleaned combined titles
            tfidf_matrix_cleaned = tfidf_vectorizer.fit_transform(list)

            # Split the vectors back into preferred and non-preferred groups
            tfidf_pref_cleaned = tfidf_matrix_cleaned[:len(paired_data_cleaned)]
            tfidf_non_pref_cleaned = tfidf_matrix_cleaned[len(paired_data_cleaned):]

        def sort_data_by_preference(Dataframe: data, column_name):
            # Sort the data by doi and Preferred columns to ensure pairs are adjacent
            data_sorted = data.sort_values(by=['doi', f'{column_name}_value'], ascending=[True, False])

            # Split the data into preferred and non-preferred titles
            preferred_titles = data_sorted[data_sorted[f'{column_name}_value'] == 1].reset_index(drop=True)
            non_preferred_titles = data_sorted[data_sorted[f'{column_name}_value'] == 0].reset_index(drop=True)

            # Merge the two dataframes side by side
            paired_data = pd.concat([preferred_titles, non_preferred_titles], axis=1, ignore_index=True)

            # Rename the columns for clarity
            paired_data_renamed = rename_columns(paired_data)
            # Remove rows with NaN values
            paired_data_cleaned = paired_data_renamed.dropna().reset_index(drop=True)

            # Combine the cleaned preferred and non-preferred titles
            combined_titles_cleaned = paired_data_cleaned[f'{column_name}_pref'].tolist() + paired_data_cleaned[
                f'{column_name}_non_pref'].tolist()

            return combined_titles_cleaned

        def rename_columns(data):
            num_columns = len(data.columns)
            middle_index = num_columns // 2

            renamed_data = data.copy()
            for i, column in enumerate(renamed_data.columns):
                if i < middle_index:
                    # Rename the first half of columns to 'something'
                    renamed_data.rename(columns={column: f'{column}_pref'}, inplace=True)
                else:
                    # Rename the second half of columns to 'else'
                    renamed_data.rename(columns={column: f'{column}_non_pref'}, inplace=True)

            return renamed_data

        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))

        data_preprocessed = preprocess_excel_column(data, "Title")

        print("data preprocessed head 10 to string")
        print(data_preprocessed.head(10).to_string())

        # Sort the data by doi and Preferred columns to ensure pairs are adjacent
        data_sorted = data_preprocessed.sort_values(by=['doi', 'Title_value'], ascending=[True, False])

        # Split the data into preferred and non-preferred titles
        preferred_titles = data_sorted[data_sorted['Title_value'] == 1].reset_index(drop=True)
        non_preferred_titles = data_sorted[data_sorted['Title_value'] == 0].reset_index(drop=True)

        # Merge the two dataframes side by side
        paired_data = pd.concat([preferred_titles, non_preferred_titles], axis=1, ignore_index=True)

        # Rename the columns for clarity
        paired_data.columns = ['Id_pref', 'Title_pref', 'doi_pref', 'Preferred_pref',
                               'Id_non_pref', 'Title_non_pref', 'doi_non_pref', 'Preferred_non_pref']

        # Display the transformed data
        paired_data.head()

        # Remove rows with NaN values
        paired_data_cleaned = paired_data.dropna().reset_index(drop=True)

        # Combine the cleaned preferred and non-preferred titles
        combined_titles_cleaned = paired_data_cleaned['Title_pref'].tolist() + paired_data_cleaned[
            'Title_non_pref'].tolist()

        # Perform TF-IDF transformation on the cleaned combined titles
        tfidf_matrix_cleaned = tfidf_vectorizer.fit_transform(combined_titles_cleaned)

        # Split the vectors back into preferred and non-preferred groups
        tfidf_pref_cleaned = tfidf_matrix_cleaned[:len(paired_data_cleaned)]
        tfidf_non_pref_cleaned = tfidf_matrix_cleaned[len(paired_data_cleaned):]

        # Convert the sparse matrices to DataFrames for ease of use
        tfidf_pref_df_cleaned = pd.DataFrame(tfidf_pref_cleaned.toarray(),
                                             columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_non_pref_df_cleaned = pd.DataFrame(tfidf_non_pref_cleaned.toarray(),
                                                 columns=tfidf_vectorizer.get_feature_names_out())

        # Display the first few rows of the feature matrices
        tfidf_pref_df_cleaned.head(), tfidf_non_pref_df_cleaned.head

        # Create the new feature matrix and target vector
        X = tfidf_pref_df_cleaned - tfidf_non_pref_df_cleaned
        X_new = pd.concat([X, -X], ignore_index=True)
        y_new = [1] * len(X) + [0] * len(X)

        print("combined feature matrices with values - training set")
        print(X_new)

        # Split the new data into training and validation sets
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.10)

        # Train the logistic regression model on the new training data
        model_new = LogisticRegression(max_iter=1000)
        model_new.fit(X_train_new, y_train_new)
        X_train_new.head()

        # Predict on the new validation set
        y_pred_new = model_new.predict(X_val_new)

        accuracy_new = accuracy_score(y_val_new, y_pred_new)
        print(str((accuracy_new) * 100) + " % - single run")

        accuracies = []
        for i in range(50):
            # Split the new data into training and validation sets
            X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.04)

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

        print("\n" + str((sum(accuracies) / len(accuracies)) * 100) + " % - avg on multiple runs")

        # import pandas as pd
        # import numpy as np
        # from sklearn.model_selection import train_test_split
        # from sklearn.preprocessing import StandardScaler
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.feature_extraction.text import TfidfVectorizer
        # from sklearn.metrics import accuracy_score
        #
        # # Load the data from the CSV file
        # df = pd.read_excel("Corrected_2_Updated_Preferred_titles.xlsx")
        #
        # # Extract features
        # features = df.apply(lambda row: [
        #     len(row['Title1']), len(row['Title2']),
        #     len(row['Title1'].split()), len(row['Title2'].split()),
        #     len(set(row['Title1'].split()) & set(row['Title2'].split())),
        #     sum(1 for c in row['Title1'] if c.isupper()),
        #     sum(1 for c in row['Title2'] if c.isupper()),
        #     any(char.isdigit() for char in row['Title1']),
        #     any(char.isdigit() for char in row['Title2']),
        #     sum(1 for c in row['Title1'] if not c.isalnum()),
        #     sum(1 for c in row['Title2'] if not c.isalnum()),
        #     "-" in row['Title1'], "-" in row['Title2']], axis=1)
        #
        # # Labels (assuming there's a 'Descriptive' column in the CSV, where 1 means Title1 is more descriptive)
        # labels = df['Descriptive'].values
        #
        # # Calculate TF-IDF
        # tfidf_vectorizer = TfidfVectorizer()
        # tfidf_features1 = tfidf_vectorizer.fit_transform(df['Title1'])
        # tfidf_features2 = tfidf_vectorizer.transform(df['Title2'])
        #
        # # Combine the TF-IDF features with the existing features
        # features = np.hstack((features.values, tfidf_features1.toarray(), tfidf_features2.toarray()))
        #
        # # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        #
        # # Feature Scaling (Optional)
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)
        #
        # # Train the Machine Learning Model (Logistic Regression)
        # model = LogisticRegression()
        # model.fit(X_train, y_train)
        #
        #
        #
        # # Make Predictions
        # y_pred = model.predict(X_test)
        #
        # # Evaluate the Model
        # accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy)
        #
        #
        # def Delta(i1, i2):
        #   val = np.abs(i1-i2)
        #   return val
        #
        # input_text_vectorized = tfidf_vectorizer.transform(["impact solar flare communication system"])
        # input_text_vectorized2 = tfidf_vectorizer.transform(["Gaming and its impact on childhood development"])
        # input_text_vectorized3 = tfidf_vectorizer.transform(["impact solar flare communication network"])
        # value = model_new.predict_proba(input_text_vectorized)[0]
        # value2 = model_new.predict_proba(input_text_vectorized3)[0]
        # print(value,value2)
        # print(Delta(value[1],value2[1]))
