import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def TFIDF_Vectorize(self, data_preprocessed: pd.DataFrame, tfidf_vectorizer: TfidfVectorizer):
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
        # paired_data.head()

        # Remove rows with NaN values
        paired_data_cleaned = paired_data.dropna().reset_index(drop=True)

        # Combine the cleaned preferred and non-preferred titles
        combined_titles_cleaned = paired_data_cleaned['Title_pref'].tolist() + paired_data_cleaned[
            'Title_non_pref'].tolist()

        # Perform TF-IDF transformation on the cleaned combined titles
        # tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 4))
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
        # print(X_new)
        return [X_new, y_new]

    def TFIDF_ModularVectorize(self, data: pd.DataFrame, text_column: str, preference_column: str,
                               tfidf_vectorizer: TfidfVectorizer):
        # Extract the text data and preference indicator columns
        text_data = data[text_column]

        #
        #   TODO: CHANGE 'True' AND 'False' TO 1 AND 0 INT 'preference_column'
        #
        # preference_indicator = (data[preference_column].astype(bool)).astype(int)  # may be temporary
        # print("1\n", data[preference_column])
        # print("2\n", data[preference_column].map({'True': 1, 'False': 0}))
        # print("4\n", data[preference_column].astype(int))
        preference_indicator = data[preference_column].astype(int)

        # Separate the data into preferred and non-preferred based on the preference indicator

        preferred_data = data[preference_indicator == 1].reset_index(drop=True)
        non_preferred_data = data[preference_indicator == 0].reset_index(drop=True)
        # Combine the text data of preferred and non-preferred titles
        combined_titles = pd.concat([preferred_data[text_column], non_preferred_data[text_column]])
        print("_________________________________________")
        # print(preferred_data['title'])
        print(preferred_data['title_is_preferred'])
        print("_________________________________________")
        # print(non_preferred_data['title'])
        print(non_preferred_data['title_is_preferred'])
        print("=========================================")
        print(combined_titles)
        print("_________________________________________")

        # Perform TF-IDF transformation on the combined titles
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_titles)
        # Split the TF-IDF vectors back into preferred and non-preferred groups
        tfidf_pref = tfidf_matrix[:len(preferred_data)]
        tfidf_non_pref = tfidf_matrix[len(preferred_data):]

        # print(tfidf_pref)
        print(tfidf_non_pref)

        # Convert the sparse matrices to DataFrames for ease of use
        tfidf_pref_df = pd.DataFrame(tfidf_pref.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        tfidf_non_pref_df = pd.DataFrame(tfidf_non_pref.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        # Create the new feature matrix and target vector
        X = tfidf_pref_df - tfidf_non_pref_df
        X_new = pd.concat([X, -X], ignore_index=True)
        y_new = [1] * len(X) + [0] * len(X)
        print(X_new, y_new)
        return X_new, y_new
