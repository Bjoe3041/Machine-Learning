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
