import pandas as pd
class Preprocesser:

    def preprocess_excel_column(self, dataframe, column_name):
        import nltk as nlp
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.stem import WordNetLemmatizer

        nlp.download('punkt')
        nlp.download('stopwords')
        nlp.download('wordnet')

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

        return processed_df


    def preprocess_modular_column(self, dataframe, column_name):
        import nltk as nlp
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.stem import WordNetLemmatizer

        nlp.download('punkt')
        nlp.download('stopwords')
        nlp.download('wordnet')

        # Extract text data from the specified column
        text_data = dataframe[column_name].astype(str)

        # Text preprocessing: lowercasing, stopwords removal, and stemming
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        processed_text_data = []

        for text in text_data:
            # Lowercasing
            text_lower = text.lower()

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

        return processed_df

    def td_idf(self, list):
        # Perform TF-IDF transformation on the cleaned combined titles
        tfidf_matrix_cleaned = tfidf_vectorizer.fit_transform(list)

        # Split the vectors back into preferred and non-preferred groups
        tfidf_pref_cleaned = tfidf_matrix_cleaned[:len(paired_data_cleaned)]
        tfidf_non_pref_cleaned = tfidf_matrix_cleaned[len(paired_data_cleaned):]

    def sort_data_by_preference(self, data: pd.DataFrame, column_name):
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