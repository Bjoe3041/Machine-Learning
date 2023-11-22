import pandas as pd
class Preprocesser:


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

