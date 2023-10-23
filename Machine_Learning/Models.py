class Models:
    def train_logistic_regression(self, X_new, y_new, test_size):
        # Split the new data into training and validation sets
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.10)

        # Train the logistic regression model on the new training data
        model_new = LogisticRegression(max_iter=1000)
        model_new.fit(X_train_new, y_train_new)