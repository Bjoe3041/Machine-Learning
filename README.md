# Machine-Learning
A system that can train new Machine Learning models, based on data from a postgreSQL database. Allows you to create new models easily and name them. Includes a GUI to make it even easier to create new models.
It connects to an API that returns a list of Articles, and using those articles it trains the model. It also functions as an API to use the models on JSON objects being sent to the API.

---

This system makes it as easy as can be to create new models whenever you want. And it also combines all models in program so there doesn't need to be a new project for each model trained.



>---


## Endpoints
#### All endpoints of the API and their descriptions

---

### GET /api/status
- **Description**: Checks the API server status.
- **Output**: JSON object with the server status.
  - **Example**: 
      ```json
      {
          "status": "Server is up"
      }
      ```

---

### POST /api/predict
- **Description**: Predicts how favorable a Title is.


- **Input**: JSON object with a title.
  - **Example**:
    ```json
    {
        "title": "The effects of embedded json on the human psyche"
    }
    ```
    
- **Output**: JSON object with a prediction result.
  - **Example**:
    ```json
    {
        "result": 0.5748231285346042
    }
    ```
  
---


### POST /api/predict/compare
- **Description**: Compares two titles, evaluates their preferability for each, and returns their evaluations and difference thereof.


- **Input**: JSON object with two titles.
  - **Example**: 
    ```json
    {
        "title1": "The significance of reading documentation",
        "title2": "Applied theory of API endpoints"
    }
    ```

- **Output**: JSON object containing the preferability percentages of each title and the delta, which is the difference in their percentages.
  - **Example**: 
    ```json
    {
        "result": {
            "delta": "1.869%",
            "title1": [
                "49.065%",
                "The significance of reading documentation"
            ],
            "title2": [
                "50.935%",
                "Applied theory of API endpoints"
            ]
        }
    }
    ```
- **Details**:
  - **Delta**: Shows the percentage difference in preferability between the two titles.
  - **Preferability Percentage || 'title1' and 'title2'**: Indicates how preferable each title is predicted to be.

    > 📝 **Note**: Each Title is returned with its preferability percentage.
