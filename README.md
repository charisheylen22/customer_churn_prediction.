# Customer Churn Prediction.
End-to-end Machine Learning project to predict customer churn using XGBoost and a Streamlit web application. | 
Projet d'apprentissage automatique de bout en bout pour prédire le désabonnement client (churn) en utilisant XGBoost et une application web Streamlit.

This project is an end-to-end Machine Learning solution designed to predict customer **churn risk** (attrition). It includes a comprehensive analysis and model training pipeline, as well as an interactive web interface built with Streamlit for real-time predictions.

## Project Overview

The primary goal is to identify customers who are likely to stop using the service based on various characteristics such as demographics, behavioral data, membership details, and more.

The project is divided into two main parts:

1.  **Data Science & Training (`01_data_preparation.ipynb`)**: Data preparation, cleaning, feature engineering, training of multiple models (Logistic Regression, Random Forest, XGBoost), and saving the best-performing model.
2.  **Web Application (`streamlit_app.py`)**: A user-friendly interface that allows users to input customer data and receive an instant churn prediction.

##  Project Structure

```bash
├── data/
│   └── customers.csv          # Raw dataset (source)
├── models/
│   ├── best_churn_model.pkl   # Trained model (XGBoost)
│   └── preprocessor.pkl       # Data transformation pipeline
├── 01_data_preparation.ipynb  # Jupyter Notebook (EDA, Training, Evaluation)
├── streamlit_app.py           # Streamlit Web Application
└── README.md                  # Project Documentation
```

##  Installation and Prerequisites

Ensure you have Python installed (version 3.8+ recommended).

1.  **Clone the repository (or download the files)**

2.  **Install dependencies**
    It is recommended to create a virtual environment. Install the necessary libraries using:

    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib streamlit
    ```

## Usage

### 1\. Model Training (Optional)

If you wish to retrain the model or explore the data analysis:

  * Open `01_data_preparation.ipynb` in Jupyter Notebook or VS Code.
  * Run all cells.
  * This will generate `best_churn_model.pkl` and `preprocessor.pkl` (make sure to move them to a `models/` directory if necessary for the Streamlit app).

### 2\. Running the Prediction App

To launch the web interface:

1.  Open your terminal in the project root directory.

2.  Run the following command:

    ```bash
    streamlit run streamlit_app.py
    ```

3.  A web page will automatically open in your browser (usually at `http://localhost:8501`).

4.  Fill in the customer details in the form and click **Predict**.

## Model Performance

Several models were evaluated. Here are the accuracy results on the test set:

| Model | Accuracy |
| :--- | :--- |
| **XGBoost** | **\~93.2%** |
| Random Forest | \~92.8% |
| Logistic Regression | \~86.0% |

The **XGBoost** model was selected and saved for the final application due to its superior performance.

**Top 3 Key Factors Influencing Churn:**

1.  **Points in Wallet**: The customer's points balance.
2.  **Membership Category**: Type of membership (No Membership, Basic, Premium, etc.).
3.  **Avg Transaction Value**: The average value of transactions.

## Technologies Used

  * **Python**: Primary programming language.
  * **Pandas & NumPy**: Data manipulation and numerical operations.
  * **Scikit-Learn**: Data preprocessing and baseline modeling.
  * **XGBoost**: High-performance Gradient Boosting algorithm.
  * **Matplotlib & Seaborn**: Data visualization.
  * **Streamlit**: Interactive web application framework.
  * **Joblib**: Model serialization and saving.

## Author

Charismata DIANGANZI HEYLEN

Project realized as part of a predictive customer retention analysis.
