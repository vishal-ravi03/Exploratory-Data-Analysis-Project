# 🚢 Titanic Survival Prediction - Exploratory Data Analysis & Baseline Model

## 📌 Project Overview

This project analyzes the classic Kaggle Titanic dataset (`titanic_train.csv`) to perform Exploratory Data Analysis (EDA). The goal is to uncover patterns related to passenger survival based on features like passenger class, sex, age, fare, and embarkation point. The analysis includes data visualization, handling missing data, converting categorical features, and building a baseline Logistic Regression model for survival prediction.

## ⚙️ Technologies Used

* Python 3
* Jupyter Notebook
* Pandas – for data cleaning and manipulation
* NumPy – for numerical operations
* Matplotlib & Seaborn – for data visualization
* Scikit-learn – for model building (Logistic Regression) and evaluation

## 📂 Dataset

The dataset (`titanic_train.csv`) contains information about passengers aboard the RMS Titanic, including whether they survived the disaster. Key attributes include:

* `PassengerId`: Unique ID for each passenger.
* `Survived`: Survival status (0 = No, 1 = Yes) - **Target variable**.
* `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
* `Name`: Passenger's name.
* `Sex`: Passenger's sex (male, female).
* `Age`: Passenger's age in years.
* `SibSp`: Number of siblings/spouses aboard.
* `Parch`: Number of parents/children aboard.
* `Ticket`: Ticket number.
* `Fare`: Passenger fare paid.
* `Cabin`: Cabin number (contains many missing values).
* `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## 🔑 Key Steps in the Analysis

1.  **Data Loading & Inspection:**
    * Read the `titanic_train.csv` dataset into a Pandas DataFrame.
    * Inspect the data using `.head()`, `.info()`, and `.isnull().sum()` to understand structure, data types, and missing values.

2.  **Exploratory Data Analysis & Visualization:**
    * Visualize missing data using a `seaborn` heatmap.
    * Create count plots to understand the distribution of `Survived`, `Sex`, `Pclass`, and `SibSp`.
    * Analyze the `Age` distribution using histograms and distribution plots.
    * Explore relationships between survival and other features (`Sex`, `Pclass`) using count plots with hues.
    * Analyze the `Fare` distribution using a histogram.
    * Use a box plot to examine the relationship between `Pclass` and `Age`.

3.  **Data Cleaning:**
    * Impute missing `Age` values based on the average age for each `Pclass`.
    * Drop the `Cabin` column due to a high percentage of missing data.
    * Remove rows with missing `Embarked` values (only a few).

4.  **Feature Engineering:**
    * Convert categorical features (`Sex`, `Embarked`) into numerical dummy variables using `pd.get_dummies()`.
    * Drop original non-numeric or less informative columns (`Sex`, `Embarked`, `Name`, `Ticket`).

5.  **Model Building & Evaluation:**
    * Separate features (X) and the target variable (y = `Survived`).
    * Split the data into training and testing sets using `train_test_split`.
    * Train a `LogisticRegression` model on the training data.
    * Make predictions on the test set.
    * Evaluate the model's performance using a `classification_report`.

## 📊  Insights & Findings

* Missing data was primarily found in `Age` and `Cabin`. `Age` was imputed, while `Cabin` was dropped.
* Survival rates varied significantly based on `Sex` (females had higher survival) and `Pclass` (higher classes had higher survival).
* The `Age` distribution showed fewer very young children and elderly passengers compared to young adults.
* Passenger ages tended to differ across passenger classes, justifying imputation based on `Pclass`.
* A baseline Logistic Regression model was built and evaluated, providing initial predictive performance metrics (see notebook for details).

## ▶️ How to Run

1.  Clone the repository or download the project files (`Titanic_EDA.ipynb`, `titanic_train.csv`).
2.  Ensure you have Python and the required libraries installed. If not, install them using pip:
    ```sh
    pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab
    ```
3.  Navigate to the project directory in your terminal.
4.  Launch Jupyter Notebook or Jupyter Lab:
    ```sh
    jupyter lab
    # or
    # jupyter notebook
    ```
5.  Open the `Titanic_EDA.ipynb` notebook.
6.  Run the cells sequentially to reproduce the analysis, visualizations, and model results.
