# Titanic Survival Prediction with Logistic Regression

This project implements a logistic regression model from scratch in Python to predict passenger survival on the Titanic. The entire machine learning workflow is covered, from data cleaning and feature engineering to model training, evaluation, and benchmarking against a standard library.

The project emphasizes a foundational understanding of classification algorithms and the data preprocessing pipeline.

📌 Project Overview
This project is organized into two main parts, contained within separate Jupyter notebooks:

**data_cleaning.ipynb:** Loads the raw Titanic dataset (train.csv, test.csv), handles missing values, engineers new features, normalizes numerical data, and encodes categorical variables to create clean, model-ready datasets.

**training.ipynb:** Implements the logistic regression algorithm using NumPy, trains the model on the cleaned data, and evaluates its classification accuracy on the test set.

📂 Project Structure

```plaintext
project-root/
├── data_cleaning.ipynb          # Notebook for data preprocessing
├── training.ipynb               # Notebook for model implementation and training
├── train.csv                    # Raw training data input
├── test.csv                     # Raw testing data input
├── train_cleaned.csv            # Generated training set
└── test_cleaned.csv             # Generated testing set
```

🎯 Key Steps & Techniques

### Data Cleaning & Feature Engineering

* **Feature Creation:** Engineered a FamilySize feature by combining the SibSp (siblings/spouses) and Parch (parents/children) columns.

**Missing Value Imputation:**

* Filled missing Age values with the median age of all passengers.
* Filled missing Fare values with the median fare for the corresponding passenger class.
* Filled missing Embarked values with the most frequent port of embarkation.

**Categorical Encoding:**

* Converted the Sex column to numerical format (female: 0, male: 1).
* Applied one-hot encoding to the Embarked column to create separate binary columns (Embarked_C, Embarked_Q, Embarked_S).

**Feature Scaling:** Normalized Pclass, Age, Fare, and FamilySize to a scale between 0 and 1 using min-max scaling.

**Feature Dropping:** Removed irrelevant columns such as Name, Ticket, and Cabin.

### Model Implementation & Training

* **Algorithm from Scratch:** Implemented a logistic regression model using only NumPy, including the sigmoid function and gradient descent for optimization.

* **Training:** Trained the model for 10,000 epochs with a learning rate of 1.

* **Benchmarking:** Compared the custom model's accuracy against the standard LogisticRegression model from Scikit-learn to validate the implementation.

📊 Evaluation & Results
The model was evaluated on the test set using classification accuracy. The from-scratch implementation achieved an accuracy nearly identical to the Scikit-learn benchmark, confirming its correctness.

* **Custom Model Accuracy:** 77.03%

* **Scikit-learn Model Accuracy:** 77.27%

Below is a plot showing the sigmoid function's fit against the "Age" feature for predicting survival probability.

<img width="567" height="453" alt="image" src="https://github.com/user-attachments/assets/148999d2-b1e5-4b95-9b7a-303a760f1866">

🚀 How to Run

**Prerequisites**

* Python 3.x
* Jupyter Notebook
* Libraries: pandas, numpy, scikit-learn, matplotlib

**Instructions**

1. Clone the repository to your local machine.

```bash
git clone https://github.com/Hassan-Darwish/Titanic-Survival-Prediction-with-Logistic-Regression
```

2. Run the data_cleaning.ipynb notebook first to generate the train_cleaned.csv and test_cleaned.csv files.

3. Run the training.ipynb notebook to train the logistic regression model and view the performance evaluation.

🛠️ Future Enhancements

* Implement more advanced classification models like Support Vector Machines (SVM) or Random Forest.
* Incorporate cross-validation for more robust model evaluation and to prevent overfitting.
* Explore more sophisticated feature engineering techniques.

📜 License

MIT License

👤 Author

Developed by Hassan Darwish
