Consumer Complaint Text Classification

Project Overview

This project aims to classify consumer complaints from the Consumer Complaint Database into four predefined categories:

0: Credit reporting, repair, or other

1: Debt collection

2: Consumer Loan

3: Mortgage

Steps Followed

1. Exploratory Data Analysis (EDA) and Feature Engineering

Analyzed the dataset structure, missing values, and class distribution.

Performed text cleaning and extracted key features for classification.

2. Text Pre-Processing

Tokenization, stopword removal, and stemming/lemmatization.

Vectorization using techniques like TF-IDF or word embeddings.

3. Selection of Multi-Class Classification Model

Experimented with different classification models such as:

Logistic Regression

Naive Bayes

Support Vector Machines (SVM)

Random Forest

Deep Learning models (LSTM, BERT, etc.)

4. Comparison of Model Performance

Evaluated model accuracy, precision, recall, and F1-score.

Compared different models to select the best-performing classifier.

5. Model Evaluation

Used confusion matrix and ROC curves for evaluation.

Hyperparameter tuning and cross-validation.

6. Prediction

Implemented the final trained model to predict categories on new complaint data.

Demonstrated the working of the model using sample inputs.

Dependencies

To run this project, install the following Python libraries:

pip install pandas numpy scikit-learn nltk seaborn matplotlib tensorflow

How to Run

Load the dataset and preprocess the text.

Train the selected classification model.

Evaluate model performance.

Predict categories for new complaints.

Results

The best-performing model achieves an accuracy of 97%.

The classification results demonstrate effective text classification for consumer complaints.

