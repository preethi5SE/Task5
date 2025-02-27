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

Used confusion matrix for evaluation.


Hyperparameter tuning and cross-validation.

6. Prediction

Implemented the final trained model to predict categories on new complaint data.

Demonstrated the working of the model using sample inputs.

Dependencies

To run this project, install the following Python libraries:
pip install pandas numpy scikit-learn nltk seaborn matplotlib tensorflow

pip install pandas numpy scikit-learn nltk seaborn matplotlib tensorflow

How to Run

Load the dataset and preprocess the text.

Train the selected classification model.

Evaluate model performance.

Predict categories for new complaints.

Results


![image](https://github.com/user-attachments/assets/2714d733-d3b8-429c-b151-8fb990395f46)
![image](https://github.com/user-attachments/assets/7af404aa-8f57-49d5-a9bf-d561fe634241)
![image](https://github.com/user-attachments/assets/4ca9877f-e581-45c4-ab3b-8c1cddec90de)
![image](https://github.com/user-attachments/assets/ab940cf3-e86f-41d4-84ae-c4f864c2d429)
![image](https://github.com/user-attachments/assets/db4ad530-f611-4963-a23b-9b35e9aaa1bb)




The best-performing model achieves an accuracy of 97.4%.

The classification results demonstrate effective text classification for consumer complaints.

