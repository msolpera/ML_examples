# Fake News Detector
This repository contains a complete system for detecting fake news using natural language processing (NLP) techniques and machine learning algorithms.

## Dataset
The project uses two datasets:

- Fake.csv: Contains news labeled as fake
- True.csv: Contains news labeled as true
Each set includes the following fields:
- title: News headline
- text: News content
- subject: News topic
- date: Publication date

## Technologies Used

- Programming Language: Python 3.9

Main Libraries:

- pandas & numpy: Data manipulation
- scikit-learn: Machine learning algorithms and evaluation
- NLTK & spaCy: Natural language processing
- TensorFlow/Keras: Neural models (prepared in the code)
- matplotlib & seaborn: Data visualization
- joblib: Model serialization



## Results
The system evaluates different machine learning models:

- Passive Aggressive Classifier
- Linear SVC
- Logistic Regression
- Multinomial Naive Bayes
- Gradient Boosting Classifier
  
Detailed evaluation results, including the confusion matrix and performance metrics, are generated in the output_fakenews/reports/ directory.
