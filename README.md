# NLP_Projet
Prediction of financial indicators (from text)

## Target
Familiar with natural language processing, through the prediction of financial indicators to judge whether the prediction is positive or negative or neutral

## Precondition
The following libraries need to be installed:

+ `pip install numpy`
+ `pip install panda`
+ `pip install nltk`
+ `pip install matplotlib`
+ `pip install sklearn`
+ `pip install wordcloud`
+ `pip install spacy`
+ `pip install elim5`
+ `pip install gensim`

## Introduction:
We use economic news and its sentiment analysis as the data set, economic news as the data of the training set, sentiment polarity as the prediction result, and train through logistic regression.

Then we processed the economic news in three different ways: no processing, nltk processing, and spacy processing

Different models are generated for each case, and scores are obtained, as well as test prediction results

## Process:
I. Data collecting
II. Text Preprocessing
III. statistics
IV. Word vectorization and model training
