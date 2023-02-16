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
```Python
import numpy as np 
import pandas as pd 
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize 
from nltk.stem import PorterStemmer
import spacy
nltk.download('punkt')
nltk.download('stopwords')

# module to split data into training / test
from sklearn.model_selection import train_test_split
!python -m spacy download en_core_web_sm

# Simple BoW vectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression

# Or TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

import eli5

from gensim.corpora.dictionary import Dictionary
```
## Introduction:
We use economic news and its sentiment analysis as the data set, economic news as the data of the training set, sentiment polarity as the prediction result, and train through logistic regression.

Then we processed the economic news in three different ways: no processing, nltk processing, and spacy processing

Different models are generated for each case, and scores are obtained, as well as test prediction results

## Data description
This data was collected through Kaggle : https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

This dataset (FinancialPhraseBank) contains the sentiments for financial news headlines from the perspective of a retail investor. And the dataset contains two columns, "Sentiment" and "News Headline". The sentiment can be negative, neutral or positive.

## Process:
I. Data collecting
```Python
# read data
df = pd.read_csv('data/all-data.csv', delimiter=',', encoding='latin-1', header=None).fillna('')
df = df.rename(columns=lambda x: ['sentiment', 'text'][x])
print(df.shape)
df.head()

```
```Python
df
```

II. Text Preprocessing
```Python
df2=df.copy()
for i,line in enumerate(df2["text"]):
    cutwords1 = word_tokenize(line)
    print('\n【Results after NLTK word segmentation：】')
    print(cutwords1)
    
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']   #liste punctuation marks
    cutwords2 = [word.lower() for word in cutwords1 if word not in interpunctuations]   #remove punctuation
    print('\n【Remove symbol results after NLTK word segmentation：】')
    print(cutwords2)
    
    stops = set(stopwords.words("english")) #stop words
    cutwords3 = [word for word in cutwords2 if word not in stops]
    print('\n【Remove stop word results after NLTK word segmentation：】')
    print(cutwords3)

    df2.loc[i,"text"]=cutwords3
```
```Python
df2
```
```Python
df3=df2.copy()
for i,t in enumerate(df3["text"]):
    df3.loc[i,"text"]=" ".join(t)
df3
```

III. statistics
III.1. Statistical Sentiment Value Ratio
```Python
cnt_pro = df['sentiment'].value_counts()
plt.figure(figsize=(12,4))
plt.bar(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=14)
plt.xlabel('sentiment', fontsize=14)
plt.show();
```
III.2. Draw word cloud
```Python
words_neu=[]
words_pos=[]
words_neg=[]
for i,sen in enumerate (df2["sentiment"]):
    if sen == "neutral" :
        line = df2.loc[i,"text"]
        words_neu.extend(line)
    if sen == "positive" :
        line = df2.loc[i,"text"]
        words_pos.extend(line)
    if sen == "negative" :
        line = df2.loc[i,"text"]
        words_neg.extend(line)
```
```Python
stop_words = set(stopwords.words('english'))
words_neu_dict={}
words_pos_dict={}
words_neg_dict={}
for w in words_neu:
    if w not in stop_words and len(w)>2 and w not in words_neu_dict:
        words_neu_dict[w]=words_neu.count(w)
for w in words_pos:
    if w not in stop_words and len(w)>2 and w not in words_pos_dict:
        words_pos_dict[w]=words_pos.count(w)
for w in words_neg:
    if w not in stop_words and len(w)>2 and w not in words_neg_dict:
        words_neg_dict[w]=words_neg.count(w)
        
        
words_neu_dict =sorted(words_neu_dict.items(), key=lambda x: x[1], reverse=True)
for i in range(5):
    del words_neu_dict[0]
words_neu_dict=dict(words_neu_dict)


words_neg_dict = sorted(words_neg_dict.items(), key=lambda x: x[1], reverse=True)
for i in range(5):
    del words_neg_dict[0]
words_neg_dict=dict(words_neg_dict)
    

words_pos_dict = sorted(words_pos_dict.items(), key=lambda x: x[1], reverse=True)
for i in range(5):
    del words_pos_dict[0]
words_pos_dict=dict(words_pos_dict)
    

# objet WordCloud
wordcloud = WordCloud(width=600, height=400).generate_from_frequencies(words_neu_dict)

# draw
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(" neutral wordcloud")
plt.axis("off")
plt.show()
```
```Python
wordcloud2 = WordCloud(width=600, height=400).generate_from_frequencies(words_pos_dict)

# draw
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.title(" positive wordcloud")
plt.show()
```
```Python
# wordcloud
wordcloud3 = WordCloud(width=600, height=400).generate_from_frequencies(words_neg_dict)

# draw
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud3, interpolation='bilinear')
plt.title("negative wordcloud")
plt.axis("off")
plt.show()
```

IV. Word vectorization and model training
IV.1. Sklearn
IV.1.1. Raw Text
```Python
X = df['text'].values
y = df["sentiment"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
```
IV.1.2. word vectorization + Logistic Regression
```Python
# Simple BoW vectorizer
vectorizer = CountVectorizer()
X_train_vec_1 = vectorizer.fit_transform(X_train)

# Instantiate a logistic regression model
model = LogisticRegression(max_iter=2000)

# Train the model
model.fit(X_train_vec_1, y_train)

# Transform the test-set
X_test_vec_1 = vectorizer.transform(X_test)

# Check performance of the model
model.score(X_test_vec_1, y_test)

# Predict on new data
y_pred = model.predict(X_test_vec_1)
y_pred

# confusion matrix by hand... :-)
pd.crosstab(y_test, y_pred)
```
IV.1.3. TF-IDF + Logistic Regression
```Python
# Or TFIDF
vectorizer = TfidfVectorizer()
X_train_vec_2 = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=2000)

# Train the model
model.fit(X_train_vec_2, y_train)

# Transform the test-set
X_test_vec_2 = vectorizer.transform(X_test)

# Check performance of the model
model.score(X_test_vec_2, y_test)

import eli5
eli5.show_weights(model, feature_names=vectorizer.get_feature_names_out(), target_names=['negative','neutral','positive'], top=20)
eli5.show_prediction(model, X_test[0], vec=vectorizer, target_names=['negative','neutral','positive'])
```
IV.1.4. Prediction
```Python
test_data = ["The Federal  is expected to drop interest rates in the next quarter."]
test_features = vectorizer.transform(test_data)
predicted_value = model.predict(test_features)
predicted_value
```
IV.1.5. TF-IDF + Logistic Regression + processed data
```Python
X = df3['text'].values
y = df3["sentiment"].values

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=18)
vectorizer = TfidfVectorizer()
X_train_vec_3 = vectorizer.fit_transform(X_train_2)

model = LogisticRegression(max_iter=2000)

# Train the model

model.fit(X_train_vec_3, y_train_2)
# Transform the test-set
X_test_vec_2 = vectorizer.transform(X_test)
# Check performance of the model
model.score(X_test_vec_2, y_test)
```
IV.2. SPACY
```Python
# load the small english language model. Large models can be downloaded for many languages.
nlp = spacy.load("en_core_web_sm")

doc = nlp(X_test[1])
[(tok.text, tok.pos_) for tok in doc]

len(X_train)

tokenlist = []
for doc in nlp.pipe(X_train[:3500]):
    tokens =[tok.text.lower() for tok in doc if tok.pos_ in ['NOUN','ADJ','ADV','VERB'] and not tok.is_stop]
    tokenlist.append(tokens)
 
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(tokenlist)

len(dictionary)
dictionary.filter_extremes(no_below=5, no_above=0.2)
len(dictionary)
dictionary[0]

vectorizer = TfidfVectorizer(vocabulary=list(dictionary.values()))
X_train_vec_2 = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=2000)

# Train the model
model.fit(X_train_vec_2, y_train)
X_test_vec_2 = vectorizer.fit_transform(X_test)

# Check performance of the model
model.score(X_test_vec_2, y_test)

eli5.show_weights(model, feature_names=vectorizer.get_feature_names_out(), target_names=['negative','neutral','positive'], top=20)
eli5.show_prediction(model, X_test[0], vec=vectorizer, target_names=['negative','neutral','positive'])

test_data = ["The Federal  is expected to drop interest rates in the next quarter."]
test_features = vectorizer.transform(test_data)
predicted_value = model.predict(test_features)
predicted_value
```

## Execution:
open terminal in the /NLP_Projet and input:

`jupyter notebook`

then click **Prediction_Financial_Indicators**
