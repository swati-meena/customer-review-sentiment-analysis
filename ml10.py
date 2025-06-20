import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv(r'/home/hp/Downloads/datasets/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
'''from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values'''


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X = tf.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)



'''
    ac1-0.675 ---countVectorizer
    ac2-0.715 -- tfidf
'''

'''from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=25)
classifier.fit(X_train,y_train)'''



''' ac-.0.72 -- tfidf'''
 
'''from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)'''

''' ac-.0.76 -- tfidf'''
'''from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.9)
classifier.fit(X_train,y_train)
'''


''' ac-.0.675 -- tfidf'''

'''from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)
'''


from lightgbm import LGBMClassifier
classifier = LGBMClassifier(learning_rate=0.1, random_state=42)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train,y_train)
print(bias)

variance = classifier.score(X_test,y_test)
print(variance)








































