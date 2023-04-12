
import pandas as pd
import numpy as np
import re
import pickle
import json 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



#data
data = pd.read_csv("Language Detection.csv")

X = data["Text"]
y = data["Language"]



#########################preprocessing#########################
le = LabelEncoder()
y = le.fit_transform(y)





##############train test split###############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


######################countvectorizer#########################
cv = CountVectorizer()
cv.fit(X_train)

with open("preprocessor.pkl","wb") as f:
        pickle.dump(cv,f)


x_train = cv.transform(X_train).toarray()
x_test  = cv.transform(X_test).toarray()



####################model training############################
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy is :",acc)


macro_averaged_f1score = f1_score(y_test,y_pred, average = 'macro')
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc , "f1_score" :macro_averaged_f1score}, outfile)


with open('model.pkl','wb') as f:
    pickle.dump(model, f)


fig, ax = plt.subplots(figsize=(30, 30))

disp = ConfusionMatrixDisplay.from_estimator(
    model, x_test, y_test, normalize="true", cmap=plt.cm.Blues,ax=ax
)
plt.savefig("confusion_matrix.png")


