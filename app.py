import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
data=pd.read_csv("diabetes.csv")
#independentvariables separatin 
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
classifier=KNeighborsClassifier(n_neighbors=5)
# train the clssifier 
classifier.fit(x,y)
#test the classifier
y_pred=classifier.predict(x)
print(accuracy_score(y_pred,y))



st.header("diabetics preduction system")
