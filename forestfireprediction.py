import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import preprocessing, neighbors
df=pd.read_csv("winequality-red.csv")
##print(df.head())
##df.drop('quality',1)
x=df.drop("quality",1)
y=df["quality"]
x=preprocessing.scale(x)
##print(y)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)
##exampledata=np.array([6.2,3.1,3.5,1.1])
##exampledata=exampledata.reshape(1,-1)
##prediction=clf.predict(exampledata)
##print(prediction)
