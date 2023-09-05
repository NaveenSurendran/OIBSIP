#!/usr/bin/env python
# coding: utf-8

# # OASIS INFOBYTE INTERN,AUGUST -2023

# Naveen S -- Data Science Intern 

# TASK 1   
# 
# IRIS FLOWER CLASSIFICATION:
# 
# Iris flower has three species; setosa, versicolor, and virginica, which differs according to their measurements.
# 
# Now assume that you have the measurements of the iris flowers according to their species, 
# 
# and here your task is to train a machine learning model that can learn from the measurements of the iris species and classify them.

# In[44]:


from IPython.display import Image
Image(filename=r"C:\Users\NAVEEN\OasisImages\iris img.png",width=800,height=900)


# # import Libraries:

# In[45]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # loading the Data:

# In[46]:


iris_df=pd.read_csv(r"C:\Users\NAVEEN\oasisdatas\Irisintern.csv")
print("the data is loaded successfully")


# In[47]:


iris_df.head()


# In[48]:


iris_df


# # DATA PROCESSING:

# In[49]:


iris_df.shape


# In[50]:


iris_df.size


# In[51]:


iris_df.info()


# In[52]:


iris_df.describe()


# In[53]:


iris_df['Species'].value_counts()


# In[54]:


n=iris_df.isnull().sum()
print(n)
print("no null value in dataset")


# In[55]:


print("unique number of values in dataset Species:",iris_df["Species"].nunique())
print("Unique Species in iris dataset:",iris_df["Species"].unique())


# # PAIRPLOT:

# In[56]:


sns.pairplot(iris_df, hue = "Species",markers = "X")
plt.show()
print("*It shows that Iris-Setosa is separated from both other species in all features.")


# # Data Visualization:

# In[57]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=iris_df,hue='Species')
plt.subplot(1,2,2)
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=iris_df,hue='Species')
plt.show()


# In[58]:


#To Check correlation
iris_df.corr()


# In[59]:


#Use Heatmap to
plt.figure(figsize=(10,7))
sns.heatmap(iris_df.corr(),annot = True,cmap = "Oranges_r")
plt.show()


# In[60]:


# Check value counts
iris_df["Species"].value_counts().plot(kind="pie",autopct = "%1.1f%%",shadow=True, figsize=(5,5))
plt.title("Percentage values in each Species", fontsize = 12 , c = "g")
plt.ylabel("",fontsize=10,c="r")
plt.show()


# 1.We can see ,all Species has equal values in dataset.
# 2.Iris-Setosa:50
# 3.Iris- Versicolor:50
# 4.Iris- Virginica : 50

# In[61]:


# Scatterplot for Sepal Length and Sepal Width
sns.scatterplot(iris_df["SepalLengthCm"], iris_df["SepalWidthCm"], hue = iris_df["Species"])
plt.show()


# In[62]:


sns.jointplot(data = iris_df , x = "SepalLengthCm", y = "SepalWidthCm" , size = 7 , hue = "Species")
plt.show()


# In[63]:


plt.figure(figsize = (15,15))
plt.subplot(2,2,1)
sns.barplot(x = "Species",y = "SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Bar plot SepalLengthCm Vs Species")

plt.subplot(2,2,2)
sns.boxplot(x = "Species",y = "SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Bax-plot SepalLengthCm Vs Species")

plt.subplot(2,2,3)
sns.barplot(x = "Species",y = "SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Bar plot SepalLengthCm Vs Species")

plt.subplot(2,2,4)
sns.boxplot(x = "Species",y = "SepalLengthCm", data=iris_df, palette=("Spectral"))
plt.title("Bax-plot SepalLengthCm Vs Species")
plt.show()


# In[64]:


#Distribution Plot 
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
sns.distplot(iris_df["SepalLengthCm"],color="y").set_title("Sepal Length interval")
plt.subplot(2,2,2)
sns.distplot(iris_df["SepalWidthCm"],color="r").set_title("Sepal Width interval")
plt.subplot(2,2,3)
sns.distplot(iris_df["PetalLengthCm"],color="g").set_title("Petal Length interval")
plt.subplot(2,2,4)
sns.distplot(iris_df["PetalWidthCm"],color="b").set_title("Petal Width interval")
plt.show()


# In[65]:


X = iris_df.iloc[:,[0,1,2,3]]
X.head()


# In[66]:


y = iris_df.iloc[:, - 1]
y.head()


# In[67]:


print(X.shape)
print(y.shape)


# # Model Building
# 
# #Supervised Machine Learning
# 
# 
# #Split data into Training and Testing Set

# In[68]:


x= iris_df.drop("Species", axis=1)
y= iris_df["Species"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[69]:


print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("Y_train.shape:", y_train.shape)
print("Y_test.shape:", y_test.shape)


# In[70]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
x_new= np.array([[151, 5, 2.9, 1]])
prediction= knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[71]:


bprint(X_train.shape)
print(X_test.shape)
print(y_train.shabpe)
print(y_test.shape)


# # 6 Different Algorithm

# # Logistic Regression
# # Random forest classifier
# # Decision Tree classifier
# # Support Vector Machine
# # K-NN Classifier 
# # Naive Bayes 

# In[72]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train, y_train)
print("Logistic regression successfully implemented")
y_pred = lr.predict(X_test)
y_pred


# In[73]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:-")
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy is:-",accuracy*100)
print("Classification Report:-")
print(classification_report(y_test,y_pred))


# In[74]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print("Rndom Forest Classifier successfully Implimented")
y_pred = rfc.predict(X_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:- ")
print(cm)

#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# In[75]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
print("Decision Tree Algorithm is successfully implimented.")
y_pred = dtree.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:- ")
print(cm)
#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# # For Visualzing the Decision Tree

# In[76]:


from sklearn.tree import plot_tree

feature = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
classes = ['Iris-Setosa','Iris-Versicolor','Iris-Virginica']
plt.figure(figsize=(10,10))
plot_tree(dtree, feature_names = feature, class_names = classes, filled = True);


# In[77]:


from sklearn.svm import SVC
svc= SVC()
svc.fit(X_train, y_train)
print("Support vactor classifier is successfully implemented")
y_pred = svc.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:- ")
print(cm)
#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# In[78]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 7)
knn.fit(X_train, y_train)
print("K-Nearest Neighbors classifier is successfully implemented")
y_pred = knn.predict(X_test)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:- ")
print(cm)
#accuracy test
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# In[79]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Naive Bayes is successfully implemented")
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print("Confusion Matrix:- ")
print(cm)
# Accuracy test
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:- ", accuracy*100)
print("Classification Report:-")
print( classification_report(y_test, y_pred))


# Result
# 1. Accuracy of Logistic Regression :- 100%
# 2. Accuracy of Random Forest Classifier:-100%
# 3. Accuracy of Decision Tree :- 96.66%
# 4. Accuracy of Support Vector Machine :- 100%
# 5. Accuracy of K-NN Classifier :- 100%
# 6. Accuracy of Naive Bayes :- 100%

# # TESTING THE  MODEL

# In[80]:


input_data=(4.9,3.0,1.4,0.2)
#changing the input data to a numpy array
input_data_as_nparray = np.asarray(input_data)
#reshape the data as we are predicting the label for only the instance
input_data_reshaped = input_data_as_nparray.reshape(1,-1)
prediction = dtree.predict(input_data_reshaped)
print("The category is",prediction)


# # Thank You
