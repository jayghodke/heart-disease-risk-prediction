#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv(r'C:\Users\owner\Desktop\framingham.csv')


# In[2]:


data.head(10)


# In[3]:


data.shape


# In[4]:


data.dtypes


# In[5]:


duplicate_data = data[data.duplicated()]
duplicate_data


# In[6]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


fig = plt.figure(figsize = (15,15))
ax = fig.gca()
data.hist(ax = ax)


# In[8]:


data_corr = data.corr()
sns.heatmap(data_corr)


# In[9]:


data = data.drop(['education'], axis=1)


# In[10]:


data.isna().sum()


# In[11]:


data = data.dropna()
data.isna().sum()
data.columns


# In[12]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[13]:


X = data.iloc[:,0:14]
Y = data.iloc[:,-1]

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,Y)
datascores = pd.DataFrame(fit.scores_)
datacolumns = pd.DataFrame(X.columns)


# In[14]:


featureScores = pd.concat([datacolumns, datascores], axis=1)
featureScores.columns = ['Parameters','Values']  
print(featureScores.nlargest(14,'Values'))


# In[15]:


featureScores = featureScores.sort_values(by='Values', ascending=False)
featureScores


# In[16]:


plt.figure(figsize=(30,10))
sns.barplot(x='Parameters', y='Values', data=featureScores, palette = "jet_d")
plt.box(False)
plt.xlabel('\n Features', fontsize=14)
plt.ylabel('Importance \n', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[17]:


features_list = featureScores["Parameters"].tolist()[:10]
features_list


# In[18]:


data = data[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]
data.head()


# In[20]:


data_corr = data.corr()
sns.heatmap(data_corr, cmap = 'RdYlBu')


# In[21]:


data.describe()
sns.pairplot(data)


# In[22]:


sns.boxplot(data.totChol)
outliers = data[(data['totChol'] > 500)] 
outliers


# In[23]:


data = data.drop(data[data.totChol > 500].index)
sns.boxplot(data.totChol)


# In[24]:


data_clean = data


# In[25]:


scaler = MinMaxScaler(feature_range=(0,1)) 
data_scaled = pd.DataFrame(scaler.fit_transform(data_clean), columns=data_clean.columns)


# In[26]:


data_scaled.describe()
data.describe()


# In[27]:


Y = data_scaled['TenYearCHD']
X = data_scaled.drop(['TenYearCHD'], axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=25)


# In[28]:


len(X_train)
len(X_test)


# In[29]:


target_count = data_scaled.TenYearCHD.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
sns.countplot(data_scaled.TenYearCHD, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=10)
plt.ylabel('Patient Count',fontsize=10)
plt.title('Patients Heart Disease Outcome\n')
plt.show()


# In[30]:


shuffled_data = data_scaled.sample(frac=1,random_state=4)

# Put all the fraud class in a separate dataset.
CHD_data = shuffled_data.loc[shuffled_data['TenYearCHD'] == 1]

#Randomly select 492 observations from the non-fraud (majority class)
non_CHD_data = shuffled_data.loc[shuffled_data['TenYearCHD'] == 0].sample(n=611,random_state=42)

# Concatenate both dataframes again
normalized_data = pd.concat([CHD_data, non_CHD_data])

# check new class counts
normalized_data.TenYearCHD.value_counts()

# plot new count
sns.countplot(normalized_data.TenYearCHD, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=11)
plt.ylabel('Patient Count',fontsize=11)
plt.title('Patients Heart Disease Outcome after Resampling\n')
#plt.savefig('Balance Heart Disease.png')
plt.show()


# In[31]:


Y_train = normalized_data['TenYearCHD']
X_train = normalized_data.drop('TenYearCHD', axis=1)

from sklearn.pipeline import Pipeline

classifiers = [LogisticRegression(),SVC(),DecisionTreeClassifier(),KNeighborsClassifier(2)]

for classifier in classifiers:
    pipe = Pipeline(steps=[('classifier', classifier)])
    pipe.fit(X_train, Y_train)   
    print("The accuracy score of {0} is: {1:.2f}%".format(classifier,(pipe.score(X_test, Y_test)*100)))


# In[32]:


#logistic regression
normalized_data_reg = LogisticRegression().fit(X_train, Y_train)

normalized_data_reg_pred = normalized_data_reg.predict(X_test)

acc = accuracy_score(Y_test, normalized_data_reg_pred)
print(f"The accuracy score for LogReg is: {round(acc,3)*100}%")

f1 = f1_score(Y_test, normalized_data_reg_pred)
print(f"The f1 score for LogReg is: {round(f1,3)*100}%")

precision = precision_score(Y_test, normalized_data_reg_pred)
print(f"The precision score for LogReg is: {round(precision,3)*100}%")

recall = recall_score(Y_test, normalized_data_reg_pred)
print(f"The recall score for LogReg is: {round(recall,3)*100}%")


# In[33]:


cnf_matrix_log = confusion_matrix(Y_test, normalized_data_reg_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True,cmap="Blues" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix Logistic Regression\n', y=1.1)


# In[34]:


#SVM
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, Y_train)
    svc_scores.append(svc_classifier.score(X_test, Y_test))


# In[35]:


from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.bar(kernels, svc_scores, color = colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Accuracy')
plt.title('Support Vector Classifier scores for different kernels')


# In[36]:


#Decision tress
dtree = DecisionTreeClassifier()

dtree.fit(X_train, Y_train)

normalized_data_dtree_pred = dtree.predict(X_test)

acc = accuracy_score(Y_test, normalized_data_dtree_pred)
print(f"The accuracy score for DTC is: {round(acc,3)*100}%")

f1 = f1_score(Y_test, normalized_data_dtree_pred)
print(f"The f1 score for DTC is: {round(f1,3)*100}%")

precision = precision_score(Y_test, normalized_data_dtree_pred)
print(f"The precision score for DTC is: {round(precision,3)*100}%")

recall = recall_score(Y_test, normalized_data_dtree_pred)
print(f"The recall score for DTC is: {round(recall,3)*100}%")


# In[37]:


dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, Y_train)
    dt_scores.append(dt_classifier.score(X_test, Y_test))


# In[40]:


plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'blue')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Accuracy')
plt.title('Decision Tree Classifier Accuracy for different number of maximum features')


# In[38]:


# plotting confusion matrix Decision Tree

cnf_matrix_dtree = confusion_matrix(Y_test, normalized_data_dtree_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_dtree), annot=True,cmap="Blues" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix Decision Tree\n', y=1.1)


# In[39]:


knn_scores = []
for k in range(1,11):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, Y_train)
    knn_scores.append(knn_classifier.score(X_test, Y_test))
    


# In[41]:


plt.plot([k for k in range(1, 11)], knn_scores, color = 'red')
for i in range(1,11):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 11)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('K Neighbors Classifier scores for different K values')


# In[42]:



#initialize model
knn = KNeighborsClassifier(n_neighbors = 2)

#fit model
knn.fit(X_train, Y_train)

# prediction = knn.predict(x_test)
normalized_data_knn_pred = knn.predict(X_test)

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(Y_test, normalized_data_knn_pred)
print(f"The accuracy score for KNN is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(Y_test, normalized_data_knn_pred)
print(f"The f1 score for KNN is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(Y_test, normalized_data_knn_pred)
print(f"The precision score for KNN is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(Y_test, normalized_data_knn_pred)
print(f"The recall score for KNN is: {round(recall,3)*100}%")


# In[43]:


acc_test = knn.score(X_test, Y_test)
print("The accuracy score of the test data is: ",acc_test*100,"%")
acc_train = knn.score(X_train, Y_train)
print("The accuracy score of the training data is: ",round(acc_train*100,2),"%")


# In[44]:


'''Cross Validation is used to assess the predictive performance of the models and and to judge 
how they perform outside the sample to a new data set'''

cv_results = cross_val_score(knn, X, Y, cv=6) 

print ("Cross-validated scores:", cv_results)
print("The Accuracy of Model with Cross Validation is: {0:.2f}%".format(cv_results.mean() * 100))


# In[45]:


# plotting confusion matrix KNN

cnf_matrix_knn = confusion_matrix(Y_test, normalized_data_knn_pred)

ax= plt.subplot()
sns.heatmap(pd.DataFrame(cnf_matrix_knn), annot=True,cmap="Blues" , fmt='g')

ax.set_xlabel('Predicted ');ax.set_ylabel('True'); 


# In[46]:


'''the AUC ROC Curve is a measure of performance based on plotting the true positive and false positive rate 
and calculating the area under that curve.The closer the score to 1 the better the algorithm's ability to 
distinguish between the two outcome classes.'''

fpr, tpr, _ = roc_curve(Y_test, normalized_data_knn_pred)
auc = roc_auc_score(Y_test, normalized_data_knn_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.box(False)
plt.title ('ROC CURVE KNN')
plt.show()

print(f"The score for the AUC ROC Curve is: {round(auc,3)*100}%")


# In[47]:


def start_questionnaire():
    my_predictors = []
    parameters=['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male']
    
    print('Input Patient Information:')
    
    age = input("Patient's age: >>> ") 
    my_predictors.append(age)
    male = input("Patient's gender. male=1, female=0: >>> ") 
    my_predictors.append(male)
    cigsPerDay = input("Patient's smoked cigarettes per day: >>> ") 
    my_predictors.append(cigsPerDay)
    sysBP = input("Patient's systolic blood pressure: >>> ") 
    my_predictors.append(sysBP)
    diaBP = input("Patient's diastolic blood pressure: >>> ")
    my_predictors.append(diaBP)
    totChol = input("Patient's cholesterin level: >>> ") 
    my_predictors.append(totChol)
    prevalentHyp = input("Was Patient hypertensive? Yes=1, No=0 >>> ") 
    my_predictors.append(prevalentHyp)
    diabetes = input("Did Patient have diabetes? Yes=1, No=0 >>> ") 
    my_predictors.append(diabetes)
    glucose = input("What is the Patient's glucose level? >>> ") 
    my_predictors.append(diabetes)
    BPMeds = input("Has Patient been on Blood Pressure Medication? Yes=1, No=0 >>> ")
    my_predictors.append(BPMeds)
    
    my_data = dict(zip(parameters, my_predictors))
    my_df = pd.DataFrame(my_data, index=[0])
    scaler = MinMaxScaler(feature_range=(0,1)) 
   
    # assign scaler to column:
    my_df_scaled = pd.DataFrame(scaler.fit_transform(my_df), columns=my_df.columns)
    my_y_pred = knn.predict(my_df)
    print('\n')
    print('Result:')
    if my_y_pred == 1:
        print("The patient will develop a Heart Disease.")
    if my_y_pred == 0:
        print("The patient will not develop a Heart Disease.")
        
start_questionnaire()


# In[ ]:




