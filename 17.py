#!/usr/bin/env python
# coding: utf-8

# In[354]:


import pandas as pd


# In[356]:


data=pd.read_csv(r"C:\Users\Aryan\Desktop\python\Churn_Modelling.csv")
data.head(10)


# In[358]:


new_data=data.drop(["RowNumber","CustomerId","Surname","Geography"],axis='columns')


# In[402]:


new_data.info()


# In[404]:


new_data.fillna(method='ffill',inplace=True)


# In[406]:


new_data.info()


# In[408]:


df=pd.get_dummies(new_data)
new_data.head(10)


# In[410]:


df.astype(float)


# In[412]:


df['Balance'].value_counts()


# In[414]:


df['IsBalanceZero'] = df['Balance'] == 0
df['MultipleProducts'] = df['NumOfProducts'] > 1
df['HasCrCard'].value_counts()


# In[416]:


df.head(5)
df.drop(["HasCrCard"], axis='columns', inplace=True)
df.astype(float)


# In[418]:


from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score


# In[420]:


rux= df.drop('Exited', axis='columns') 
ruy= df['Exited']


# In[422]:


ru = RandomUnderSampler(random_state=42)
x, y = ru.fit_resample(rux, ruy)


# In[424]:


# x['Balance'].value_counts()


# In[426]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[428]:


rfc=RandomForestClassifier(max_depth=5, min_samples_leaf=4, min_samples_split=5)
rfc.fit(x_train,y_train)


# In[429]:


rfc.score(x_test,y_test)


# In[432]:


y_predict = rfc.predict(x_test)
y_pred=rfc.predict(x_train)


# In[434]:


test_a = accuracy_score(y_test,y_predict)
train_a = accuracy_score(y_train,y_pred)
print("test_accuracy",test_a)
print("train_accurary",train_a)


# In[436]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_predict)
cm


# In[438]:


precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
print("recall:",recall)
print("precision:", precision)


# In[440]:


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_predict)
f1


# In[442]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rfc_best, x, y, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')


# In[293]:


# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(x_train, y_train)

# print(f'Best Parameters: {grid_search.best_params_}')


# In[294]:


# best_params = grid_search.best_params_
# rfc_best = RandomForestClassifier(**best_params) 
# rfc_best.fit(x_train, y_train)


# In[295]:


# rfc_best = RandomForestClassifier(**best_params) 
# rfc_best.fit(x_train, y_train)
# rfc.score(x_test,y_test)


# In[296]:


# y_predict = rfc_best.predict(x_test) 
# y_pred = rfc_best.predict(x_train)


# In[297]:


# test_a = accuracy_score(y_test, y_predict) 
# train_a = accuracy_score(y_train, y_pred) 
# print(f'Test Accuracy: {test_a}') 
# print(f'Train Accuracy: {train_a}')


# In[298]:


# precision = precision_score(y_test, y_predict) 
# recall = recall_score(y_test, y_predict) 
# f1 = f1_score(y_test, y_predict) 
# print(f'Precision: {precision}') 
# print(f'Recall: {recall}') 
# print(f'F1 Score: {f1}')


# In[299]:


# importances = rfc_best.feature_importances_
# feature_names = x_train.columns
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# print(feature_importance_df)


# In[300]:


# from sklearn.model_selection import cross_val_score

# # Initialize the Random Forest model with the best parameters
# rfc_best = RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=100)

# # Perform cross-validation
# cv_scores = cross_val_score(rfc_best, x, y, cv=5)

# # Print cross-validation scores
# print(f'Cross-Validation Scores: {cv_scores}')
# print(f'Mean CV Score: {cv_scores.mean()}')


# In[ ]:





# In[ ]:




