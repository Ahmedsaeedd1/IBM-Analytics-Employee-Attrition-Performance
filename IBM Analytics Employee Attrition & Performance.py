#!/usr/bin/env python
# coding: utf-8

# # IBM HR Analytics Employee Attrition & Performance
# 
# Uncover the factors that lead to employee attrition and explore important questions such as ‘show me a breakdown of distance from home by job role and attrition’ or ‘compare average monthly income by education and attrition’. This is a fictional data set created by IBM data scientists.
# 
# Education :
#  'Below College' ,
#  'College' ,
#  'Bachelor' ,
#  'Master' ,
#  'Doctor'
# 
# EnvironmentSatisfaction :
#  'Low' ,
#  'Medium' ,
#  'High' ,
#  'Very High' 
# 
# JobInvolvement :
#  'Low' ,
#  'Medium' ,
#  'High' ,
#  'Very High'
# 
# JobSatisfaction :
#  'Low' ,
#  'Medium' ,
#  'High',
#  'Very High'
# 
# PerformanceRating
#  'Low' ,
#  'Good' ,
#  'Excellent' ,
#  'Outstanding'
# 
# RelationshipSatisfaction :
#  'Low' ,
#  'Medium' ,
#  'High' ,
#  'Very High'
# 
# WorkLifeBalance :
#  'Bad' ,
#  'Good' ,
#  'Better' ,
#  'Best'

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb
import sklearn 
import scipy
import statsmodels.api as sm
import warnings 
warnings.filterwarnings("ignore")


# # Reading dataset

# In[41]:


df = pd.read_csv('IBM HR Analytics Employee Attrition.csv')
df.head()


# In[42]:


df.shape


# # Data perprocessing & EDA

# In[43]:


def dataoveriew(df, message):
    print(f'{message}:\n')
    print("Rows:", df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nFeatures:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())


# In[44]:


dataoveriew(df, 'Overiew of the training dataset')


# In[45]:


#checking datatypes
df.info()


# # Dividing the columns into 2 categories (continuous and categorical)

# In[46]:


cat =[]
con=[]
for i in df.columns:
    if df[i].dtypes == "object":
        cat.append(i)
    else:
        con.append(i)


# In[47]:


df.describe().columns


# In[48]:


cat_df=df[['Attrition','BusinessTravel','Department','Over18','Gender','OverTime','JobRole','MaritalStatus','EducationField']]


# In[49]:


con_df=df[['Age', 'DailyRate', 'DistanceFromHome', 'Education','EmployeeNumber', 'EnvironmentSatisfaction',
           'HourlyRate','JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome','MonthlyRate', 'NumCompaniesWorked',
           'PercentSalaryHike','PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel',
           'TotalWorkingYears', 'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']]


# In[50]:


# check for missing values 
df.isnull().sum()


# In[51]:


# statistical measures of the dataset
df.describe()


# it seems that some of columns are not normalised 

# # visiualizing the data

# In[90]:


sb.distplot(df.DistanceFromHome)
plt.show()
df['DistanceFromHome_sqrt']=np.sqrt(df.DistanceFromHome)
sb.distplot(df.DistanceFromHome_sqrt)
plt.show()
df['DistanceFromHome_log']=np.log(df.DistanceFromHome)
sb.distplot(df.DistanceFromHome_log)
plt.show()


# In[91]:


sb.distplot(df.MonthlyIncome)
plt.show()
df['MonthlyIncome_sqrt']=np.sqrt(df.MonthlyIncome)
sb.distplot(df.MonthlyIncome_sqrt)
plt.show()
df['MonthlyIncome_log']=np.log(df.MonthlyIncome)
sb.distplot(df.MonthlyIncome_log)
plt.show()


# In[92]:


sb.distplot(df.PercentSalaryHike)
plt.show()
df['PercentSalaryHike_sqrt']=np.sqrt(df.PercentSalaryHike)
sb.distplot(df.PercentSalaryHike_sqrt)
plt.show()
df['PercentSalaryHike_log']=np.log(df.PercentSalaryHike)
sb.distplot(df.PercentSalaryHike_log)
plt.show()


# In[93]:


sb.distplot(df.YearsAtCompany)
plt.show()
df['YearsAtCompany_sqrt']=np.sqrt(df.YearsAtCompany)
sb.distplot(df.YearsAtCompany_sqrt)
plt.show()


# In[94]:


sb.distplot(df.YearsSinceLastPromotion)
plt.show()
df['YearsSinceLastPromotion_sqrt']=np.sqrt(df.YearsSinceLastPromotion)
sb.distplot(df.YearsSinceLastPromotion_sqrt)
plt.show()


# # correlation through heatmap

# In[57]:


corr_heatmap= con_df.corr()
f, ax = plt.subplots(figsize=(20,12))
sb.heatmap(corr_heatmap,vmax=0.8,annot=True)


# In[58]:


#show how mush % employees left the organizatin
df.Attrition.value_counts(normalize= True)


# In[59]:


Attrition= df.Attrition.value_counts()
sb.barplot(x=df.Attrition.index,y=df.Attrition.values)


# In[60]:


df['Attrition'].value_counts().plot(kind= "pie")


# most of them is no so this is class imbalanced problem

# In[61]:


df.OverTime.value_counts(normalize= True)


# In[62]:


OverTime= df.Attrition.value_counts()
sb.barplot(x=df.OverTime.index,y=df.OverTime.values)


# In[63]:


df['OverTime'].value_counts().plot(kind= "pie")


# In[64]:


#Bar plots
BarPlot_columns=['Age', 'DistanceFromHome','JobInvolvement','TotalWorkingYears',
'TrainingTimesLastYear','WorkLifeBalance', 'JobLevel' ,'TotalWorkingYears' , 'YearsInCurrentRole']


# In[65]:


#method for performing bar plots
def Bar_plots(var):
    col = pd.crosstab(df[var],df.Attrition)
    col.div(col.sum(1).astype(float),axis = 0).plot(kind = "bar",stacked= False , figsize=(8,4))
    plt.xticks(rotation=90)


# In[66]:


for col in BarPlot_columns:
    Bar_plots(col)


# # Insights :
# 1- attrition is very high wih employees between 18 : 22 years old
# 
# 2- attrition is more when distance of hte office is more from home
# 
# 3- attrition is high with employees's education in HR field
# 
# 4- employees wgo work overtime have high attrition than who didn't
# 
# 5- emploees who are working less than 2 years have more attrition

# # outliers checking and treatment

# In[67]:


import numpy as np
from scipy import stats


# In[68]:


z= np.abs(stats.zscore(df[['Age', 'DailyRate', 'DistanceFromHome', 'Education','EmployeeNumber', 'EnvironmentSatisfaction',
           'HourlyRate','JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome','MonthlyRate', 'NumCompaniesWorked',
           'PercentSalaryHike','PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel',
           'TotalWorkingYears', 'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']]))
print(z)
threshold = 3 
print(np.where(z>3))


# In[69]:


print(z[0][0])


# In[70]:


df


# # removing outliers

# In[71]:


df_out = df[(z<3).all(axis=1)]


# In[77]:


df_out1 =df_out.drop(['DistanceFromHome_sqrt','DistanceFromHome_log',
                      'MonthlyIncome_sqrt','MonthlyIncome_log',
                      'PercentSalaryHike_sqrt','PercentSalaryHike_log'
                      ,'YearsAtCompany_sqrt','YearsSinceLastPromotion_sqrt'],axis=1)


# In[78]:


df_out1.head()


# In[79]:


df_out1.shape


# dividing final dataset into categorical and continous variables

# In[80]:


numerical_df= df_out1.select_dtypes(include=np.number)
categorical_df= df_out1.select_dtypes(exclude=np.number)
numeric_cols =list (numerical_df.columns)
categorical_cols = list(categorical_df.columns)


# # converting categorical variables to binary

# In[81]:


categorical_df_dummies= pd.get_dummies(df_out1[categorical_cols],drop_first= True)
final_df= pd.concat([categorical_df_dummies,numerical_df],axis=1)
final_df.head()


# In[82]:


final_df.shape


# # Creating models:

# In[83]:


X=final_df[['BusinessTravel_Travel_Frequently','BusinessTravel_Travel_Rarely',
            'Department_Research & Development','Department_Sales',
            'EducationField_Life Sciences','EducationField_Marketing',
            'EducationField_Medical','EducationField_Other',
            'EducationField_Technical Degree','StockOptionLevel','TotalWorkingYears',
            'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']]
Y= final_df[['Attrition_Yes']]


# # train - test - split

# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=42)


# RandomForestClassifier Model

# In[95]:




from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,max_depth=4)
model=rfc.fit(xtrain,ytrain)
pred= model.predict(xtest)

from sklearn.metrics import confusion_matrix ,accuracy_score
print('the confusion matrix \n',confusion_matrix(ytest['Attrition_Yes'],pred))
print('the accuracy of the RandomForestClassifier is',accuracy_score(ytest['Attrition_Yes'],pred))


# In[85]:


model.feature_importances_


# In[86]:


import matplotlib.pyplot as plt
f, ax =plt.subplots(figsize=(10,12))
plt.barh(X.columns,model.feature_importances_)


# LogisticRegression Model

# In[96]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
model_lr=lr.fit(xtrain,ytrain)
pred= model_lr.predict(xtest)

from sklearn.metrics import confusion_matrix ,accuracy_score
print('the confusion matrix \n',confusion_matrix(ytest['Attrition_Yes'],pred))
print('the accuracy of the LogisticRegression is',accuracy_score(ytest['Attrition_Yes'],pred))


# DecisionTreeClassifier Model

# In[98]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=4)
model_dtc=dtc.fit(xtrain,ytrain)
pred= model_dtc.predict(xtest)

from sklearn.metrics import confusion_matrix ,accuracy_score
print('the confusion matrix \n',confusion_matrix(ytest['Attrition_Yes'],pred))
print('the accuracy of the DecisionTreeClassifier is',accuracy_score(ytest['Attrition_Yes'],pred))


# In[ ]:




