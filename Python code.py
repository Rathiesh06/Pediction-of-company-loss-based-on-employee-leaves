#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from scipy import stats
import random
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn import preprocessing
from statsmodels.formula.api import ols
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn import metrics
from numpy import cov
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# In[71]:


os.chdir(r"C:\Users\Compusoft\Desktop\Data Scientist\Project\employee-absenteeism-master\employee-absenteeism-master")


# In[72]:


os.getcwd()


# In[73]:


absent = pd.read_excel("Absenteeism_at_work_Project.xls")


# In[74]:


absent.shape


# In[75]:


absent['Absenteeism time in hours'].value_counts()


# In[76]:


absent.dtypes


# In[77]:


# Missing Value Analysis
missing_val = pd.DataFrame(absent.isnull().sum())


# In[78]:


missing_val = missing_val.reset_index()


# In[79]:


missing_val = missing_val.rename(columns = {'index':'variables',0:'Missing_Percentage'})


# In[80]:


missing_val['Missing_Percentage'] = (missing_val['Missing_Percentage']/len(absent))*100


# In[81]:


missing_val


# In[82]:


# Imputing missing values with help of mean and median
absent['Reason for absence'] = absent['Reason for absence'].fillna(absent['Reason for absence'].median())
absent['Month of absence'] = absent['Month of absence'].fillna(absent['Month of absence'].median())
absent['Transportation expense'] = absent['Transportation expense'].fillna(absent['Transportation expense'].median())
absent['Distance from Residence to Work'] = absent['Distance from Residence to Work'].fillna(absent['Distance from Residence to Work'].median())
absent['Service time'] = absent['Service time'].fillna(absent['Service time'].median())
absent['Service time'] = absent['Service time'].fillna(absent['Service time'].median())
absent['Age'] = absent['Age'].fillna(absent['Age'].median())
absent['Work load Average/day '] = absent['Work load Average/day '].fillna(absent['Work load Average/day '].median())
absent['Hit target'] = absent['Hit target'].fillna(absent['Hit target'].median())
absent['Disciplinary failure'] = absent['Disciplinary failure'].fillna(absent['Disciplinary failure'].median())
absent['Education'] = absent['Education'].fillna(absent['Education'].median())
absent['Social drinker'] = absent['Social drinker'].fillna(absent['Social drinker'].median())
absent['Social smoker'] = absent['Social smoker'].fillna(absent['Social smoker'].median())
absent['Son'] = absent['Son'].fillna(absent['Son'].median())
absent['Pet'] = absent['Pet'].fillna(absent['Pet'].median())
absent['Height'] = absent['Height'].fillna(absent['Height'].median())
absent['Weight'] = absent['Weight'].fillna(absent['Weight'].median())
absent['Body mass index'] = absent['Body mass index'].fillna(absent['Body mass index'].mean())
absent['Absenteeism time in hours'] = absent['Absenteeism time in hours'].fillna(absent['Absenteeism time in hours'].median())


# In[83]:


absent.isnull().sum()


# In[84]:


data = absent.copy()


# In[85]:


absent['ID'] = absent['ID'].astype('category')
absent['Reason for absence'] = absent['Reason for absence'].astype('category')
absent['Month of absence'] = absent['Month of absence'].astype('category')
absent['Day of the week'] = absent['Day of the week'].astype('category')
absent['Seasons'] = absent['Seasons'].astype('category')
absent['Disciplinary failure'] = absent['Disciplinary failure'].astype('category')
absent['Education'] = absent['Education'].astype('category')
absent['Social drinker'] = absent['Social drinker'].astype('category')
absent['Social smoker'] = absent['Social smoker'].astype('category')


# In[86]:


# outlier analysis
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(absent['Transportation expense'])


# In[87]:


plt.boxplot(absent['Distance from Residence to Work'])


# In[88]:


plt.boxplot(absent['Service time'])


# In[89]:


plt.boxplot(absent['Age'])


# In[90]:


plt.boxplot(absent['Work load Average/day '])


# In[ ]:





# In[91]:


plt.boxplot(absent['Hit target'])


# In[92]:


plt.boxplot(absent['Son'])


# In[93]:


plt.boxplot(absent['Pet'])


# In[94]:


plt.boxplot(absent['Weight'])


# In[95]:


plt.boxplot(absent['Height'])


# In[96]:


plt.boxplot(absent['Body mass index'])


# In[97]:


plt.boxplot(absent['Absenteeism time in hours'])


# In[100]:


for i in absent :
    print(i)
    q75,q25 = np.percentile(absent.loc[:,i],[75,25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    print(min)
    print(max)
    
    absent.loc[absent[i]< min,:i] = np.nan
    absent.loc[absent[i]> max,:i] = np.nan


# In[99]:


# calculating minimum and maximum values
q75,q25 = np.percentile(absent['Transportation expense'],[75,25])

iqr = q75 - q25

minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)
print(minimum)
print(maximum)

absent.loc[absent['Transportation expense']< minimum,:'Transportation expense'] = np.nan
absent.loc[absent['Transportation expense']> maximum,:'Transportation expense'] = np.nan


# In[63]:


q75,q25 = np.percentile(absent['Age'],[75,25])

iqr = q75 - q25

minimum2 = q25 - (iqr*1.5)
maximum2 = q75 + (iqr*1.5)
print(minimum2)
print(maximum2)

absent.loc[absent['Age']< minimum2,:'Age'] = np.nan
absent.loc[absent['Age']> maximum2,:'Age'] = np.nan


# In[64]:


q75,q25 = np.percentile(absent['Service time'],[75,25])

iqr = q75 - q25

minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)
print(minimum)
print(maximum)

absent.loc[absent['Service time']< minimum,:'Service time'] = np.nan
absent.loc[absent['Service time']> maximum,:'Service time'] = np.nan


# In[65]:


q75,q25 = np.percentile(absent['Work load Average/day '],[75,25])

iqr = q75 - q25

minimum3 = q25 - (iqr*1.5)
maximum3 = q75 + (iqr*1.5)
print(minimum3)
print(maximum3)

absent.loc[absent['Work load Average/day ']< minimum3,:'Work load Average/day '] = np.nan
absent.loc[absent['Work load Average/day ']> maximum3,:'Work load Average/day '] = np.nan


# In[66]:


q75,q25 = np.percentile(absent['Hit target'],[75,25])

iqr = q75 - q25

minimum4 = q25 - (iqr*1.5)
maximum4 = q75 + (iqr*1.5)
print(minimum4)
print(maximum4)

absent.loc[absent['Hit target']< minimum4,:'Hit target'] = np.nan
absent.loc[absent['Hit target']> maximum4,:'Hit target'] = np.nan


# In[67]:


q75,q25 = np.percentile(absent['Pet'],[75,25])

iqr = q75 - q25

minimum6 = q25 - (iqr*1.5)
maximum6 = q75 + (iqr*1.5)
print(minimum6)
print(maximum6)

absent.loc[absent['Pet']< minimum6,:'Pet'] = np.nan
absent.loc[absent['Pet']> maximum6,:'Pet'] = np.nan


# In[68]:


q75,q25 = np.percentile(absent['Height'],[75,25])

iqr = q75 - q25

minimum8 = q25 - (iqr*1.5)
maximum8 = q75 + (iqr*1.5)
print(minimum8)
print(maximum8)

absent.loc[absent['Height']< minimum8,:'Height'] = np.nan
absent.loc[absent['Height']> maximum8,:'Height'] = np.nan


# In[101]:


# imputing outliers values with median

absent['Transportation expense'] = absent['Transportation expense'].fillna(absent['Transportation expense'].median())
absent['Age'] = absent['Age'].fillna(absent['Age'].median())
absent['Work load Average/day '] = absent['Work load Average/day '].fillna(absent['Work load Average/day '].median())
absent['Hit target'] = absent['Hit target'].fillna(absent['Hit target'].median())
absent['Service time'] = absent['Service time'].fillna(absent['Service time'].median())
absent['Pet'] = absent['Pet'].fillna(absent['Pet'].median())
absent['Height'] = absent['Height'].fillna(absent['Height'].median())
absent['Absenteeism time in hours'] = absent['Absenteeism time in hours'].fillna(absent['Absenteeism time in hours'].median())


# In[102]:


# Copying data in new object "data"

absent['ID'] = data['ID']
absent['Reason for absence'] = data['Reason for absence']
absent['Month of absence'] = data['Month of absence']
absent['Day of the week'] = data['Day of the week']
absent['Seasons'] = data['Seasons']
absent['Distance from Residence to Work'] = data['Distance from Residence to Work']
absent['Disciplinary failure'] = data['Disciplinary failure']
absent['Education'] = data['Education']
absent['Son'] = data['Son']
absent['Social drinker'] = data['Social drinker']
absent['Social smoker'] = data['Social smoker']
absent['Weight'] = data['Weight']
absent['Body mass index'] = data ['Body mass index']


# In[103]:


# checking missing values after outlier analysis

missval = pd.DataFrame(absent.isnull().sum())

missval


# In[104]:


absent['ID'] = absent['ID'].astype('category')
absent['Reason for absence'] = absent['Reason for absence'].astype('category')
absent['Month of absence'] = absent['Month of absence'].astype('category')
absent['Day of the week'] = absent['Day of the week'].astype('category')
absent['Seasons'] = absent['Seasons'].astype('category')
absent['Disciplinary failure'] = absent['Disciplinary failure'].astype('category')
absent['Education'] = absent['Education'].astype('category')
absent['Social drinker'] = absent['Social drinker'].astype('category')
absent['Social smoker'] = absent['Social smoker'].astype('category')


# In[105]:


# feature selection
numeric_c = absent[['Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target',
     'Son', 'Pet', 'Weight', 'Height', 'Body mass index','Absenteeism time in hours']]


# In[106]:


# Feature selection
corr = numeric_c.corr()


# In[107]:


f,ax = plt.subplots(figsize = (10,8))
sns.heatmap(corr,mask = np.zeros_like(corr,dtype = np.object),cmap = sns.diverging_palette(220,10,as_cmap = True),square = True, ax=ax,annot = True)


# In[108]:


# anova for categorical variable
factor = absent[['ID', 'Reason for absence', 'Month of absence', 'Day of the week','Seasons', 'Disciplinary failure', 'Education', 'Social drinker',
       'Social smoker',]]


# In[109]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Reason for absence"]))


# In[110]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Month of absence"]))


# In[111]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Day of the week"]))


# In[112]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Seasons"]))


# In[113]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Disciplinary failure"]))


# In[114]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Education"]))


# In[115]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Social drinker"]))


# In[116]:


print(stats.f_oneway(absent["Absenteeism time in hours"],absent["Social smoker"]))


# In[117]:


data = absent.copy()


# In[118]:


absent = absent.drop(['ID','Seasons','Education','Height','Hit target','Pet','Body mass index','Disciplinary failure','Age','Social smoker','Social drinker','Son'],axis = 1)


# In[119]:


absent.shape


# In[120]:


# DAta normalisation
#Normality check
absent['Transportation expense'].hist(bins = 20)


# In[121]:


absent['Distance from Residence to Work'].hist(bins = 20)


# In[122]:


absent['Service time'].hist(bins = 20)


# In[123]:


absent[ 'Work load Average/day '].hist(bins = 20)


# In[124]:


absent['Weight'].hist(bins = 20)


# In[125]:


# Data Normalisation
from sklearn.preprocessing import normalize
normalized_absent = preprocessing.normalize(absent)


# In[126]:


absent.dtypes


# In[127]:


# ML Algorithm
## dividing data into train and test
train,test = train_test_split(absent,test_size= 0.2)


# In[128]:


# Decision Tree Regression
random.seed(123)
fit = DecisionTreeRegressor(max_depth = 2).fit(train.iloc[:,0:8],train.iloc[:,8])


# In[129]:


predictions_dt = fit.predict(test.iloc[:,0:8])


# In[130]:


mse_dt = (mean_squared_error(test.iloc[:,8], predictions_dt))
print(mse_dt)


# In[131]:


rmse_dt = sqrt(mean_squared_error(test.iloc[:,8],predictions_dt))
print(rmse_dt)


# In[132]:


# Random forest
# n = 100
random.seed(123)
rfregressor100 = RandomForestRegressor(n_estimators = 100, random_state = 0)
rfregressor100.fit(train.iloc[:,0:8],train.iloc[:,8])


# In[133]:


predictions_rf100 = rfregressor100.predict(test.iloc[:,0:8])


# In[134]:


mse_rf100 = (mean_squared_error(test.iloc[:,8], predictions_rf100))
print(mse_rf100)


# In[135]:


rmse_rf100 = sqrt(mean_squared_error(test.iloc[:,8],predictions_rf100))
print(rmse_rf100)


# In[136]:


# Random forest for n = 200
random.seed(123)
rfregressor200 = RandomForestRegressor(n_estimators = 200, random_state = 0)
rfregressor200.fit(train.iloc[:,0:8],train.iloc[:,8])


# In[137]:


predictions_rf200 = rfregressor200.predict(test.iloc[:,0:8])


# In[138]:


mse_rf200 = (mean_squared_error(test.iloc[:,8], predictions_rf200))
print(mse_rf200)


# In[139]:


rmse_rf200 = sqrt(mean_squared_error(test.iloc[:,8],predictions_rf200))
print(rmse_rf200)


# In[140]:


# Random forest for n = 300

rfregressor300 = RandomForestRegressor(n_estimators = 300, random_state = 0)
rfregressor300.fit(train.iloc[:,0:8],train.iloc[:,8])


# In[141]:


predictions_rf300 = rfregressor300.predict(test.iloc[:,0:8])


# In[142]:


mse_rf300 = (mean_squared_error(test.iloc[:,8], predictions_rf300))
print(mse_rf300)


# In[143]:


rmse_rf300 = sqrt(mean_squared_error(test.iloc[:,8],predictions_rf300))
print(rmse_rf300)


# In[144]:


# Random forest for n = 500

rfregressor500 = RandomForestRegressor(n_estimators = 500, random_state = 0)
rfregressor500.fit(train.iloc[:,0:8],train.iloc[:,8])


# In[145]:


predictions_rf500 = rfregressor500.predict(test.iloc[:,0:8])


# In[146]:


mse_rf500 = (mean_squared_error(test.iloc[:,8], predictions_rf500))
print(mse_rf500)


# In[147]:


rmse_rf500 = sqrt(mean_squared_error(test.iloc[:,8],predictions_rf500))
print(rmse_rf500)


# In[148]:


# Linear regression 

absent['Reason for absence'] = absent['Reason for absence'].astype('float')
absent['Day of the week'] = absent['Day of the week'].astype('float')
absent['Month of absence'] = absent['Month of absence'].astype('float')


# In[149]:


train1,test1 = train_test_split(absent,test_size = 0.2)


# In[150]:


line_regression = sm.OLS(train1.iloc[:,8],train1.iloc[:,0:8]).fit()


# In[151]:


line_regression.summary()


# In[152]:


predictions_lr = line_regression.predict(test1.iloc[:,0:8])


# In[153]:


mse_lr = (mean_squared_error(test.iloc[:,8], predictions_lr))
print(mse_lr)


# In[154]:


rmse_linear = sqrt(mean_squared_error(test1.iloc[:,8],predictions_lr))
print(rmse_linear)


# In[155]:


## LOSS per month
data.shape


# In[156]:


loss = data[['Month of absence','Service time','Work load Average/day ','Absenteeism time in hours']]


# In[157]:


loss["loss_month"] = (loss['Work load Average/day ']*loss['Absenteeism time in hours'])/loss['Service time']


# In[158]:


loss.shape
loss.head(5)


# In[159]:


loss["loss_month"] = np.round(loss["loss_month"]).astype('int64')


# In[160]:


No_absent = loss[loss['Month of absence'] == 0]['loss_month'].sum()
January = loss[loss['Month of absence'] == 1]['loss_month'].sum()
February = loss[loss['Month of absence'] == 2]['loss_month'].sum()
March = loss[loss['Month of absence'] == 3]['loss_month'].sum()
April = loss[loss['Month of absence'] == 4]['loss_month'].sum()
May = loss[loss['Month of absence'] == 5]['loss_month'].sum()
June = loss[loss['Month of absence'] == 6]['loss_month'].sum()
July = loss[loss['Month of absence'] == 7]['loss_month'].sum()
August = loss[loss['Month of absence'] == 8]['loss_month'].sum()
September = loss[loss['Month of absence'] == 9]['loss_month'].sum()
October = loss[loss['Month of absence'] == 10]['loss_month'].sum()
November = loss[loss['Month of absence'] == 11]['loss_month'].sum()
December = loss[loss['Month of absence'] == 12]['loss_month'].sum()


# In[161]:


loss.head(5)


# In[162]:


data1 = {'No Absent': No_absent, 'Janaury': January,'Febraury': February,'March': March,
       'April': April, 'May': May,'June': June,'July': July,
       'August': August,'September': September,'October': October,'November': November,
       'December': December}


# In[163]:


workloss = pd.DataFrame.from_dict(data1,orient = 'index')


# In[164]:


workloss.rename(index = str, columns={0:"Workload loss pr month"})


# In[ ]:




