
# # **Capstone project: Providing data-driven suggestions for HR Department of Salifort Motors**

# ## Description and deliverables
#  

# 
# 
# The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?
# 
# Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.
# 
# If you can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

# 
# The dataset  contains 15,000 rows and 10 columns for the variables listed below. 
# 
# 
# Variable  |Description |
# -----|-----|
# satisfaction_level|Employee-reported job satisfaction level 
# last_evaluation|Score of employee's last performance review 
# number_project|Number of projects employee contributes to|
# average_monthly_hours|Average number of hours employee worked per month|
# time_spend_company|How long the employee has been with the company (years)
# Work_accident|Whether or not the employee experienced an accident while at work
# left|Whether or not the employee left the company
# promotion_last_5years|Whether or not the employee was promoted in the last 5 years
# Department|The employee's department
# salary|The employee's salary (U.S. dollars)


# 
# *  Who are your stakeholders for this project?
# 
# - What are you trying to solve or accomplish? 
# - What are your initial observations when you explore the data?- 
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.) - 
# - Do you have any ethical considerations in this stage? - 
# 
# 
# 
# -Stakeholders is the senior HR leadership for this project
# 
# -Trying to predict whether an employee leaves the company and identify the reasons to develop solution for retention
# 
# -there are 10 features and 15000 obserevations, only two features are string and rest are int, 
# 
# -Python, HR Dataset
# 
# -The considerations are for identifying the employees that they will leave but actually they will not (false positive). 
# company will loose resources for retaining them. further if employees are identified as staying when they will actually leave
#(false negative) will result in company loosing the financial cost of training these employees. 
# 

# 
# 

# ### Import packages

# Import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
import pickle



df0 = pd.read_csv("HR_capstone_dataset.csv")

# Display first few rows of the dataframe
df0.head(5)


# ## Step 2. Data Exploration (Initial EDA and data cleaning)
# 
# - Understand your variables
# - Clean your dataset (missing data, redundant data, outliers)

# Gather basic information about the data
df0.info()

# Gather descriptive statistics about the data

# Gather descriptive statistics about the data
df0.describe()


# Display all column names
df0.columns

# Rename columns as needed
df0=df0.rename(columns={'last_evaluation': 'last_performance_evaluation',
                   'number_project': 'projects_completed',
                   'time_spend_company': 'years_in_company',
                   'Work_accident': 'accident',
                   'left': 'left_company',
                   'promotion_last_5years': 'promoted_in_5years',
                   'Department': 'department',
                   'salary': 'salary_level'})

# Display all column names after the update
df0.columns

df0.head(5)

# Check for missing values
df0.isna().sum()


df0.shape

# Check for duplicates
df0.duplicated().value_counts()

# Inspect some rows containing duplicates as needed
df0[df0.duplicated()]

# Drop duplicates and save resulting dataframe in a new variable as needed
df_1=df0.drop_duplicates()

# Display first few rows of new dataframe as needed
df_1.head(5)


# Create a boxplot to visualize distribution of `tenure` and detect any outliers

fig=plt.figure(figsize=(8,4))
sns.boxplot(data=df_1,
           x='years_in_company')
plt.title('boxplot showing outliers in years spent in the company');

# Determine the number of rows containing outliers
outliers=df_1[df_1['years_in_company']>5]
outliers.shape
 

# ### Reflect on these questions as you complete the analyze stage.
# 
# - What did you observe about the relationships between variables?
# - What do you observe about the distributions in the data?
# - What transformations did you make with your data? Why did you chose to make those decisions?
# - What are some purposes of EDA before constructing a predictive model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?

# 
# - the variables are having random scatter plots. there is no relationship between any of the variables
# 
# - the satisfaction levels and the performance evaluation are right skewed. the total number of projects completed and the years 
# spent in comapny are left skewed. the average monthly hours worked is however unevenly distributed with maximum people working 
# around 150 hrs or 250 hours. 


# ## Step 2. Data Exploration (Continue EDA)

# Get numbers of people who left vs. stayed
print(df_1['left_company'].value_counts())

# Get percentages of people who left vs. stayed
print(df_1['left_company'].value_counts(normalize=True))


# ### Data visualizations

fig, axes = plt.subplots(1,4, figsize=(20, 5))

sns.histplot(data=df_1, x='satisfaction_level', multiple='stack', hue='left_company', ax=axes[0])

sns.histplot(data=df_1, x='last_performance_evaluation', multiple='stack', hue='left_company', ax=axes[1])

sns.histplot(data=df_1, x='average_montly_hours', multiple='stack', hue='left_company', ax=axes[2])

sns.histplot(data=df_1, x='projects_completed', multiple='stack', hue='left_company', ax=axes[3])


# The histograms gives the distribution of the data for four variables. 
# 1) The distribution of satisfaction level indicates that employees who have left the company are divided into three groups. Employees having satisfaction level 0.7 ~  1, 0.35 ~ 0.45, and 0.1. Need to check for reasons for this behaviour. 
# 2) The performance evaluation distribution has two groups, employees with evaluation of 0.75 ~ 1 and 0.45 ~ 0.55. Further, exploration of this needs to be explored.
# 3) Average monthly hours distribution also has two groups with 220  ~ 300 and 125 ~ 160. The two groups further requires exploration. 
# 4) The projects completed distribution indicates that employees who have completed 6~7 projects are leaving in higher proportions.


fig=plt.figure(figsize=(10,5))
sns.scatterplot(data=df_1,
               x='last_performance_evaluation',
               y='satisfaction_level', hue='left_company', size=0.05, alpha = 0.4)
plt.axvline(x=df_1['last_performance_evaluation'].mean(), color='red', ls= '--')
plt.title('scatter plot between last year performance evaluation and satisfaction level');


# The employees that are marked above 0.5 have satisfaction levels of 0.5 and above, 
# these employees are still working at the company. Further, a set of employees marked 0.75 to 0.99 have low satisfaction, 
# and marked 0.3 to 0.6 have moderate satisfaction level. A set of employees marked between 0.8 to 10 have higher satisfaction level 
# between 0.7 to 0.9, however, they have left the company. The mean of performance evaluation is slightly above 0.7. The employees 
# having higher performance evaluation and still lower satisfaction needs to be explored.


fig=plt.figure(figsize=(5,5))
sns.boxplot(data=df_1,
               x='projects_completed',
               y='satisfaction_level', hue='left_company')
plt.axhline(y=df_1['satisfaction_level'].mean(), color='red', ls= '--')
plt.title('scatter plot between projects completed and satisfaction level');

# The box between projects completed and satisfaction level of employees indicates that for 2 ~ 3 project completed, 
# employees have lower satisfaction levels. The level increases with completed projects 3 ~ 5, however, it drops when completed 
# projects are more than 5. For employees who completed 3 ~ 5 projects have higher satisfaction level, but they also includes who have 
# left the company. Further, employees who have completed 6 ~ 7 projects, have very low satisfaction levels. Low satisfaction levels may 
# be related to the employees leaving the company. The low satisfaction levels may be related to higher workloads. 
# Relationship between peojects completed and average monthly hours worked needs to be explored. 


sns.scatterplot(data=df_1,
               x='average_montly_hours',
               y='satisfaction_level', hue= 'left_company', size=0.05, alpha=0.2)
plt.title('scatter plot between average monthly hours worked and satisfaction level');

# The scatter plot indicates that employees working between 250 ~ 300 hrs have low satisfaction level. 
# There are two groups 125 ~ 175 hrs working and 220~280 hrs with moderate and high satisfation level respectively, 
# who have left the company. Working hours is a factor in employees leaving the company.

sns.scatterplot(data=df_1,
               x='average_montly_hours',
               y='last_performance_evaluation', hue='left_company', size=0.05, alpha=0.3)
plt.title('scatter plot between performance evaluation and average monthly hours');

# The scatter plot between monthly working hours and last performance evaluation indicates that higher working hours are related to 
# higher performance evaluations. Employees working around 220~300 hrs monthly are having higher ratings in performance evaluation.
# This indicates all employees who are working longer hours and have higher evauation are also leaving the company. 
# Reason for employees leaving who are working moderate hours and moderately marked in evaluation with moderate satisfaction level but 
# still leaving the company needs to be explored. 

sns.boxplot(data=df_1,
               x='projects_completed',
               y='average_montly_hours', hue='left_company')
plt.title('box plot between average monthly hours and project completed');

# The box plot indicates that higher number of prjects completed by the employees, the average number of hours they work in a month 
# also increases, the employees who have completed 6 - 7 projects have not stayed in the company. Workload and long working hours may 
# be related to employees leaving the company

sns.boxplot(data=df_1,
               x='promoted_in_5years',
               y='average_montly_hours', hue='left_company')
plt.title('box plot between employees promoted in last year and average monthly hours they work');

# To check whether promotion plays a part in employees leaving, box plot is created between employees promoted in last five years and 
# avaerage monthly hours they worked. It indicates that employees who have not been promoted have worked higher number of hours. 
# A set of employees who were promoted and have worked moderately, have also left the company 

sns.boxplot(data=df_1,
               x='promoted_in_5years',
               y='satisfaction_level', hue='left_company')
plt.title('box plot between employees promoted in last year and satisfaction level they work');

# Employees who were not promoted and left the company had a lower satisfaction level in the company. 
# These are the employees shown at the bottom level in scatter plots above. It also indicates that promotion plays a part in employees' 
# satisfaction level and hence may contribute to them leaving the company.  

sns.boxplot(data=df_1,
               x='promoted_in_5years',
               y='last_performance_evaluation', hue='left_company')
plt.title('box plot between employees promoted in last year and their last peformance evaluation');

# The scatter plot indicates that employees who have worked for for above 275 hours and not been promoted have left the company. 
# The promotion is a factor that may have lead to the employees leaving the company. 

sns.boxplot(data=df_1,
               x='years_in_company',
               y='satisfaction_level', hue='left_company')
plt.title('box plot between years in company and satisfaction level');

# The box plot between years spent in the company and satisfaction level indicates two groups that left the company. 
# Employees with less than three years have moderate satisfaction level and employees with 5 and 6 years in the company having above 
# moderate satisfaction level. Those with 7 and above years in the company have not left and have modertae satisfaction levels. 

sns.boxplot(data=df_1,
               x='years_in_company',
               y='last_performance_evaluation', hue='left_company')
plt.title('box plot between years spend in the company and last performance evaluation');

# The box plot again shows two grouos who left, with years in company 4 ~ 6 years have comparatively higher performance evaluation. 
# These employees have higher evaluation rating and high satisfaction, still left the company. The reason needs to be identified. 
# The second group with less than 3 years with moderate evaluation. 

sns.histplot(data=df_1, 
            x='years_in_company', hue='left_company', multiple='stack')
plt.title('histogram for years in company');

# Histogram shows the count of employees with yeras spent in the company indicates that max number of employees those who have left 
# have spent around 3 ~ 6 years in the company.

sns.boxplot(data=df_1,
               x='years_in_company',
               y='projects_completed', hue='left_company')
plt.title('boxplot graph between projects complated in company and years spent');

# The box plot indicates that two groups leaving the company, with around 2 years and 5 ~ 6 years spent in company. 
# They have completed around 4 ~ 5 projects in the company. 

df_left=df_1[df_1['left_company'] == 1]

sns.histplot(data=df_left, x='years_in_company', hue='promoted_in_5years', multiple='dodge');

# The histogram indicates that employees who have left the company are not promoted in last five years.
# only a few spending around three years have been promoted. Need to further investigate what is the exact number of these employees. 

df_left['promoted_in_5years'].value_counts()

# The number clearly indicates employees those who have not been promoted have left the company.

sns.histplot(data=df_1, x='department', hue='left_company', multiple='dodge');
plt.title('histogram showing distribution of employees in various departments');
plt.xticks(rotation = 45);

# The histogram indicates that number of employees who have left is almost similar across the departments, 
# with only slightly higher in Sales and Technical departments. 

sns.histplot(data=df_1, x='salary_level', hue='left_company', multiple='dodge');
plt.title('histogram showing distribution of employees according to salary distribution');
plt.xticks(rotation = 45);

# Histogram indicates that employees with low and medium salary levels are the one leaving in larger numbers.

ax=sns.histplot(data=df_1, x='salary_level', hue='promoted_in_5years', multiple='dodge');
plt.title('histogram showing distribution of employees according to salary distribution');
plt.xticks(rotation = 45);

sns.pairplot(df_1)

df_1['total_hours_worked']=df_1['average_montly_hours'] * df_1['years_in_company']*12
df_1['average_hours_per_project']=df_1['total_hours_worked']/ df_1['projects_completed']
df_1['projects_completed_per_year'] = df_1['projects_completed']/ df_1['years_in_company']

df_1.head(5)

sns.pairplot(df_1, vars=['satisfaction_level', 'last_performance_evaluation', 'projects_completed',
                        'average_montly_hours', 'years_in_company', 
                         'total_hours_worked', 'average_hours_per_project', 
                         'projects_completed_per_year'])

df_2=df_1.drop(columns=['total_hours_worked', 'average_hours_per_project', 
                         'projects_completed_per_year'], axis=1)


# The pair plots between initial variables does not shows any sever multicolinearity between them. 
# The assumption of multicolinearity has been met.

# - Determine which models are most appropriate
# - Construct the model
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data
# 

# 
# **Logistic Regression model assumptions**
# - Outcome variable is categorical
# - Observations are independent of each other
# - No severe multicollinearity among X variables
# - No extreme outliers
# - Linear relationship between each X variable and the logit of the outcome variable
# - Sufficiently large sample size

# ## Step 3. Model Building, Step 4. Results and Evaluation
# - Fit a model that predicts the outcome variable using two or more independent variables
# - Check model assumptions
# - Evaluate the model

# ### Identify the type of prediction task.

# ### Identify the types of models most appropriate for this task.

# ### Modeling

df_2=pd.get_dummies(df_2, columns=['department', 'salary_level'], drop_first=True)

df_2.head(5)

fig, axes = plt.subplots(1,3, figsize=(14, 4))

sns.boxplot(data=df_1, x='projects_completed', ax=axes[0])

sns.boxplot(data=df_1, x='average_montly_hours', ax=axes[1])

sns.boxplot(data=df_1, x='years_in_company', ax=axes[2])

plt.tight_layout()

# The box plots indicates the outliers in years spent in the company. 
# For model with logistic regression, outliers need to be removed.

sns.boxplot(df_2['years_in_company'])

df_3=df_2

median=np.median(df_3['years_in_company'] < 5)

df_3['years_in_company']=np.where(df_3['years_in_company'] > 5, median, df_3['years_in_company'])

sns.boxplot(df_3['years_in_company'])

# The dataset df_3 has been created for modeling using logistic regression after removing the outliers 

X=df_3.drop(columns=['left_company'], axis=1)

y=df_3['left_company']

sns.heatmap(df_3.corr())

# The heat map indicates not excessive correlation between the variables. 

df_3.corr()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify = y)

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train_scaled=std.fit(X_train)
X_train_scaled=X_train_scaled.transform(X_train)

clf=LogisticRegression(solver='liblinear')

clf.fit(X_train, y_train)

clf.coef_

clf.intercept_

y_clf_pred= clf.predict(X_test)

precision_clf = metrics.precision_score(y_test, y_clf_pred)
recall_clf = metrics.recall_score(y_test, y_clf_pred)
accuracy_clf = metrics.accuracy_score(y_test, y_clf_pred)
f1_clf = metrics.f1_score(y_test, y_clf_pred)

print('precision score', metrics.precision_score(y_test, y_clf_pred))
print('recall score', metrics.recall_score(y_test, y_clf_pred))
print('accuracy score', metrics.accuracy_score(y_test, y_clf_pred))
print('f1 score', metrics.f1_score(y_test, y_clf_pred))

cm=metrics.confusion_matrix(y_test, y_clf_pred, labels=clf.classes_)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

from sklearn.metrics import classification_report
target_labels=['staying', 'left']
print(classification_report(y_test, y_clf_pred, target_names=target_labels))

auc_clf=metrics.roc_auc_score(y_test, y_clf_pred)
auc_clf

def results(model_name, classifier):
    y_pred=classifier.predict(X_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc=metrics.roc_auc_score(y_test, y_pred)

    results=pd.DataFrame({'model name' : [model_name],
                         'precision': [precision],
                         'recall': [recall],
                         'accuracy': [accuracy],
                          'f1': [f1],
                         'auc score': [auc]})
    return results


result_clf = results('LR', clf)

result_clf

# naive Bayes model
gnb_scaled=GaussianNB()
gnb_scaled.fit(X_train, y_train)

y_gnb_pred=gnb_scaled.predict(X_test)

print('precision score gnb', metrics.precision_score(y_test, y_gnb_pred))
print('recall score gnb', metrics.recall_score(y_test, y_gnb_pred))
print('accuracy score gnb', metrics.accuracy_score(y_test, y_gnb_pred))
print('f1 score gnb', metrics.f1_score(y_test, y_gnb_pred))

cm=metrics.confusion_matrix(y_test, y_gnb_pred, labels=gnb_scaled.classes_)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb_scaled.classes_)
disp.plot()

print(classification_report(y_test, y_gnb_pred, target_names=target_labels))
metrics.roc_auc_score(y_test, y_gnb_pred)

results_gnb = results('GNB', gnb_scaled)
results_gnb

def final_results(result_1, result_2):
    final=pd.concat([result_1, result_2], axis=0)
    return final

Result_Final=final_results(result_clf, results_gnb)
Result_Final

tree_params={'max_depth':[3,4,5,6],
            'min_samples_leaf': [5,6,7,8] }

scoring=['accuracy', 'precision', 'f1', 'recall']

# tuned decision tree model
tuned_decision_tree=DecisionTreeClassifier()

dt_clf=GridSearchCV(tuned_decision_tree, tree_params, scoring=scoring, cv=4, refit='f1')

get_ipython().run_cell_magic('time', '', 'dt_clf.fit(X_train, y_train)')

dt_clf.best_params_

dt_clf.best_estimator_

y_dt_pred=dt_clf.best_estimator_.predict(X_test)

print('precision score dt', metrics.precision_score(y_test, y_dt_pred))
print('recall score dt', metrics.recall_score(y_test, y_dt_pred))
print('accuracy score dt', metrics.accuracy_score(y_test, y_dt_pred))
print('f1 score dt', metrics.f1_score(y_test, y_dt_pred))

cm=metrics.confusion_matrix(y_test, y_dt_pred, labels=dt_clf.classes_)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_clf.classes_)
disp.plot()

print(classification_report(y_test, y_dt_pred, target_names=target_labels))

results_dt = results('DT', dt_clf)
results_dt

Result_Final_1=final_results(Result_Final, results_dt)
Result_Final_1.reset_index()

# random forest model 
rf=RandomForestClassifier(random_state=42)

rf_params={'max_depth': [2,3,4,5],
          'min_samples_leaf': [1,2,3],
          'min_samples_split': [3,4,5],
          'max_features': [0.7, 0.8],
          'n_estimators': [100, 125]}

rf_clf=GridSearchCV(rf, rf_params, scoring=scoring, cv=4, refit='f1')

get_ipython().run_cell_magic('time', '', 'rf_clf.fit(X_train, y_train)')

rf_clf.best_params_

y_rf_pred=rf_clf.best_estimator_.predict(X_test)

print('precision score rf', metrics.precision_score(y_test, y_rf_pred))
print('recall score rf', metrics.recall_score(y_test, y_rf_pred))
print('accuracy score rf', metrics.accuracy_score(y_test, y_rf_pred))
print('f1 score rf', metrics.f1_score(y_test, y_rf_pred))

cm=metrics.confusion_matrix(y_test, y_rf_pred, labels=rf_clf.classes_)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_clf.classes_)
disp.plot()

print(classification_report(y_test, y_rf_pred, target_names=target_labels))

results_RF = results('RF', rf_clf)
results_RF

Result_Final_2=final_results(Result_Final_1, results_RF)
Result_Final_2.reset_index(drop=True)

xgb=XGBClassifier()

xgb_params={'max_depth': [4],
           'min_child_weight': [1],
           'learning_rate':[0.2],
           'n_estimators': [100]}

xgb_clf=GridSearchCV(xgb, xgb_params, scoring=scoring, cv=4, refit='f1')

get_ipython().run_cell_magic('time', '', 'xgb_clf.fit(X_train, y_train)')

y_xgb_pred=xgb_clf.best_estimator_.predict(X_test)

print('precision score xgb', metrics.precision_score(y_test, y_xgb_pred))
print('recall score xgb', metrics.recall_score(y_test, y_xgb_pred))
print('accuracy score xgb', metrics.accuracy_score(y_test, y_xgb_pred))
print('f1 score xgb', metrics.f1_score(y_test, y_xgb_pred))

cm=metrics.confusion_matrix(y_test, y_xgb_pred, labels=xgb_clf.classes_)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_clf.classes_)
disp.plot()

print(classification_report(y_test, y_xgb_pred, target_names=target_labels))

print('auc score lr', metrics.roc_auc_score(y_test, y_clf_pred))
print('auc score gnb', metrics.roc_auc_score(y_test, y_gnb_pred))
print('auc score dt', metrics.roc_auc_score(y_test, y_dt_pred))
print('auc score rf', metrics.roc_auc_score(y_test, y_rf_pred))
print('auc score xgb', metrics.roc_auc_score(y_test, y_xgb_pred))

results_xgb = results('XGB', xgb_clf)
results_xgb

Result_Final_3=final_results(Result_Final_2, results_xgb)
Result_Final_3.reset_index(drop=True)

from xgboost import plot_importance
plot_importance(xgb_clf.best_estimator_);
