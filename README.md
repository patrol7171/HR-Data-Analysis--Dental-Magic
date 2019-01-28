
***
# <font color=limegreen>**Dental Magic, Inc**</font> 
***

## HR Employee Data Analysis Project - Part I

### *Research Questions:*
-  **What is the overall diversity profile of the current organization?**
-  **What factors appear to contribute to employee turnover?**
-  **What are our best recruiting sources if we want to ensure a diverse organization?**
-  **Is there any relationship between who a person works for and their performance score?**

### *Observed Trends:* 

1. Our analysis suggests that there appears to be somewhat of an equal distribution of men and women amongst the staff (more women tip the scales then men, but by a small margin with the exception of the Production department). However, it's clear more racial diversity is needed as our analysis indicates most of the staff is caucasian. The most abundant age range is between 31 and 39.  The marital status  of most employees are either single or married.

1. Based on our analysis, the majority of all terminations were voluntary.  We can also conclude that the top 3 reasons for employee termination are:
    1) Found another position; 
    2) Unhappy; 
    3) More money offered somewhere else

1. It appears that, in 2018, a significant amount of funds went toward employment sources that promote diversity.  However, we can also conclude that the most prevalent employment source for "non-white" members of the staff was the Diversity Job Fair.

1. There were very few managers that gave more low performance scores when compared to all members of management, but further research will be needed to verify if these scores are really justifiable. 

<br>
### *Modules and Data Source Setup*


```python
import seaborn as sns
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn import metrics
import pandas as pd
import numpy as np
import math
import types
import colorsys
from scipy import stats
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt  
matplotlib.style.use('ggplot')
plt.style.use('seaborn-talk')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import mpld3
from mpld3 import plugins
mpld3.enable_notebook()
import pylab
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)
%matplotlib inline
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy import func
import mysql.connector
from config import mysql_pswd
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



```python
#Define colors
from matplotlib import cm
current_palette = sns.color_palette("muted", n_colors=30)
cmap1 = cm.get_cmap('gist_rainbow')
cmap2 = cm.get_cmap('rainbow')
cs1 = cm.Dark2(np.arange(40))
cs2 = cm.Paired(np.arange(40))
```


```python
PASSWORD = mysql_pswd
engine = create_engine('mysql://root:'+PASSWORD+'@localhost/dental_magic_hr_db')
Base = automap_base()
Base.prepare(engine, reflect=True)
Base.classes.keys()
```




    ['employee_data', 'production_staff', 'recruiting_costs', 'salary_grid']




```python
Employees = Base.classes.employee_data
session = Session(engine)
```


```python
sql = "Select * from employee_data"
df_all = pd.read_sql_query(sql, session.bind)
df_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 310 entries, 0 to 309
    Data columns (total 30 columns):
    ID                  310 non-null int64
    LastName            310 non-null object
    FirstName           310 non-null object
    EmployeeNumber      310 non-null int64
    MarriedID           310 non-null int64
    MaritalStatusID     310 non-null int64
    GenderID            310 non-null int64
    EmpStatusID         310 non-null int64
    DeptID              310 non-null int64
    PerfScoreID         310 non-null int64
    Age                 310 non-null int64
    PayRate             310 non-null float64
    State               310 non-null object
    Zip                 310 non-null int64
    DOB                 310 non-null object
    Sex                 310 non-null object
    MaritalDesc         310 non-null object
    CitizenDesc         310 non-null object
    Hispanic_Latino     310 non-null object
    RaceDesc            310 non-null object
    HireDate            310 non-null object
    DaysEmployed        310 non-null int64
    TerminationDate     310 non-null object
    ReasonForTerm       310 non-null object
    EmploymentStatus    310 non-null object
    Department          310 non-null object
    Position            310 non-null object
    ManagerName         310 non-null object
    EmployeeSource      310 non-null object
    PerformanceScore    310 non-null object
    dtypes: float64(1), int64(11), object(18)
    memory usage: 72.7+ KB
    


```python
sql2 = "Select * from recruiting_costs"
df_recruit_costs = pd.read_sql_query(sql2, session.bind)
df_recruit_costs.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22 entries, 0 to 21
    Data columns (total 14 columns):
    ID                  22 non-null int64
    EmploymentSource    22 non-null object
    January_2018        22 non-null int64
    February_2018       22 non-null int64
    March_2018          22 non-null int64
    April_2018          22 non-null int64
    May_2018            22 non-null int64
    June_2018           22 non-null int64
    July_2018           22 non-null int64
    August_2018         22 non-null int64
    September_2018      22 non-null int64
    October_2018        22 non-null int64
    November_2018       22 non-null int64
    December_2018       22 non-null int64
    dtypes: int64(13), object(1)
    memory usage: 2.5+ KB
    

<br>
<br>
### Current Employee Demographics and Diversity Profile


```python
employment_status = df_all.groupby(by='EmploymentStatus').size().sort_values(ascending=False).head(10)
employment_status
```




    EmploymentStatus
    Active                    183
    Voluntarily Terminated     88
    Terminated for Cause       14
    Leave of Absence           14
    Future Start               11
    dtype: int64




```python
employed_total=df_all['ReasonForTerm'].str.contains('Not applicable').sum()
termed_total=(employment_status.sum())-employed_total

data={'Currently Employed':employed_total, 'Terminated':termed_total}
employment_status2 = pd.Series(data,index=['Currently Employed','Terminated'])
employment_status2.index.name = 'EmploymentStatus2'
employment_status2
```




    EmploymentStatus2
    Currently Employed    208
    Terminated            102
    dtype: int64




```python
group_names=['CURRENTLY EMPLOYED', 'TERMINATED']
group_size=[208,102]
subgroup_names=['Leave of Absence', 'Future Start', 'Active', 'Terminated for Cause', 'Voluntarily Terminated']
subgroup_size=[14,11,183,14,88]
# Create colors
a, b = [plt.cm.Greens, plt.cm.Oranges]
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=2.6, labels=group_names, colors=[a(0.6), b(0.6)], textprops=dict(color="gray"))
plt.setp(mypie, width=0.6, edgecolor='white')
# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=2.6-0.6, labels=subgroup_names, labeldistance=0.25, rotatelabels=True,
                   colors=[a(0.4), a(0.3), a(0.2), b(0.4), b(0.3)])
plt.setp(mypie2, width=1.5, edgecolor='white')
plt.margins(0,0)
plt.show()
```


![png](Images/output_14_0.png)



```python
pie1Labels = list(employment_status2.index)
pie1Sizes = list(employment_status2.values)
pie2Labels = list(employment_status.index)
pie2Sizes = list(employment_status.values)

fig = plt.figure(figsize=(15,7))
ax = plt.subplot2grid((1,2),(0,0),rowspan=1,colspan=1)
plt.pie(pie1Sizes,labels=None,autopct='%1.1f%%',colors=cs1)
plt.title('Employee Standings For All On Record')
plt.legend(labels=pie1Labels, bbox_to_anchor=(0.4,0.75), loc="center left", fontsize=12, bbox_transform=plt.gcf().transFigure)

ax = plt.subplot2grid((1,2),(0,1),rowspan=1,colspan=1)
plt.pie(pie2Sizes,labels=None,autopct='%1.1f%%',colors=cs2)
plt.title('Employment Status - All Employees On File')
plt.legend(labels=pie2Labels, bbox_to_anchor=(1.03,0.75), loc="center right", fontsize=12, bbox_transform=plt.gcf().transFigure)
plt.show()
```


![png](Images/output_15_0.png)



```python
df_active = df_all[~df_all['EmploymentStatus'].str.contains('Terminated for Cause|Voluntarily Terminated')]
df_active['Department'].value_counts()[0:20].plot(kind='barh',figsize=(10,8), title='Current Staff Count Per Dept')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bb724b1668>




![png](Images/output_16_1.png)



```python
df_active['Department'].value_counts()[0:20].plot(kind='pie',figsize=(8,8),title='Current Staff Percentages By Dept',
                                                  autopct='%1.1f%%', label='')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bb725a7a90>




![png](Images/output_17_1.png)



```python
plt.figure(figsize=(16,8))
sns.countplot('Position', data = df_active)
plt.xticks(rotation = 60, ha='right')
plt.title('Current Staff Count Per Position')
plt.tight_layout()
```


![png](Images/output_18_0.png)



```python
locTbl = pd.crosstab(index=df_active["Department"], columns=df_active["State"])
locTbl.plot(kind="barh",figsize=(15,7),stacked=True,colormap=cmap1,title='Staff Location By Dept').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb72985d68>




![png](Images/output_19_1.png)



```python
g1 = sns.factorplot("RaceDesc", data=df_active, aspect=4, kind="count")
g1.set_xticklabels(rotation=0)
g1 = plt.title("Staff Racial Distribution")
```


![png](Images/output_20_0.png)



```python
df_active['RaceDesc'].value_counts()[0:20].plot(kind='pie',figsize=(8,8), title='Staff Percentage Per Race',
                                                autopct='%1.1f%%', label='', colormap=cmap2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bb730fe8d0>




![png](Images/output_21_1.png)



```python
sns.countplot(df_active["Sex"])
g3 = plt.title("Staff Gender Counts")
```


![png](Images/output_22_0.png)



```python
g2 = sns.factorplot("Age", data=df_active, aspect=4, kind="count")
g2.set_xticklabels(rotation=0)
g2 = plt.title("Distribution of Ages Among Staff")
```


![png](Images/output_23_0.png)



```python
dept_df = df_active.copy()
dept_df.drop(['DeptID', 'EmpStatusID', 'EmployeeNumber', 'GenderID',
              'ID', 'MarriedID', 'MaritalStatusID', 'Zip', 'PerfScoreID'], axis=1, inplace=True)
pd.pivot_table(dept_df,index=["Department"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DaysEmployed</th>
      <th>PayRate</th>
    </tr>
    <tr>
      <th>Department</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Admin Offices</th>
      <td>32.500000</td>
      <td>1480.125000</td>
      <td>30.432500</td>
    </tr>
    <tr>
      <th>Executive Office</th>
      <td>63.000000</td>
      <td>1973.000000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>IT - Information Systems</th>
      <td>37.600000</td>
      <td>1120.050000</td>
      <td>46.104500</td>
    </tr>
    <tr>
      <th>Production</th>
      <td>38.680000</td>
      <td>1649.808000</td>
      <td>23.473600</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>38.333333</td>
      <td>1723.222222</td>
      <td>55.407407</td>
    </tr>
    <tr>
      <th>Software Engineering</th>
      <td>36.428571</td>
      <td>1656.000000</td>
      <td>48.640000</td>
    </tr>
  </tbody>
</table>
</div>




```python
table1 = pd.crosstab(index=df_active["Department"], columns=df_active["MaritalDesc"])
table1.plot(kind="barh",figsize=(10,5),stacked=True,title='Marital Status Distribution By Dept').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb73d20860>




![png](Images/output_25_1.png)



```python
table2 = pd.crosstab(index=df_active["Department"], columns=df_active["RaceDesc"])
table2.plot(kind="barh", figsize=(10,5), stacked=True, title='Race Distribution By Dept').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb735eaba8>




![png](Images/output_26_1.png)



```python
table3 = pd.crosstab(index=df_active["Department"], columns=df_active["Sex"])
table3.plot(kind="barh", figsize=(10,5), stacked=False, title='Gender Distribution By Dept')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bb73f74320>




![png](Images/output_27_1.png)


<br>
<br>
### *Factors Contributing to Employee Turnover*


```python
status_df = df_all.copy()
status_df.drop(['DeptID', 'EmpStatusID', 'EmployeeNumber', 'GenderID', 'ID',
                'MarriedID', 'MaritalStatusID', 'Zip', 'PerfScoreID'], axis=1, inplace=True)    
pd.pivot_table(status_df,index=["EmploymentStatus"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DaysEmployed</th>
      <th>PayRate</th>
    </tr>
    <tr>
      <th>EmploymentStatus</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Active</th>
      <td>38.087432</td>
      <td>1612.857923</td>
      <td>33.835902</td>
    </tr>
    <tr>
      <th>Future Start</th>
      <td>36.636364</td>
      <td>585.181818</td>
      <td>32.509091</td>
    </tr>
    <tr>
      <th>Leave of Absence</th>
      <td>41.357143</td>
      <td>1526.500000</td>
      <td>27.767857</td>
    </tr>
    <tr>
      <th>Terminated for Cause</th>
      <td>37.357143</td>
      <td>715.428571</td>
      <td>33.021429</td>
    </tr>
    <tr>
      <th>Voluntarily Terminated</th>
      <td>40.613636</td>
      <td>781.909091</td>
      <td>26.109886</td>
    </tr>
  </tbody>
</table>
</div>




```python
rft_df = (status_df[~status_df.ReasonForTerm.str.startswith('Not applicable')]).copy()
pd.pivot_table(rft_df,index=["ReasonForTerm"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DaysEmployed</th>
      <th>PayRate</th>
    </tr>
    <tr>
      <th>ReasonForTerm</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Another position</th>
      <td>42.100000</td>
      <td>760.050000</td>
      <td>26.675000</td>
    </tr>
    <tr>
      <th>attendance</th>
      <td>38.428571</td>
      <td>1122.142857</td>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>career change</th>
      <td>38.888889</td>
      <td>765.888889</td>
      <td>31.277778</td>
    </tr>
    <tr>
      <th>gross misconduct</th>
      <td>32.000000</td>
      <td>1596.000000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>hours</th>
      <td>41.222222</td>
      <td>505.333333</td>
      <td>29.577778</td>
    </tr>
    <tr>
      <th>maternity leave - did not return</th>
      <td>29.666667</td>
      <td>1386.666667</td>
      <td>34.083333</td>
    </tr>
    <tr>
      <th>medical issues</th>
      <td>31.666667</td>
      <td>723.666667</td>
      <td>31.806667</td>
    </tr>
    <tr>
      <th>military</th>
      <td>43.500000</td>
      <td>1014.250000</td>
      <td>18.750000</td>
    </tr>
    <tr>
      <th>more money</th>
      <td>41.363636</td>
      <td>762.363636</td>
      <td>22.545455</td>
    </tr>
    <tr>
      <th>no-call, no-show</th>
      <td>35.333333</td>
      <td>13.666667</td>
      <td>35.533333</td>
    </tr>
    <tr>
      <th>performance</th>
      <td>35.000000</td>
      <td>537.750000</td>
      <td>29.375000</td>
    </tr>
    <tr>
      <th>relocation out of area</th>
      <td>41.600000</td>
      <td>1001.600000</td>
      <td>28.200000</td>
    </tr>
    <tr>
      <th>retiring</th>
      <td>51.750000</td>
      <td>1148.000000</td>
      <td>29.812500</td>
    </tr>
    <tr>
      <th>return to school</th>
      <td>37.000000</td>
      <td>351.800000</td>
      <td>19.800000</td>
    </tr>
    <tr>
      <th>unhappy</th>
      <td>41.000000</td>
      <td>743.285714</td>
      <td>23.553571</td>
    </tr>
  </tbody>
</table>
</div>




```python
rft_df['ReasonForTerm'].value_counts()[0:20].plot(kind='barh',figsize=(10,8),colormap=cmap1,title='Reasons For Termination')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bb72970ac8>




![png](Images/output_31_1.png)



```python
table4 = pd.crosstab(index=rft_df["ReasonForTerm"], columns=rft_df["RaceDesc"])
table4.plot(kind="barh", figsize=(10,5), stacked=True, title='Reason For Term By Race').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb73b7eba8>




![png](Images/output_32_1.png)



```python
table5 = pd.crosstab(index=rft_df["ReasonForTerm"], columns=rft_df["MaritalDesc"])
table5.plot(kind="barh", figsize=(10,5), stacked=True, title='Reason For Term By Marital Status').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb731da278>




![png](Images/output_33_1.png)



```python
table6 = pd.crosstab(index=rft_df["ReasonForTerm"], columns=rft_df["Sex"])
table6.plot(kind="barh", figsize=(10,5), stacked=True, title='Reason For Term By Gender').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb73318dd8>




![png](Images/output_34_1.png)


<br>
<br>
### *Determinants for Verifying Recruiting Sources Ensure Diversity*


```python
costs_df = df_recruit_costs.copy()
costs_df.drop(['ID'], axis=1, inplace=True)
costTbl = pd.pivot_table(costs_df,index=["EmploymentSource"])
col_order = ['January_2018', 'February_2018', 'March_2018', 'April_2018', 'May_2018', 'June_2018',
             'July_2018', 'August_2018', 'September_2018', 'October_2018', 'November_2018', 'December_2018']
newCostTbl = costTbl.reindex(col_order, axis=1)
newCostTbl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>January_2018</th>
      <th>February_2018</th>
      <th>March_2018</th>
      <th>April_2018</th>
      <th>May_2018</th>
      <th>June_2018</th>
      <th>July_2018</th>
      <th>August_2018</th>
      <th>September_2018</th>
      <th>October_2018</th>
      <th>November_2018</th>
      <th>December_2018</th>
    </tr>
    <tr>
      <th>EmploymentSource</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Billboard</th>
      <td>520</td>
      <td>520</td>
      <td>520</td>
      <td>520</td>
      <td>0</td>
      <td>0</td>
      <td>612</td>
      <td>612</td>
      <td>729</td>
      <td>749</td>
      <td>910</td>
      <td>500</td>
    </tr>
    <tr>
      <th>Careerbuilder</th>
      <td>410</td>
      <td>410</td>
      <td>410</td>
      <td>820</td>
      <td>820</td>
      <td>410</td>
      <td>410</td>
      <td>820</td>
      <td>820</td>
      <td>1230</td>
      <td>820</td>
      <td>410</td>
    </tr>
    <tr>
      <th>Company Intranet - Partner</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Diversity Job Fair</th>
      <td>0</td>
      <td>5129</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4892</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Employee Referral</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Glassdoor</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Information Session</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Internet Search</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MBTA ads</th>
      <td>640</td>
      <td>640</td>
      <td>640</td>
      <td>640</td>
      <td>640</td>
      <td>640</td>
      <td>640</td>
      <td>1300</td>
      <td>1300</td>
      <td>1300</td>
      <td>1300</td>
      <td>1300</td>
    </tr>
    <tr>
      <th>Monster.com</th>
      <td>500</td>
      <td>500</td>
      <td>500</td>
      <td>440</td>
      <td>500</td>
      <td>500</td>
      <td>440</td>
      <td>500</td>
      <td>440</td>
      <td>440</td>
      <td>500</td>
      <td>500</td>
    </tr>
    <tr>
      <th>Newspager/Magazine</th>
      <td>629</td>
      <td>510</td>
      <td>293</td>
      <td>810</td>
      <td>642</td>
      <td>675</td>
      <td>707</td>
      <td>740</td>
      <td>772</td>
      <td>805</td>
      <td>838</td>
      <td>870</td>
    </tr>
    <tr>
      <th>On-campus Recruiting</th>
      <td>0</td>
      <td>0</td>
      <td>2500</td>
      <td>0</td>
      <td>0</td>
      <td>2500</td>
      <td>0</td>
      <td>0</td>
      <td>2500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>On-line Web application</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>0</td>
      <td>492</td>
      <td>0</td>
      <td>829</td>
      <td>744</td>
      <td>0</td>
      <td>610</td>
      <td>0</td>
      <td>0</td>
      <td>510</td>
      <td>0</td>
      <td>810</td>
    </tr>
    <tr>
      <th>Pay Per Click</th>
      <td>110</td>
      <td>110</td>
      <td>60</td>
      <td>121</td>
      <td>110</td>
      <td>109</td>
      <td>130</td>
      <td>146</td>
      <td>105</td>
      <td>109</td>
      <td>105</td>
      <td>110</td>
    </tr>
    <tr>
      <th>Pay Per Click - Google</th>
      <td>330</td>
      <td>330</td>
      <td>180</td>
      <td>362</td>
      <td>197</td>
      <td>152</td>
      <td>389</td>
      <td>437</td>
      <td>315</td>
      <td>327</td>
      <td>315</td>
      <td>176</td>
    </tr>
    <tr>
      <th>Professional Society</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Search Engine - Google Bing Yahoo</th>
      <td>330</td>
      <td>410</td>
      <td>388</td>
      <td>372</td>
      <td>472</td>
      <td>412</td>
      <td>416</td>
      <td>495</td>
      <td>619</td>
      <td>502</td>
      <td>389</td>
      <td>378</td>
    </tr>
    <tr>
      <th>Social Networks - Facebook Twitter etc</th>
      <td>420</td>
      <td>481</td>
      <td>452</td>
      <td>479</td>
      <td>392</td>
      <td>508</td>
      <td>578</td>
      <td>466</td>
      <td>389</td>
      <td>439</td>
      <td>491</td>
      <td>478</td>
    </tr>
    <tr>
      <th>Vendor Referral</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Website Banner Ads</th>
      <td>400</td>
      <td>400</td>
      <td>300</td>
      <td>388</td>
      <td>592</td>
      <td>610</td>
      <td>620</td>
      <td>669</td>
      <td>718</td>
      <td>767</td>
      <td>816</td>
      <td>865</td>
    </tr>
    <tr>
      <th>Word of Mouth</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
newCostTbl.plot(kind="barh", figsize=(10,8), stacked=True, colormap=cmap1,
                title='2018 Employment Source Costs').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb7248b6a0>




![png](Images/output_37_1.png)



```python
hTbl = pd.crosstab(index=df_all["EmployeeSource"], columns=df_all["RaceDesc"])
hTbl.plot(kind="barh", figsize=(15,12), stacked=True, title='Pre-Hire Employee Source By Race').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb757daef0>




![png](Images/output_38_1.png)



```python
hTbl2 = pd.crosstab(index=df_all["EmployeeSource"], columns=df_all["Sex"])
hTbl2.plot(kind="barh", figsize=(15,12), stacked=True, title='Pre-Hire Employee Source By Gender').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb75c5ed30>




![png](Images/output_39_1.png)



```python
hired_df=df_all.copy()
hired_df['HireMnthYr']=pd.to_datetime(hired_df['HireDate']).dt.strftime('%m/%Y')
new_hired_df = hired_df[hired_df['HireMnthYr'].str.endswith(('2018','2017', '2016', '2015', '2014'))]

plt.figure(figsize=(16,5))
sns.countplot('HireMnthYr', data = new_hired_df)
plt.xticks(rotation=60, ha='right')
plt.title('Staff Hire Date Counts Within Last Five Years (Month/Year)')
plt.tight_layout()
```


![png](Images/output_40_0.png)


<br>
<br>
### *Determinants For Correlations Between Employee Performance and Immediate Supervisor*


```python
deptMgr_df = df_all[['Department', 'ManagerName']].copy()
newdeptMgr_df = deptMgr_df[~deptMgr_df[['Department', 'ManagerName']].apply(frozenset, axis=1).duplicated()]
newdeptMgr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Department</th>
      <th>ManagerName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Admin Offices</td>
      <td>Brandon R. LeBlanc</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Admin Offices</td>
      <td>Janet King</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Admin Offices</td>
      <td>Board of Directors</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Executive Office</td>
      <td>Board of Directors</td>
    </tr>
    <tr>
      <th>11</th>
      <td>IT - Information Systems</td>
      <td>Janet King</td>
    </tr>
    <tr>
      <th>12</th>
      <td>IT - Information Systems</td>
      <td>Simon Roup</td>
    </tr>
    <tr>
      <th>25</th>
      <td>IT - Information Systems</td>
      <td>Jennifer Zamora</td>
    </tr>
    <tr>
      <th>30</th>
      <td>IT - Information Systems</td>
      <td>Eric Dougall</td>
    </tr>
    <tr>
      <th>34</th>
      <td>IT - Information Systems</td>
      <td>Peter Monroe</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Production</td>
      <td>Janet King</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Production</td>
      <td>Michael Albert</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Production</td>
      <td>Elijiah Gray</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Production</td>
      <td>Webster Butler</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Production</td>
      <td>Amy Dunn</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Production</td>
      <td>Ketsia Liebig</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Production</td>
      <td>Brannon Miller</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Production</td>
      <td>David Stanley</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Production</td>
      <td>Kissy Sullivan</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Production</td>
      <td>Kelley Spirea</td>
    </tr>
    <tr>
      <th>260</th>
      <td>Sales</td>
      <td>Lynn Daneault</td>
    </tr>
    <tr>
      <th>261</th>
      <td>Sales</td>
      <td>John Smith</td>
    </tr>
    <tr>
      <th>287</th>
      <td>Sales</td>
      <td>Janet King</td>
    </tr>
    <tr>
      <th>288</th>
      <td>Sales</td>
      <td>Debra Houlihan</td>
    </tr>
    <tr>
      <th>291</th>
      <td>Software Engineering</td>
      <td>Alex Sweetwater</td>
    </tr>
    <tr>
      <th>300</th>
      <td>Software Engineering</td>
      <td>Jennifer Zamora</td>
    </tr>
    <tr>
      <th>302</th>
      <td>IT - Information Systems</td>
      <td>Brian Champaigne</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(16,5))
sns.countplot('ManagerName', data = df_all)
plt.xticks(rotation=60, ha="right")
plt.title('Employee Count Per Manager')
plt.tight_layout()
```


![png](Images/output_43_0.png)



```python
ps_df = df_all[~df_all.PerformanceScore.str.startswith('Not applicable')]
mgrTbl = pd.crosstab(index=ps_df["ManagerName"], columns=ps_df["PerformanceScore"])
mgrTbl.plot(kind="barh",figsize=(12,10),stacked=True,
            title='Performance Score Counts Given Per Manager').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb73b55e10>




![png](Images/output_44_1.png)



```python
deptTbl = pd.crosstab(index=ps_df["Department"], columns=ps_df["PerformanceScore"])
deptTbl.plot(kind="barh",figsize=(10,8),stacked=True,title='Performance Score Counts Per Dept').legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x1bb767f9b38>




![png](Images/output_45_1.png)



```python
p1 = sns.factorplot("PerformanceScore", data=ps_df, aspect=4, kind="count")
p1.set_xticklabels(rotation=0)
p1 = plt.title("Staff Performance Score Distribution")
```


![png](Images/output_46_0.png)



```python
df_active['PerformanceScore'].value_counts()[0:20].plot(kind='pie',figsize=(8,8),autopct='%1.1f%%', colormap=cmap2,
                                                        title='Current Staff Performance Score Percentages',label='')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1bb7697c668>




![png](Images/output_47_1.png)

