#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing all the necessary python libraries to perform our analysis.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



# In[4]:


#Acquiring the data from csv file(Source-Kaggle)
df=pd.read_csv(r"D:\Data_Analyst_Bootcamp\Naukri_Data_Analysis_Project\Updated_datasets\naukri_com-job_sample.csv")


# In[5]:


#Let us have a brief overview of the data.
df.head(5)


# In[230]:


#Fetching the list of columns
df.columns


# In[232]:


#Getting the number of rows and columns using shape attribute
df.shape


# In[233]:


#Getting the number of missing values for each column
count_missing=df.isnull().sum()


# In[234]:


#Display the missing values feature wise.
count_missing


# In[235]:


#Getting the count of missing values in a list format
count_missing.to_list()


# In[236]:


#Getting the % of missing values out of total missing values
perc_of_missing_values=count_missing*100/len(df)


# In[237]:


perc_of_missing_values.to_list()


# In[238]:


#Getting a dataframe with count and percentage of missing values using the dataframe method
missing_count_df=pd.DataFrame({'count_missing':count_missing,'perc_of_missing_values':perc_of_missing_values})


# In[239]:


# This way we can visualize the count and percentage of missing values for each respective feature
missing_count_df


# In[226]:


#Changing the background of dataframe for better visualization aspects
missing_count_df.style.background_gradient(cmap='Spectral')


# In[240]:


#Printing the columns with the missing values using string formatting 
for col in df.columns:
    print("{} has {} null values".format(col,df[col].nunique()))


# In[17]:


#Getting the dataframe with column,unique values count and list of unique values using for loop.
feature_unique_count=[]
for col in df.columns:
    feature_unique_count.append([col,df[col].nunique(),df[col].unique()])
    
    
    


# In[18]:


#Getting the dataframe with column,unique values count and list of unique values using list comprehension.
feature_unique_cnt=[[col,df[col].nunique(),df[col].unique()] for col in df.columns]


# In[19]:


count_df=pd.DataFrame(feature_unique_cnt,columns=['col_name','count','unique'])


# In[20]:


#Implementing kind of styling with the count column in the dataframe.
count_df.style.background_gradient(cmap='Spectral')


# # Data Cleaning and Feature Engineering

# # Data Cleaning on the Payrate feature

# In[24]:


#Lets clean the payrate feature and do the feature engineering to extract min_payrate and max_payrate.
df['payrate'][0].split('-')


# In[25]:


#Hold the count of values for each split length in len_pay list. 
len_pay=[]
for pay in df['payrate']:
    len_pay.append(len(str(pay).split('-')))


# In[26]:


#Checking the count of values for each length type after splitting of list as values greater than 2 are kind of invalid entries
pd.Series(len_pay).value_counts()


# In[32]:


#Creating a dataframe after splitting the columns into multiple columns 
payrate_split=df.payrate.str.split('-',expand=True)


# In[35]:


payrate_split


# # Lets us perform the data-preprocessing on the payrate_split dataframe and derive the min payrate feature

# In[ ]:


#Lets clean the min payrate feature first and then we will convert it into int or float datatype.
# We will clean it into 4 steps:-
 #1> Remove all the white spaces.
 #2> Replace the comma with empty string.
 #3> Remove all the set of characters.
 #4> Then we will convert it into int or float datatype.


# In[36]:


#Removing the whitespaces from the min_pay feature of the dataframe i.e payrate_split[0]
payrate_split[0]=payrate_split[0].str.strip()


# In[42]:


#Removing the comma from the min_pay feature of the dataframe i.e payrate_split[1]
payrate_split[0]=payrate_split[0].str.replace(',','')


# In[43]:


payrate_split


# # We will try to convert the payrate feature into numeric value to perform further analysis on this feature

# In[44]:


#Approach_1:-(Working correctly using in-built function)
payrate_split[0]=pd.to_numeric(payrate_split[0],errors='coerce')


# In[45]:


payrate_split[0]


# # Let us now perform the pre-processing on the max  payrate feature and derive the ma payrate feature

# In[ ]:


#Lets clean the max payrate feature first and then we will convert it into int or float datatype.
#We will clean it into 4 steps:-
 #1> Remove all the white spaces.
 #2> Replace the comma with empty string.
 #3> Remove all the set of characters.
 #4> Then we will convert it into int or float datatype.


# In[46]:


payrate_split[1][0]


# In[47]:


#Removing the white spaces from max_payrate feature
payrate_split[1]=payrate_split[1].str.strip()


# In[48]:


#Replacing the comma with empty string.
payrate_split[1]=payrate_split[1].str.replace(',','')


# In[49]:


#Replacing all the non-digit characters with the empty string. 
payrate_split[1]=payrate_split[1].str.replace('\D+','',regex=True)


# In[50]:


#Updating the record manually for one of the values as it is not correct slightly.
payrate_split.loc[1,1]='250000'


# In[51]:


payrate_split[1]


# In[52]:


payrate_split.dtypes


# In[53]:


#Converting the payrate max_pay feature into numeric datatype.
payrate_split[1]=pd.to_numeric(payrate_split[1])


# In[54]:


payrate_split[1]


# In[55]:


#Creating a dataframe of the min_pay and max_pay feature.
pay=pd.concat([payrate_split[0],payrate_split[1]],axis="columns",sort=False)


# In[56]:


#Adding the column names to min_pay and max_pay
pay.columns=['min_pay','max_pay']


# In[57]:


#Here finally our minimum payrate feature is also prepared and max payrate feature is also prepared so we will append it to our main dataframe.
df=pd.concat([df,pay],axis="columns",sort=False)


# In[59]:


df.dtypes


# In[60]:


df.columns


# # Performing feature engineering on the experience feature.

# In[75]:


#We are splitting the  experience feature and checking if split length is greater than 2 then it is an invalid entry. 
len1=[]
for exp in df['experience'].dropna():
    if len(exp.split('-'))!=2:
        len1.append(exp)
        
        


# In[76]:


#Getting the list of invalid entry in our experience feature and we need to clean this feature  
len1


# In[77]:


def split_exp2(exp):
    '''
     > Takes experience feature as the parameter
     > After splitting the experience feature if length of list is equal to 2 then store the first
       slice value as min_exp and next_slice value as max_exp
     > If length is greater than 2 then it is kind of an invalid entry and function will return NaN for both
    '''
    try:
        if len(exp.split('-'))==2:
            min_exp=exp.split('-')[0]
            max_exp=exp.split('-')[1]
        return pd.Series([min_exp,max_exp])
    except:
        return pd.Series([np.nan,np.nan])
    


# In[78]:


#Adding min and max exp feature into our main dataframe
df[['min_exp','max_exp']]=df['experience'].apply(split_exp2).rename(columns={0:'min_exp',1:'max_exp'})


# In[79]:


#In the next 4 blocks we are validating the behaviour of the above function
nm=pd.DataFrame(df['experience'].str.contains('Not Mentioned'))


# In[80]:


nm


# In[81]:


nm[nm['experience']==True].index


# In[82]:


df['experience'][7193]


# In[83]:


split_exp2(df['experience'][7193])


# In[84]:


#We are replacing the yrs string with the empty value
df['max_exp']=df['max_exp'].str.replace('yrs','')


# In[86]:


df.head(5)


# In[87]:


#Changing the datatype of the min and max experience feature to float datatype to perform our further analysis
df['max_exp']=df['max_exp'].astype(float)
df['min_exp']=df['min_exp'].astype(float)


# In[93]:


df.columns


# # We are deriving the avg_pay and avg_exp feature for more in-   depth analysis.

# In[96]:


df['avg_exp']=(df['min_exp']+df['max_exp'])/2


# In[97]:


df['avg_pay']=(df['min_pay']+df['max_pay'])/2


# In[111]:


df.head(5)


# In[116]:


df.columns


# # Performing feature engineering on the postdate feature to derive day, month and year attribute.

# In[117]:


df['postdate']


# In[118]:


df['postdate'].dtype


# In[119]:


#Converting the post date feature into the datetime datatype to extract the attributes. 
df['postdate']=pd.to_datetime(df['postdate'])


# In[120]:


#Approach_1:-Using user defined function.
def fetch_dt_attributes(df,feature):
    try:
        return pd.Series([df[feature].dt.day,df[feature].dt.month,df[feature].dt.year])
    except:
        print("The datatype is not supported")
        
        
        
        



# In[121]:


#Adding the day,month and year feature to our dataframe after the value is getting returned from the function.
df[['day','month','year']]=fetch_dt_attributes(df,'postdate')


# In[122]:


df.columns


# In[125]:


#Approach:-2 Using map approach:-
def fetch_dt_att2(x):
    return ([x.day,x.month,x.year])


# In[126]:


fe_date=pd.DataFrame(map(fetch_dt_att2,df['postdate'])).rename(columns={0:'day',1:'month',2:'year'})


# In[127]:


fe_date


# In[131]:


#Adding the day,month and year feature to our dataframe
pd.concat([df,fe_date],axis="columns")


# # Preparing our job location feature for our analysis

# In[132]:


# Let us prepare the job location feature.
rep=pd.read_csv(r"D:\Data_Analyst_Bootcamp\Naukri_Data_Analysis_Project\Updated_datasets\replacements.csv").set_index('Unnamed: 0')


# In[133]:


df['joblocation_address'].value_counts()


# In[134]:


#Creating a copy of our original dataframe.
data=df.copy()


# In[135]:


#This dictionary contains the list of replacements to be done for job location feature
replacement_dict=rep.to_dict()


# In[136]:


replacement_dict


# In[137]:


data.replace(replacement_dict,inplace=True,regex=True)


# In[138]:


data['joblocation_address'].value_counts()


# In[402]:


data['joblocation_address'].unique()


# In[150]:


data.columns


# In[147]:


#Dropping the unnecessary columns that are not rquired as part of furher analysis and improve the performance of the dataframe.
data.drop(columns=['experience','payrate','postdate','uniq_id'],axis=1,inplace=True)


# In[ ]:


#Let us detect the outliers in the min pay and max pay feature and try to remove the outliers for better set of analysis


# In[151]:


#Approach2:-Using the IQR and box plot to treat the outliers:-
#First we are calculating the Q1,Q3 that is 25th percentile and 75th percentile value 
Q1,Q3=data.min_pay.quantile([0.25,0.75])
Q1,Q3




# In[152]:


#Interquartile Range i.e difference of Q3 and Q1
IQR=Q3-Q1
IQR


# In[153]:


#Calculating the lower limit:-
lower=Q1- 1.5*IQR
lower


# In[154]:


#Calculating the upper limit:-
upper=Q3+1.5*IQR
upper


# In[155]:


#Anything out of lower and upper limit is treated as outlier
lower,upper


# In[156]:


#Storing our min pay feature after removing outliers
data['min_pay']=data[(data.min_pay>lower) & (data.min_pay<upper)]['min_pay']


# In[ ]:





# In[157]:


Q1_max,Q3_max=data.max_pay.quantile([0.25,0.75])
Q1_max,Q3_max


# In[158]:


IQR_max=Q3_max-Q1_max
IQR_max


# In[159]:


lower_max=Q1- 1.5*IQR


# In[160]:


upper_max=Q3+1.5*IQR


# In[165]:


lower_max


# In[167]:


upper_max


# In[168]:


data[(data.max_pay<lower_max) | (data.max_pay>upper_max)]['max_pay']


# In[133]:


#Our data is ready and we are trying to export it  to csv file for any further analysis. 
data.to_csv(r"D:\Data_Analyst_Bootcamp\Naukri_Data_Analysis_Project\Updated_datasets\naukri_data.csv",index=False)


# In[169]:


#Final list of columns for our data
data.columns


# In[170]:


#Now our data is ready we have to perform some descriptive analysis on the data.
import warnings
warnings.filterwarnings('ignore')



# In[171]:


#We have loaded our final dataframe with the features ready and pre-processed to perform our data visualization
naukri_df=pd.read_csv(r"D:\Data_Analyst_Bootcamp\Naukri_Data_Analysis_Project\Updated_datasets\naukri_data.csv")


# In[174]:


naukri_df


# In[175]:


naukri_df.head(4)


# In[176]:


#Let us perform descriptive analysis of the data.
df.describe().T


# In[177]:


#Performing descriptive analysis with the categorical feature.
df.describe(include=['O']).T


# In[ ]:


#Types of analysis we are going to perform on the data.
#Uni-Variate,Bi-Variate and MultiVariate analysis


# In[ ]:


#Uni-Variate Analysis:-We perform this analysis on he basis of one variable may be categorical or numerical variable.
#Examples:-
#Graphical analysis for continous variables(Boxplot,Histogram,Distribution_Plot),Tabular Analysis(using inbuilt function),Categorical Analysis(Country/Strength,Bar Chart,Pie Chart,CountPlot)
#Bi-Variate Analysis:-
#Scenario_1:-X>Cat,Y>Num(Pie,Donut Chart),Scenario_2:-X>Num,Y>Num(Correlation,ScatterPlot),Scenario_3:-X>Cat,Y>Cat(Very rare chi-square test)





# # Let us perform bi-variate analysis on the data and derive   various insights 
# 

# In[178]:


#We have to extract all the categorical features from the dataframe using various approaches:-
#1>Using list comprehension:-

#List of categorical feature.
categorical=[col for col in naukri_df.columns if naukri_df[col].dtype=='object']



# In[179]:


print(categorical)


# In[184]:


#List of numerical feature.
numerical=[col for col in naukri_df.columns if naukri_df[col].dtype!='object']


# In[187]:


print(numerical)


# # We will perform the data visualization on the  basis of various aspects and understand the trend and relationship between the data

# In[411]:


#Since pandas version 2.0.0 now you need to add numeric_only=True param to avoid the issue
plt.figure(figsize=(12,6))
sns.heatmap(naukri_df.corr(numeric_only=True),cmap='PuOr',annot=True,linewidth=4,linecolor='yellow')
plt.title("Heatmap of Job market analysis of India",fontsize="30",color="green")
plt.show()


# # Key_Insights >

# 
# ☞If my min pay increases then avg_pay also increases as it has a positive correlation of 0.048 as per the above heatmap.
# 
# ☞Min_pay and min_exp is also highly correlated as it has correlation value of  0.71 showing positive correlation
# 
# ☞If my min_pay also increases then my max_pay also increases as both these features have a very high correlation value of 1.
# 
# 

# # Perform the analysis that which company has maximum       number of jobs.

# In[188]:


#Perform the analysis that which company has maximum number of jobs:-
comp=naukri_df['company'].value_counts().reset_index()


# In[189]:


comp.columns=['company','number of jobs']


# In[190]:


comp


# # Visualizing the top 5 companies having maximum number of jobs using bar chart

# In[191]:


fig=px.bar( comp,comp['company'][0:5],comp['number of jobs'][0:5],
            color=comp['number of jobs'][0:5],
            text=comp['number of jobs'][0:5],
            title="Companies vs No. of jobs",
            labels={ "x": "Companies",
                     "y": "Number of jobs" })
fig.update_layout(
           font_family="Nunito",
           font_color="blue",
           title_font_family="Nunito",
           title_font_color="green",
           legend_title_font_color="green"
)
fig.update_xaxes(title_font_family="Nunito")
fig.update_yaxes(title_font_family="Nunito")
fig.show()


# # Key Insights >
# 
# 

# In[ ]:


☞ Indian Institute of Bombay, Confidential are the organizations with the largest job openings


# # Visualizing the companies having maximum contribution to  jobs using pie chart

# In[192]:


#Visualizing the companies having maximum contribution to jobs using pie chart.
#Plotting a pie chart:-
plt.figure(figsize=(14,8))
plt.pie(labels=comp['company'][0:10],x=comp['number of jobs'][0:10],autopct="%0.1f%%",shadow=False)
plt.title("Company wise number of jobs contribution",color="green",fontsize=20)
plt.show()


# In[195]:


naukri_df.columns


# In[196]:


#Automating our code to perform different plots as per the needs of further analysis
def perform_analysis(naukri_df,feature,col1,col2,chart=3):
    '''
    This will return us plots depending upon whatever chart we want
    Parameters
    ------
    Data:Dataframe
    feature:column that we have to consider for analysis
    col1:first column that we want to assign to df
    col2:second column that we want to assign to new df
    @chart:if its value is 1 then it represents bar chart
    @chart:if its value is 2 then it represents pie chart
    
    '''
    dataframe=naukri_df[feature].value_counts().reset_index()
    dataframe.columns=[col1,col2]
    if chart==1: #Use seaborn bar chart here
        plt.bar(dataframe[col1][0:10],dataframe[col2][0:10])
        plt.xticks(rotation='vertical')
        plt.title("{} vs Number of jobs".format(feature),color="green",fontsize=20)
        plt.show()
    elif chart==2:
        #Plotting a pie chart:-
        plt.figure(figsize=(14,8))
        plt.pie(labels=dataframe[col1][0:10],x=dataframe[col2][0:10],shadow=False,autopct="%0.1f%%")
        plt.title("{} vs number of jobs".format(feature),color="green",fontsize=20)
        plt.show()
    elif chart==3:
        fig=px.bar(dataframe,x=dataframe[col1][0:5],y=dataframe[col2][0:5],color=dataframe[col2][0:5],text=dataframe[col2][0:5],height=800,title="{} vs {}".format(col1,col2),labels={
                     "x": "{}".format(col1),
                     "y": "{}".format(col2),
                 })
        fig.update_layout(
           font_family="Times New Roman",
           font_color="blue",
           title_font_family="Times New Roman",
           title_font_color="green",
           legend_title_font_color="green"
)
        fig.update_xaxes(title_font_family="Arial")
        fig.update_yaxes(title_font_family="Arial")
        fig.show()
        


# In[197]:


naukri_df['industry'].value_counts()


# # Perform the analysis that which top 5 industries  has maximum  number of jobs in the job market

# In[198]:


#We have to perform the analysis with respect to the industries that which industries has most number of jobs.
perform_analysis(naukri_df,'industry','Industries','Number of Jobs',3)


# # Key Insights >
# ☞ Software services sector,Education,BPO,Banking Financial Services,Recruitment    Staffing has maximum number of jobs in the job market

# # Perform the analysis that which top 5 Jobtitle has maximum  number of jobs in the job market

# In[199]:


#We have to perform the analysis with respect to Jobtitle that which job title has most number of jobs.
perform_analysis(naukri_df,'jobtitle','Jobtitles','Number of Jobs',3)


# # Key Insights:-
# 
# ☞ Business Development Executive,Business Development Manager,Software Engineer,Project Manager has most number of jobs share in the overall job market as per the above visuals

# # Perform the analysis that what are the top 5 most rated skills in the industry
# 

# In[216]:


#We have to perform the analysis with respect to Jobtitle that which job title has most number of jobs
perform_analysis(naukri_df,'skills','Skills','Count',3)


# # Key Insights:- 
# ☞ It Software-Application Programming has most number of jobs in the job market followed by Sales,ITES,Teaching,HR

# # Perform the analysis that what are the top 5 job locations with the most number of jobs

# In[217]:


perform_analysis(naukri_df,'joblocation_address','City','Number of jobs',3)


# ☞ Bangalore,Mumbai,Delhi,Hyderabad and Chennai are the joblocations with most number of jobs in the job market

# # Let us now perform the analysis on the  experience and payrate feature and derive some key insights out of it

# In[218]:


#Analysis between between min_exp and min_pay.
plt.figure(figsize=(14,6))
sns.stripplot(x='min_exp',y='min_pay',data=naukri_df)
sns.boxplot(x='min_exp',y='min_pay',data=naukri_df)
plt.title("Analyzing the relation between min experience and min pay")


# # Key Insights:-

# ☞ As the min_exp increases the min_pay also increases not always but there is a   upward trend showing to us

# In[219]:


plt.figure(figsize=(14,6))
sns.stripplot(x='max_exp',y='max_pay',data=naukri_df)
sns.boxplot(x='max_exp',y='max_pay',data=naukri_df)
plt.title("Striplots between max_exp and max_pay")
plt.ylim(50000,6000000)


# # Key Insights:- As the min_exp increases the min_pay also increases not always but there is a upward trend

# In[220]:


data.columns


# # Perform the analysis on the industry wise min and max pay
# 

# In[221]:


data[['max_pay','industry']].groupby(['industry']).median().sort_values(by=['max_pay'],ascending=False).head(10).plot(kind='bar',color='Green')
plt.title("Top 10 Industries with Max_Pay")


# # Key Insights>
# 

# ☞ Pulp and Paper, Strategy and Management  consulting firms are the top 3 firms with maximum pay.

# In[222]:


data[['min_pay','industry']].groupby(['industry']).median().sort_values(by=['min_pay'],ascending=False).head(10).plot(kind='bar',color='Green')
plt.title("Top 10 Industries with the min pay")


# In[223]:


df.columns


# # Perform the analysis on the top rated skills in the industry.

# In[224]:


#Let us analyze which skill has max pay compared to all other skill sets.
sns.set(style="whitegrid")
plt.figure(figsize=(20,10))
sns.boxplot(x="skills",y="avg_pay",data=naukri_df,showmeans=True)
plt.xlabel("Skills",fontsize=30)
plt.ylabel("Avg_Pay",fontsize=30)
plt.title("Skills vs AvgPay")
plt.xticks(rotation='vertical')
plt.ylim(100000,5000000)
plt.show()


# # Key Insights >

# ☞ As per the above plot Top Management,It Software Middleware and IT-Software mobile are the top three rated skills in the job market as per the avg_pay as these has higher average pay compared to others

# In[183]:





# # Perform the analysis on the available position in the industry.

# In[225]:


#Use plotly bar chart for customizations with the plot:-

industry_positions=naukri_df.groupby('industry')['numberofpositions'].sum().reset_index().sort_values(by="numberofpositions",ascending=False)
industry_positions
fig=px.bar(industry_positions,x=industry_positions['industry'][0:5],y=industry_positions['numberofpositions'][0:5],color=industry_positions['industry'][0:5],text=industry_positions['numberofpositions'][0:5],height=800,title="Industry vs Noofpositions",labels={
                     "x": "Industries",
                     "y": "Number of positions"
                 })
fig.update_layout(
           font_family="Times New Roman",
           font_color="blue",
           title_font_family="Calibri",
           title_font_color="green",
           legend_title_font_color="green"
)
fig.show()



# # Key Insights >

# ☞ BPO/Call Centre/ITES,Medical Healthcare hospitals,It Software Services are  the top 3 industries

# # Key learnings from the real world case study:-
# >
# 
# >
# 
# >
# 
# >
# 
# >
# 
# >

# In[ ]:


-----------------------------------End of Case Study---------------------------

