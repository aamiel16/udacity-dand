
# coding: utf-8

# # Titanic Exploratory Data Analysis

#  References:
#  
#  http://www.history.com/this-day-in-history/titanic-sinks
#  
#  https://en.wikipedia.org/wiki/Lifeboats_of_the_RMS_Titanic

# # 1 Introduction
# 
# <p align="justify">The RMS Titanic, one of the largest and most luxurious ocean liners ever built from its time. Spanning about 883 feet from stern to bow, a height of 175 feet, and a massive weight of 46,000 tons, it was made to carry about 3,500 passengers and crew. The ship was thought to be unsinkable. However on April 14, 1912 just before midnight, the ship failed to divert its course from an iceberg, which left at least 5 of its hull compartments ruptured. Due to the shortage of lifeboats, only about 700 passengers survived out of the approximately 2,224 passengers on board. 
# </p>
# 
# <p align="justify">
# This exploratory data analysis is submitted as partial fullfilment of the requirements for Udacity's Data Analyst Nanodegree Program. In this notebook, I shall be going into the step by step process of data analysis, in order to answer some questions regarding the given data.
# </p>
# 

# ## 1.1 Data Wrangling

# ### Imports and Loading the Data 

# In[144]:

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
get_ipython().magic('matplotlib inline')
fig_prefix = '../figures/2015-09-20-as-titanic-explo-lab-'


# In[145]:

# Getting the titanic data
titanic_df = pd.read_csv('../data/titanic_data.csv')


# ### Some information about the data.

# In[146]:

# Some information about the data
titanic_df.info()


# From here we can see that there are about **891 entries** with a total of **12 columns**. The data types for each column can be observed as well. The variable descriptions as obtained from Kraggle is showed from this text file *[data_descriptions.txt](../data/data_descriptions.txt)* and summarized by the table below:
# 
# <img src=../res/table_desc.png />

# In[147]:

# Looking at some entries of the data
titanic_df.head(6)


#  

# ## 1.2 Data Exploration

# ### Counting the NaN/null entries by column
# Upon looking at the data, I observed that some rows have no entries under the Age and Cabin column. In order to explore the data even further, I looked at each column and check the number of NaN entries in that column.

# In[148]:

# Counts the number of NaN or null entries in each column
titanic_df.isnull().sum()


# Upon reading some online resources, I found out that there are options on how to deal with these rows. 
# 
# First option is by dropping the rows that have no entries under the Age and Cabin column, but after observing that a large number of rows have no entries under the Age and Cabin column, it would greatly affect my dataset if I were to drop a large number of these rows. 
# 
# Hence, my second option, using some basic regression to predict the values of Age and Cabin columns based on other rows' value.

# # 2 Analysis
# 
# In this part, I would go further into the analysis of factors that could have affected survivability.

# ## 2.1 Data Visualization

# ### Survivability by Sex
# As per reports, most of the 700 survivors of Titanic are female, and children. This can be seen and reflected in the graph below.

# In[149]:

# Looking into survivability by sex
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Sex', y='Survived', data=titanic_df, estimator=np.sum, ci=0)

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xticklabels(['Male', 'Female'])
ax.set_xlabel('Sex', fontsize=13)
ax.set_ylabel('No. of Survivors', fontsize=13)
ax.set_title('Survivability by Sex', fontsize=14)

fig.savefig(fig_prefix+'survivability_by_sex')


# ### Survivability by Class

# In[150]:

# Looking into survivability by Class
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Pclass', y='Survived', data=titanic_df, estimator=np.sum, ci=0)

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xticklabels(['First Class', 'Second Class', 'Third Class'])
ax.set_xlabel('Passenger Class', fontsize=13)
ax.set_ylabel('No. of Survivors', fontsize=13)
ax.set_title('Survivability by Class', fontsize=14)

fig.savefig(fig_prefix+'survivability_by_class')


# ### Survivability by Sex in each Class

# In[151]:

# Looking into survivability by Class
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df, estimator=np.sum, ci=0)

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xticklabels(['First Class', 'Second Class', 'Third Class'])
ax.set_xlabel('Passenger Class', fontsize=13)
ax.set_ylabel('No. of Survivors', fontsize=13)
ax.set_title('Survivability by Sex in each Class', fontsize=14)

fig.savefig(fig_prefix+'survivability_by_sex_and_class')


# ## 1.3 Data Cleaning

# ### Filling up the Age Column
# Since the Age column has the least number of blank entries, I decided to fill this column first. In filling up the missing age, I could have gotten the median age by sex and just subtitute this median age for every missing value based on the sex of that row. But, I have observed from the data that the title of names (i.e. Mr, Ms, Master, etc.) could also have something to do with age, especially that the title `Master` seemed to be associated with children.

#     Scratch: Thinking of just finding the median age by sex and putting the median age on the missing values. BUT! Why not try extracting the title of the name (i.e. Mr., Mrs, etc) and then find the median age based on the sex and the title.
#     
#     Steps:
#     1. Split surname by using , as delimiter
#     2. Create a list of titles from the data
#     3. Unify these titles

# ### Splitting the Name
# Using the Name column of the data frame, I would split the name and add three additional columns for that name's title, first name, and surname in the data frame.

# In[152]:

# Splitting the surname from the rest of the name
s_surname = titanic_df['Name'].str.split(',', expand=True)

# Splitting the title from the rest of the name
s_title = s_surname[1].str.split('.', expand=True)

# Putting the name sections in series
s_firstname = s_title[1].str.strip()
s_title = s_title[0].str.strip()
s_surname = s_surname[0]

# # Editing the column name
s_firstname = s_firstname.rename('Firstname')
s_title = s_title.rename('Title')
s_surname = s_surname.rename('Surname')


# In[153]:

# Adding the name sections to the data frame
titanic_df = titanic_df.join([s_title, s_firstname, s_surname])


# In[154]:

# Confirming that the columns have been added
titanic_df.head()


# ### Organizing the Title

# Now that the additional columns have been added, I would then would like to look at the different titles that I would be working with. Upon looking at the different titles, I decided to unify some of the titles in order to fill the missing ages according to the passenger's title.

# In[155]:

# Looking at the different titles in the Data
list_title = titanic_df['Title'].unique()
list_title


# In[187]:

# Looking at the different titles with missing ages
list_title_missing = titanic_df[titanic_df['Age'].isnull()]['Title'].unique()
list_title_missing


# In[188]:

# Function that would unify or organize the title based on the titles with missing ages
def unify_title(df):
    '''
    Returns the respective title
    '''
    if df in ['Don', 'Rev', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer', 'Mr']:
        return 'Mr'
    elif df in ['Lady', 'the Countess', 'Mrs']:
        return 'Mrs'
    elif df in ['Mme', 'Mlle', 'Ms', 'Miss']:
        return 'Miss'
    else:
        return df


# In[182]:

titanic_df['Unify_title'] = titanic_df['Title'].apply(unify_title)


# ### Filling the Missing Ages

# In filling up the missing ages, I Since, I wouldn't want to modify the original data, I would make a new column `'Filled_age'` that would contain all passenger's age.

# In[193]:

mean_age_by_title = titanic_df.groupby('Unify_title', as_index=False).mean()[['Unify_title', 'Age']]


# In[194]:

sns.boxplot(x='Unify_title', y='Age', data=titanic_df)


# In[ ]:



