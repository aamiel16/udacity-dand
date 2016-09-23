
# coding: utf-8

# # Titanic Exploratory Data Analysis

#  References:
#  
#  http://www.history.com/this-day-in-history/titanic-sinks
#  
#  https://en.wikipedia.org/wiki/Lifeboats_of_the_RMS_Titanic

# # 1 Introduction
# 
# <p align="justify">The RMS Titanic, one of the largest and most luxurious ocean liners ever built from its time. Spanning about 883 feet from stern to bow, a height of 175 feet, and a massive weight of 46,000 tons, it was made to carry about 3,500 passengers and crew. With its massive build, the ship was thought to be unsinkable. However on April 14, 1912 just before midnight, the ship failed to divert its course from an iceberg, which left at least 5 of its hull compartments ruptured. Due to the shortage of lifeboats, only about 700 passengers survived out of the approximately 2,224 passengers on board. 
# </p>
# 
# <p align="justify">
# This exploratory data analysis is submitted as partial fullfilment of the requirements for Udacity's Data Analyst Nanodegree Program. In this notebook, I shall be going into the step by step process of data analysis, in order to answer some questions regarding the given data. The data to be used in this notebook was provided by Udacity and the description of variables was obtained from Kaggle.
# </p>
# 

# ## 1.1 Data Wrangling

# ### Imports and Loading the Data 

# In[2]:

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
get_ipython().magic('matplotlib inline')
fig_prefix = '../figures/2015-09-20-as-titanic-explo-lab-'


# In[3]:

# Getting the titanic data
titanic_df = pd.read_csv('../data/titanic_data.csv')


# ### Some information about the data.

# In[4]:

# Some information about the data
titanic_df.info()


# >**Note: **From here we can see that there are about **891 entries** with a total of **12 columns**. The data types for each column can be observed as well. The variable descriptions as obtained from Kaggle is showed from this text file *[data_descriptions.txt](../data/data_descriptions.txt)*.

# In[5]:

# Looking at some entries of the data
titanic_df.head(6)


# ## 1.2 Data Cleaning
# Looking at the the data under the `Pclass`, `Sex`, and `Embarked` columns, the values didn't represent well their meaning. And since, I'm going to use the data in visualizing some of the analysis, it is much better to change them here.

# ### Cleaning the `Pclass`, `Sex`, and `Embarked` Columns
# For the `Pclass` column, I would put the respective socio-economic status value for each numeric value. *(1= Upper Class, 2 = Middle Class, 3 = Lower Class)*. Meanwhile, for the `Sex` column, I would just capitalize each word of the gender. And finally, for the `Embarked` column, I would put the respective embark locations *(C = Cherbourg; Q = Queenstown; S = Southampton)*.

# In[6]:

# Functions to clean data
def clean_pclass(df_col):
    '''Returns the string counterpart of the Pclass'''
    p_class = ["Upper Class", "Middle Class", "Lower Class"]
    return p_class[df_col-1]

def clean_embark(df_col):
    '''Returns the whole name of embarked location'''
    if df_col=='C':
        return 'Cherbourg'
    elif df_col=='Q':
        return 'Queenstown'
    else:
        return 'Southampton'
    
def clean_data(df):
    '''Returns the cleaned data frame'''
    df['Pclass'] = df['Pclass'].apply(clean_pclass)
    df['Sex'] = df['Sex'].apply(str.title)
    df['Embarked'] = df['Embarked'].apply(clean_embark)
    return df


# In[7]:

titanic_df = clean_data(titanic_df)


# In[8]:

# Looking at the data
titanic_df.head()


# > **Note: **We can see that the changes have reflected in the data frame.

# ## 1.3 Data Exploration
# 
# To start with the exploration, I am going to start off with the basic details that the data has to offer.

# ### Counting the Total Male and Female Passengers

# In[30]:

def count_entries(df, col_arr):
    tmp_df = pd.DataFrame()
    for name in col_arr:
        surv = len(df[df['Survived']==1][name])
        not_surv
        tmp_df[name] = 
    return ser


# In[38]:

titanic_df.groupby(['Sex', 'Survived'], as_index=False).count()


# ### Counting the NaN/null Entries by Column
# Upon looking at the data, I observed that some rows have no entries under the Age and Cabin column. In order to explore the data even further, I looked at each column and check the number of NaN entries in that column.

# In[10]:

# Counts the number of NaN or null entries in each column
titanic_df.isnull().sum()


# > **Note: **It can be observed that only the `Age` and `Cabin` columns have missing values. I would later go into this.

# Remember that the `Age` and `Cabin` columns have some `NaN` or missing values in some of its rows. Upon reading some online resources, I found out that there are options on how to deal with these rows. 
# 
# First option is by dropping the rows that have no entries under the `Age` and `Cabin` column, but after observing that a large number of rows have no entries under the respective columns, it would greatly affect my dataset if I were to drop these rows. 
# 
# Hence my second option, using some basic tools to predict the values of `Age` and `Cabin` columns based on other rows' value.

# ### Filling-up the `Age` Column
# Since the Age column has the least number of blank entries, I decided to fill this column first. In filling up the missing age, I could have gotten the median age by sex, and just subtitute this median age for every missing value based on the sex of that row. But, I have observed from the data that the title of names (i.e. Mr, Ms, Master, etc.) could also have something to do with age, especially that the title `Master` seemed to be associated with children. Hence, I decided to use these title, in order to predict or fill the missing age.

#     Scratch: Thinking of just finding the median age by sex and putting the median age on the missing values. BUT! Why not try extracting the title of the name (i.e. Mr., Mrs, etc) and then find the median age based on the sex and the title.
#     
#     Steps:
#     1. Split surname by using , as delimiter
#     2. Create a list of titles from the data
#     3. Unify these titles

# ### Splitting the `Name` Column
# Using the `Name` column of the data frame, I would split the passenger's name and add three additional columns: `Title`, `Firstname`, `Surname` the name's title, first name, and surname, respectively.

# In[11]:

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


# In[12]:

# Adding the name sections to the data frame
titanic_df = titanic_df.join([s_title, s_firstname, s_surname])


# In[13]:

# Confirming that the columns have been added
titanic_df.head()


# ### Organizing the `Title` Column

# Now that the additional columns have been added, I then would like to look at the different titles that I would be working with. Upon looking at the different titles, I decided to unify some of the titles in order to fill the missing ages according to the passenger's title.

# In[14]:

# Looking at the different titles in the Data
titanic_df['Title'].unique()


# In[15]:

# Looking at the different titles with missing ages
titanic_df[titanic_df['Age'].isnull()]['Title'].unique()


# In[16]:

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


# In[17]:

titanic_df['UniTitle'] = titanic_df['Title'].apply(unify_title)


# ### Filling the Missing Ages

# In filling up the missing ages, I wouldn't want to modify the original data, hence I would make a new column `Filled Age` that would contain all passenger's age. Also, since the mean is greatly affected by outliers in the data set, I created a boxplot of `Ages` by `Title`, in order to see the presence of outliers.

# In[18]:

# Plotting the box plot per Unititle
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax = sns.boxplot(x='UniTitle', y='Age', data=titanic_df)

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xlabel('Title', fontsize=13)
ax.set_ylabel('Age', fontsize=13)
ax.set_title('Box Plot of Ages by Title', fontsize=14)

fig.savefig(fig_prefix+'box_plot_ages_by_title')


# > **Note: **Since there are some outliers, the mean `Age` for the `Titles`: `Mr` and `Miss`, would be affected. Hence, I decided to use the median instead.

# In[19]:

# A DataFrame that contains the median age for each title
median_age_by_title = titanic_df.groupby('UniTitle').median().round(2)['Age']
median_age_by_title


# In[20]:

# Creating the new column that would contain the complete age data
titanic_df['Filled Age'] = titanic_df['Age']   

# Filling the missing ages
for i in range(len(titanic_df)):
    if np.isnan(titanic_df.loc[i, 'Filled Age']):
        titanic_df.loc[i, 'Filled Age'] = median_age_by_title.loc[titanic_df.loc[i, 'UniTitle']]


# In[21]:

# Looking at the new data frame
titanic_df[['Age', 'UniTitle', 'Filled Age']].head(6)


# In[22]:

# Visualizing the two distributions
fig = plt.figure(figsize=(10,6))

# Plotting the Histogram of Age (where we drop the NA rows), and Filled Age
plt.hist([titanic_df['Age'].dropna(), titanic_df['Filled Age']], label=['Age','Filled Age'], bins=15)

# Plot Customizations
plt.legend(loc='best')
plt.xlabel('Age', fontsize=13)
plt.ylabel('Distribution', fontsize=13)
plt.title('Distribution of Age and Filled Age Column', fontsize=14)

plt.savefig(fig_prefix+'distribution_age_and_fill_age')


# >**Note: **Looking at this figure, it seemed that the difference between the `Age` and `Filled Age` column can be seen at the age bin of around 30. Did most of the `NaN` or null values dropped have `UniTitle` of `Mr`? Specifically, is most of the missing `Age` values are Male?

# ### Another look into the missing `Age` values

# In[23]:

# Missing Age Values by Gender
print (titanic_df[titanic_df['Age'].isnull()]['Sex']=='Male').sum()   # Number of missing age values from male
print (titanic_df[titanic_df['Age'].isnull()]['Sex']=='Female').sum() # Number of missing age values from female


# > **Note: **From this, we could see that a large portion of the missing `Age` values are indeed Male, and since the `Title Mr` have been given to the majority of Male passengers, it is understandable that there would be a huge difference at the age bin of 30, which is the median age of `UniTitle Mr`.

# ### Filling-up the `Cabin` Column

# The `Cabin` column seemed to be missing a lot of values. To be specific, 687 rows have no values under the `Cabin` column. Since there are only about 891 entries in the data frame, almost 77% of the values under the `Cabin` column is missing. Hence, I have decided not to fill this column, and proceed with the analysis part of the data.

# # 2 Analysis
# 
# In this part, I would go further into the analysis of factors that could have affected a person's survivability in the Titanic incident.

# ## 2.1 Data Visualization

# ### Survivability by Sex
# As per reports, most of the 700 survivors of Titanic are female, and children. I wanted to confirm this by looking into visualizations of the data.

# In[24]:

# Looking into survivability by sex
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Sex', y='Survived', data=titanic_df, estimator=np.sum, ci=0)

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xlabel('Sex', fontsize=13)
ax.set_ylabel('No. of Survivors', fontsize=13)
ax.set_title('Survivability by Sex', fontsize=14)

fig.savefig(fig_prefix+'survivability_by_sex')


# > **Note: ** From the figure above, we could see that most of the survivors are indeed female.

# ### Survivability by Class
# Looking at a passenger's socio-economic status and survivability.

# In[25]:

# Looking into survivability by Class
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Pclass', y='Survived', data=titanic_df, estimator=np.sum, ci=0,
                 order=['Upper Class', 'Middle Class', 'Lower Class'])

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xlabel('Socio-Economic Status', fontsize=13)
ax.set_ylabel('No. of Survivors', fontsize=13)
ax.set_title('Survivability by Class', fontsize=14)

fig.savefig(fig_prefix+'survivability_by_class')


# > **Note: ** From here, we could see that the socio-economic status may also have played a role in a passenger's survivability.

# ### Survivability by Sex in each Class
# Using the two graphs before, let's look at the survivability by gender in each class.

# In[26]:

# Looking into survivability by Class
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df, estimator=np.sum, ci=0)

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xlabel('Socio-Economic Status', fontsize=13)
ax.set_ylabel('No. of Survivors', fontsize=13)
ax.set_title('Survivability by Sex in each Class', fontsize=14)

fig.savefig(fig_prefix+'survivability_by_sex_and_class')


# > **Note:** Using the two graphs before, it is not surprising that for each of the class, the number of female survivors are greater than the male survivors; and that the Upper Class had higher survivability compared to the other two classes.

# ### Survivability by Age

# In[27]:

# Looking into the survivability distrubition by age
fig = plt.figure(figsize=(13,7))

# Plotting the distribution
# For Survived
ax1 = fig.add_subplot(211)
ax1= sns.distplot(titanic_df[titanic_df['Survived']==1]['Filled Age'], label="Survived", color='g', bins=15)
ax1.set_title('Distribution of Age (Survived)', fontsize=13)
ax1.set_xlabel('Age', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set(xlim=(1,titanic_df['Filled Age'].max()))

# For Not Survived
ax2 = fig.add_subplot(212)
ax2 = sns.distplot(titanic_df[titanic_df['Survived']==0]['Filled Age'], label="Not Survived", color='b', bins=15)
ax2.set_title('Distribution of Age (Not Survived)', fontsize=13)
ax2.set_xlabel('Age', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set(xlim=(1,titanic_df['Filled Age'].max()))

plt.tight_layout()

fig.savefig(fig_prefix+'survivability_by_age')


# In[ ]:




# In[ ]:



