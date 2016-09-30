
# coding: utf-8

# # Titanic Exploratory Data Analysis
# 
# ### Author: Adrian Amiel Sanchez

# <a id='sections' />

# ## Sections:
# 
#    - <a href=#section1> 1 Introduction </a>
#        - <a href=#section1.1> 1.1 Data Wrangling </a>
#        - <a href=#section1.2> 1.2 Data Cleaning </a>
#        - <a href=#section1.3> 1.3 Filling the Missing Values </a>
#        
#        
#    - <a href=#section2> 2 Data Exploration & Visualizations </a>
#        - <a href=#section2.1> 2.1 Looking into the Cabin Column </a>
#        - <a href=#section2.2> 2.2 Looking into Survival </a>
#        - <a href=#section2.3> 2.3 Looking into Sex/Gender </a>
#        - <a href=#section2.4> 2.4 Looking into Class/Socio-Economic Status </a>
#        - <a href=#section2.5> 2.5 Looking into Age </a>
#        - <a href=#section2.6> 2.6 Looking into Embarked Location </a>
#        - <a href=#section2.7> 2.7 Looking into Family Size </a>
#        
#        
#    - <a href=#section3> 3 Summary & Conclusion </a>  

# <a id='section1' />

# # 1 Introduction
# 
# <p align="justify">The RMS Titanic, one of the largest and most luxurious ocean liners ever built from its time. Spanning about 883 feet from stern to bow, a height of 175 feet, and a massive weight of 46,000 tons, it was made to carry about 3,500 passengers and crew. With its massive build, the ship was thought to be unsinkable. However on April 14, 1912 just before midnight, the ship failed to divert its course from an iceberg, which left at least 5 of its hull compartments ruptured. Due to the shortage of lifeboats, only about 700 passengers survived out of the approximately 2,224 passengers on board. 
# </p>
# 
# <p align="justify">
# This exploratory data analysis is submitted as partial fullfilment of the requirements for Udacity's Data Analyst Nanodegree Program. In this notebook, I shall be going into the step by step process of data analysis, in order to answer some questions regarding the given data. The data to be used in this notebook was provided by Udacity and the description of variables was obtained from Kaggle.
# </p>
# 

# <a id='section1.1' />

# ## 1.1 Data Wrangling

# ### Imports and Loading the Data 

# In[1]:

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurations
get_ipython().magic('matplotlib inline')

# Figure Prefix
fig_prefix = '../figures/final/2015-09-20-as-titanic-explo-final-' 


# In[2]:

# Getting the titanic data
titanic_df = pd.read_csv('../data/titanic_data.csv')


# ### Some information about the data

# In[3]:

# Some information about the data
titanic_df.info()


# >**Analysis: **From here we can see that there are about **891 entries** with a total of **12 columns**. The data types for each column can be observed as well. The variable descriptions as obtained from Kaggle is showed from this text file *[data_descriptions.txt](../data/data_descriptions.txt)*.

# In[4]:

# Looking at some entries of the data
titanic_df.head()


# <a id='section1.2' />

# ## 1.2 Data Cleaning
# Looking at the the data,  the values didn't represent well their meaning under the `Pclass`, `Sex`, and `Embarked` columns. Since, I'm going to use the data in visualizing some of the analysis, it is much better to change them here.

# ### Cleaning the Pclass, Sex, and Embarked Columns
# For the `Pclass` column, I would put the respective socio-economic status value for each numeric value *(1= Upper Class, 2 = Middle Class, 3 = Lower Class)*. Meanwhile, for the `Sex` column, I would just capitalize each word of the gender. And finally, for the `Embarked` column, I would put the respective embark locations *(C = Cherbourg; Q = Queenstown; S = Southampton)*.

# In[5]:

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


# In[6]:

titanic_df = clean_data(titanic_df)


# In[7]:

# Looking at the cleaned data
titanic_df.head()


# > **Note: **We can see that the changes have reflected in the data frame but it can be observed that the `Age` and `Cabin` column have some missing values.

# ### Is there other columns with `NaN` entries?
# Upon looking at the data, I observed that some rows have no entries under the Age and Cabin column. Is there other columns with missing values? In order to explore the data even further, I looked at each column and check the number of NaN entries in that column.

# #### Counting the `NaN` entries per column

# In[8]:

# Counts the number of NaN or null entries in each column
titanic_df.isnull().sum()


# > **Analysis: **It can be seen that only the `Age` and `Cabin` column have missing values. Specifically, there are 177 missing entries under the `Age` column, whereas 687 entries are missing under the `Cabin` column.

# <a id='section1.3' />

# ## 1.3 Filling the Missing Values

# Remember that the `Age` and `Cabin` columns have some `NaN` or missing values in some of its rows. Upon reading some online resources, I found out that there are options on how to deal with these rows. 
# 
# First option is by dropping the rows that have no entries under the `Age` and `Cabin` column, but after observing that a large number of rows have no entries under the respective columns, it would greatly affect my dataset if I were to drop these rows. Hence my second option, using some information from rows with values under the `Age` and `Cabin` columns, I would predict the missing values of these columns.

# ### Splitting the Name Column
# In filling up the missing age, I could have gotten the median age by sex, and just subtitute this median age for every missing value based on the sex of that row. But, I have observed that the title of names (i.e. Mr, Ms, Master, etc.) could also have something to do with age, especially that the title `Master` seemed to be associated with children. Hence, I decided to use these title, in order to predict or fill the missing age.
# 
# Using the `Name` column of the data frame, I would split the passenger's name and add three additional columns: `Title`, `Firstname`, `Surname` for the name's title, first name, and surname, respectively.

# In[9]:

# Splitting the surname from the rest of the name
s_surname = titanic_df['Name'].str.split(',', expand=True)

# Splitting the title from the rest of the name
s_title = s_surname[1].str.split('.', expand=True)

# Putting the name sections in series
s_firstname = s_title[1].str.strip()   # Removes the whitespace before the str then assigns to s_firstname
s_title = s_title[0].str.strip()       # Removes the whitespace before the str then assigns to s_title
s_surname = s_surname[0]

# # Editing the column name
s_firstname = s_firstname.rename('Firstname')
s_title = s_title.rename('Title')
s_surname = s_surname.rename('Surname')


# In[10]:

# Adding the name sections to the data frame
titanic_df = titanic_df.join([s_title, s_firstname, s_surname])


# ### Organizing the Title Column

# Now that the additional columns have been added, I then would like to look at the different titles that I would be working with. Upon looking at the different titles, I decided to unify some of the titles in order to fill the missing ages according to the passenger's title.

# In[11]:

# Looking at the different titles in the Data
titanic_df['Title'].unique()


# In[12]:

# Looking at the different titles with missing ages
titanic_df[titanic_df['Age'].isnull()]['Title'].unique()


# In[13]:

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


# In[14]:

# Adding a new column for the unified title
titanic_df['UniTitle'] = titanic_df['Title'].apply(unify_title)


# In[15]:

# Confirming that the columns have been added
titanic_df.head()


# ### Filling the Age Column

# In filling up the missing ages, I wouldn't want to modify the original data, hence I would make a new column `Filled Age` that would contain all passenger's age. Also, since the mean is greatly affected by outliers in the data set, I created a boxplot of `Ages` by `UniTitle` in order to see if there are presence of outliers.

# #### Looking for outliers in the Age of each Unified Title

# In[16]:

# Plotting the box plot per Unititle
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax = sns.boxplot(x='UniTitle', y='Age', data=titanic_df)

# Plot Customizations
ax.tick_params(labelsize=12)
ax.set_xlabel('Title', fontsize=13)
ax.set_ylabel('Age', fontsize=13)
ax.set_title('Box Plot of Ages by UniTitle', fontsize=14)
plt.tight_layout()

fig.savefig(fig_prefix+'box_plot_ages_by_title')


# > **Note: **Since there are some outliers, the mean `Age` for the `Titles`: `Mr` and `Miss`, would be affected. Hence, I decided to use the median instead.

# #### Creating and filling the Filled Age Column

# In[17]:

# Creating a series that contains the median age for each title
median_age_by_title = titanic_df.groupby('UniTitle').median().round(2)['Age']
median_age_by_title


# In[18]:

# Creating the new column that would contain the complete age data
titanic_df['Filled Age'] = titanic_df['Age']

# Filling the missing ages
# Loop on each row, then look if Filled Age is null then fill by the value in the median_age_by_title
for i in range(len(titanic_df)):
    if np.isnan(titanic_df.loc[i, 'Filled Age']):
        titanic_df.loc[i, 'Filled Age'] = median_age_by_title.loc[titanic_df.loc[i, 'UniTitle']]


# In[19]:

# Looking at the filled values
# Just remove .head() to see every missing age and the corresponding filled age
titanic_df[titanic_df['Age'].isnull()][['Age', 'UniTitle', 'Filled Age']].head()  


# > **Note: ** We could see that for every NaN values, it have been filled based on the row's UniTitle column.

# ### Filling the Cabin Column

# The `Cabin` column seemed to be missing a lot of values. To be specific, 687 rows have no values under the `Cabin` column. Since there are only about 891 entries/rows in the data frame, almost 77% of the values under the `Cabin` column is missing. Hence, I have decided not to fill this column, and proceed with the analysis part of the data.

# <a id='section2' />  

# # 2 Data Exploration & Visualizations
# In this part, I would go through some analysis and visualizations about the data. Questions would be posed by the start of each part, and then answered by exploring the data and using some visualizations.
# 

# In[20]:

# Function used to customize the figures or visualizations
def plot_customize(ax, title=None, xlabel=None, ylabel=None):
    '''This Function customizes the title, xlabel, ylabel of the given axes'''
    ax.tick_params(labelsize=12)
    if xlabel!=None:
        ax.set_xlabel(xlabel, fontsize=13)
    if ylabel!=None:
        ax.set_ylabel(ylabel, fontsize=13)
    if title!=None:
        ax.set_title(title, fontsize=14)
        
    plt.tight_layout()


# <a id='section2.1' />  

# ## 2.1 Looking into the Cabin Column
# A lot of the values under the Cabin column was missing, and from this I was unable to fill the missing values. In this part, we are going to look at the missing values under the Cabin column, and determine a possible reason behind this. In this part, we are going to answer the following questions:
#    - Are all passengers that have missing values under the Cabin column, didn't survived? If not, how many did survived?
#    - What gender have the most missing values under the Cabin column?
#    - What class/socio-economic status have the most missing values under the Cabin column?

# ### Are all passengers that have missing values under the Cabin column, didn't survived? If not, how many did survived?
# One possible explanation on why these values were missing was that the passengers didn't survived. Hence, lets take a look if this can be seen in the data.

# In[21]:

# This dataframe would contain the data of every rows with missing cabin values
missing_cabin = titanic_df[titanic_df['Cabin'].isnull()]


# In[22]:

print missing_cabin.groupby('Survived').count()['PassengerId']
# Plotting 
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Survived', y='Survived', data=missing_cabin, estimator=lambda x:len(x))

# Plot Customization
plot_customize(ax, title='Distribution of Survivability of Passengers \n with Missing Cabin Values', 
               ylabel='Distribution', xlabel='Survived')
ax.set_xticklabels(["Didn't Survived", 'Survived'])

fig.savefig(fig_prefix+'dist_of_survivability_miss_cab_val')


# > **Analysis: **From there we could see that  some of the passengers, with missing cabin values, survived. But a majority of the passengers, with missing cabin values, did not survived.

# ### What gender have the most missing values under the Cabin column?

# In[23]:

print missing_cabin.groupby('Sex').count()['PassengerId']

# Plotting
ax = (missing_cabin.groupby('Sex').count()['PassengerId']).plot.pie(figsize=(7,7), autopct='%.f%%', fontsize=14)
ax.yaxis.set_visible(False)
plot_customize(ax, title='Passengers with Missing Cabin Values \n Gender Composition')

plt.savefig(fig_prefix+'pie_miss_cab_by_gender')


# > **Note: **From the pie graph above, we could see that majority of the Male passengers are missing the cabin values.

# ### What class/socio-economic status have the most missing values under the Cabin column?

# In[24]:

miss_cab_grouped_by_pclass = missing_cabin.groupby('Pclass', as_index=False).count()

# Plotting
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Pclass', y='PassengerId', data=miss_cab_grouped_by_pclass)

# Plot Customization
plot_customize(ax, title='Passengers with Missing Cabin Values \n Class Composition', 
               xlabel='Socio-Economic Status', ylabel='Distribution')

fig.savefig(fig_prefix+'dist_miss_cab_by_pclass')


# > **Analysis: **We could see that the most passengers with missing cabin values belongs to the lower class.

# <a id='section2.2' />  

# ## 2.2 Looking into Survival
# In this part, we shall start looking into the survivability of passengers, and other factors that could affect survivability. But before going deeper, let us answer one of the basic questions.

# ### How many passengers survived and did not survived?

# In[25]:

# Looking into how many passengers survived and did not survived
grouped_by_survived = titanic_df.groupby(['Survived'], as_index=True).count()['PassengerId']
grouped_by_survived.index = ["Not Survived", 'Survived',]
grouped_by_survived


# >**Analysis: **From here we could see that there are only 342 survivors out of the 891 passengers. 

# #### Visualizing the Division of Passengers who Survived and Did Not Survived

# In[26]:

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

# Plotting the passenger composition by gender
grouped_by_survived.plot.pie(subplots=True, fontsize=14, ax=ax, autopct='%.0f%%', radius=1)

# Plot Customization
ax.yaxis.set_visible(False)
ax.legend(loc='best', fontsize=11)
plot_customize(ax, title='Passenger Composition by Survived')


fig.savefig(fig_prefix+'passenger_comp_by_survived')


# >**Analysis: **It can be observed that only about 38% of the passengers survived. Specifically, only 342 out of the 549 passengers, survived. So, does the 342 passengers that survived have some things in common?

# <a id='section2.3' />  

# ## 2.3 Looking into Sex/Gender
# In this part, we are going to look into the passengers' gender. Some questions we are looking to answer under this part are:
#    - How many of the passengers are male? Female?
#    - How many survived from both of these genders? 
#    - How many did not survived from both of these genders?
#    - Does a passenger's gender affects his/her survivability?

# ### How many of the passengers are male? Female?

# In[27]:

grouped_by_sex = titanic_df.groupby(['Sex'], as_index=True).count()['PassengerId']
grouped_by_sex


# >**Analysis: **From here we could see that there are 314 female passengers and 577 male passengers in the data. 

# #### Visualizing the Passenger Composition by Gender

# In[28]:

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

# Plotting the passenger composition by gender
grouped_by_sex.plot.pie(subplots=True, fontsize=14, ax=ax, autopct='%.0f%%')

# Plot Customizations
ax.yaxis.set_visible(False)
ax.set_title('Passenger Composition by Gender', fontsize=16)

plt.tight_layout()

fig.savefig(fig_prefix+'passenger_comp_by_sex')


# >**Analysis: **From the pie graph above, we could see that about 65% of the passengers are male, while other the 35% are female.

# ### How many survived and did not survived from each gender?
# We were able to look at the composition of passengers by gender. But, how many from each gender have survived? That didn't survived?

# In[29]:

# Creating a dataframe that would summarize the survival rate from each gender
grouped_by_sex_survival = pd.DataFrame(titanic_df.groupby(['Sex', 'Survived'], as_index=True).count()['PassengerId'])
grouped_by_sex_survival.columns = ['Count']
grouped_by_sex_survival.index.set_levels(['Not Survived', 'Survived'], level=1, inplace=True)
grouped_by_sex_survival


# >**Analysis: **For the female passengers, 233 survived out of the 314 female passengers. Whereas, for the male passengers, only 109 survived out of the 468 male passengers.

# #### Visualizing the Survivability by Gender

# In[30]:

# Visualizing Survivability by Gender
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Plotting
grouped_by_sex_survival.ix['Female'].plot.pie(subplots=True, fontsize=14, ax=ax1, autopct='%.0f%%')
grouped_by_sex_survival.ix['Male'].plot.pie(subplots=True, fontsize=14, ax=ax2, autopct='%.0f%%', startangle=-100)

# Plot Customization
fig.suptitle('Survivability in each Gender', fontsize=16)

plot_customize(ax1, xlabel='Female')
ax1.legend_.remove()
ax1.yaxis.set_visible(False)

plot_customize(ax2, xlabel='Male')
ax2.legend(fontsize=12)
ax2.yaxis.set_visible(False)

fig.savefig(fig_prefix+'survival_by_each_gender')


# >**Analysis: ** Each pie graph shows the survivability from each gender. We could confirm that almost 74% of the female passengers survived, while only 19% of the male passengers survived.

# ### Does a passenger's gender affect his/her survivability?
# From the figures above, it begs to question that: 'Does a passenger's gender has something to with his/her survivability?'

# In[31]:

# Looking into survivability by sex
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Sex', y='Survived', data=titanic_df, estimator=np.sum, ci=0)

# Plot Customizations
plot_customize(ax, 'Distribution of Survivors by Sex', 'Sex', 'Distribution', )
fig.savefig(fig_prefix+'dist_survivors_by_sex')


# >**Analysis: **From the figure above, we could see that indeed majority of the survivors are female. Even though that only 35% of the passengers are female, majority of the survivors are still female.

# <a id='section2.4' />

# ## 2.4 Looking into the Class/Socio-Economic Status
# In the data, there are three classes or socio-enocomic status present: Upper Class, Middle Class, and Lower Class. Lets look into the socio-economic status of the passengers. Some questions we are hoping to answer in this part are:
#    - How many of the passengers belong to each the classes: upper class, middle class, and lower class?
#    - How many of the upper class, middle class, and lower class passengers survived?
#    - How many of the upper class, middle class, and lower class passengers did not survived?
#    - We saw earlier that most of the survivors are female, is it the same across different class?

# ### How many of the passengers are upper class, middle class, and lower class?

# In[32]:

grouped_by_pclass = titanic_df.groupby(['Pclass'], as_index=True).count()['PassengerId']
grouped_by_pclass


# #### Visualizing the Passenger Composition by Class

# In[33]:

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

# Plotting the passenger composition by gender
grouped_by_pclass.plot.pie(subplots=True, fontsize=14, ax=ax, autopct='%.0f%%')

# Plot Customizations
ax.yaxis.set_visible(False)
ax.legend(loc='best', fontsize=11)
plot_customize(ax, title='Passenger Composition by Class')

fig.savefig(fig_prefix+'passenger_comp_by_pclass')


# >**Analysis: **We could see that about 55% of the passengers belong to the lower class, while 24% belong to the upper class, and the remaining 21% belong to the middle class.

# ### How many survived and did not survived from each class?

# In[34]:

# Looking into how many survived and did not survived in each class
grouped_by_status_survival = titanic_df.groupby(['Pclass', 'Survived'], as_index=False).count()
grouped_by_status_survival = pd.DataFrame(titanic_df.groupby(['Pclass', 'Survived'], as_index=True).count()['PassengerId'])
grouped_by_status_survival.columns = ['Count']
grouped_by_status_survival.index.set_levels(['Not Survived', 'Survived'], level=1, inplace=True)
grouped_by_status_survival


# #### Visualizing the Survivability by Gender

# In[35]:

# Looking into survivability from each class
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

# Plotting
ax = sns.barplot(x='Pclass', y='Survived', hue='Survived', data=titanic_df,
                order=['Lower Class', 'Middle Class',  'Upper Class'], 
                estimator=lambda x:len(x))

# Plot Customization
plot_customize(ax, xlabel='Socio-Economic Status', ylabel='Distribution', title='Distribution of Survivability by Class')
ax.legend(loc='best', fontsize=11)
plt.tight_layout()

fig.savefig(fig_prefix+'dist_survivability_by_pclass')


# >**Analysis: **The bar graph above shows the distribution of passengers that survived and did not survived from each class. From here, we could see that a large number of passengers from the lower class, did not survive. Whereas it seemed that the passengers belonging to the upper class were more likely to survived.

# ### Does the socio-economic satus of a person affects his/her survivability?
# Looking a lot closer into each socio-economic status and the number of survivors.

# In[36]:

# Looking into survivability by Class
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Pclass', y='Survived', data=titanic_df, 
                 order=['Lower Class', 'Middle Class',  'Upper Class'],
                 estimator=np.sum, ci=0)

# Plot Customizations
plot_customize(ax, 'Distribution of Survivors by Class', 'Socio-Economic Status', 'Distribution')

fig.savefig(fig_prefix+'survivors_by_pclass')


# > **Analysis: ** From here, we could see that the socio-economic status may also have played a role in a passenger's survivability. Even though the upper class passengers just make up 24% of all the passengers, majority of the survivors are from this class. Next is the lower class passengers, where they make up almost 55% of the passengers; and lastly, the middle class passengers, that makes up 21% of the passengers.

# ### Survivability by Gender in each Class
# Earlier we saw that majority of the survivors were female. Does that mean that for all three class, female passengers were likely to have survived than male passengers?

# In[37]:

# Looking into survivability by Class
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df, 
                 order=['Lower Class', 'Middle Class',  'Upper Class'],
                 estimator=np.sum, ci=0)

# Plot Customizations
plot_customize(ax, 'Survivability by Sex in each Class', 'Socio-Economic Status', 'No. of Survivors')
ax.legend(loc='best', fontsize=11)

fig.savefig(fig_prefix+'survivability_by_sex_and_class')


# > **Analysis:** Looking at the graph, it can be seen that for each of the class women survived more than men.

# <a id='section2.5' />

# ## 2.5 Looking into Age
# After looking at the the gender, and socio economic status, let us look into the passengers' age. In this part, rather than using the `Age` column, we would be using the `Filled Age` column since this column is complete. Some questions we would want to answer are:
#    - Is there a notable difference between the Age ang Filled Age Column?
#    - What does the distribution of age looks like for those who survived?
#    - What does the distribution of age looks like for those who didn't surivived?
#    - Could we categorize the data even further using the passengers' age?
#        - If so, what would be the passenger composition by the age category?
#        - What is the survivability by age category?

# ### Looking into the Distribution of Age and Filled Age Column
# Since the `Age` Column was missing some values, the `Filled Age` Column was added. And in order to fill in the missing age values, the median for each `UniTitle` was obtained and was used to fill the missing age values. So, what does the distribution looks like before and after filling the missing age values?

# In[38]:

# Visualizing the two distributions
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
# Plotting the histogram
# For the missing ages, we're gonna drop them in this histogram
ax.hist([titanic_df['Age'].dropna(), titanic_df['Filled Age']], 
         label=['Age','Filled Age'], bins=15)

# Plot Customizations
ax.legend(loc='best')
plot_customize(ax, title='Distribution of Age and Filled Age Column', xlabel='Age', ylabel='Distribution')

fig.savefig(fig_prefix+'distribution_age_and_fill_age')


# >**Analysis: **Looking at this figure, it seemed that between the `Age` and `Filled Age` Column there is a notable difference at the age bin of around 30. Did most of the `NaN` or null values dropped have `UniTitle` of `Mr`? Specifically, is most of the missing age values are Male?

# ### Is the majority of missing age values are male?

# In[39]:

# Missing Age Values by Gender
# This just gets the entries that have null entries, then group them by gender, and then count each Id for that gender
missing_age_by_gender = titanic_df[titanic_df['Age'].isnull()].groupby('Sex').count()['PassengerId']
print missing_age_by_gender

# Plotting
ax = missing_age_by_gender.plot.pie(figsize=(7,7), autopct='%.0f%%', fontsize=14)

# Plot Customizations
ax.yaxis.set_visible(False)
ax.legend(loc='best', fontsize=11)
plot_customize(ax, title='Gender of Missing Age Values')


# > **Analysis: **From this, we could see that a large portion of the missing `Age` values are indeed Male, and since the `Title Mr` have been given to the majority of Male passengers, it is understandable that there would be a huge difference at the age bin of 30, which is the median age of `UniTitle Mr`.

# ### Looking into the Distribution of Age by Survival
# Let's look at the age distribution between those who survived, and those who didn't. What does the distribution of age of those who survived looks like? How about the age distribution of those who didn't survived?

# In[40]:

# Plotting
g = sns.FacetGrid(titanic_df, row="Survived", hue='Survived', 
                  size=3, aspect=4, xlim=(1,titanic_df['Filled Age'].max()))

g.map(sns.distplot, "Filled Age")

# Getting the list of axes
ax_list = g.axes.ravel() 

# Customizing each axes
plot_customize(ax_list[0], ylabel='Density', title="Didn't Survived")
plot_customize(ax_list[1], ylabel='Density', xlabel='Filled Age', title="Survived")

fig.savefig(fig_prefix+'survivability_by_age')


# > **Analysis: **For the passengers that didn't survived, the mode seemed to be around the age bin of 30 yrs old. Whereas for the passengers that survived, the distribution seemed to be multi-modal.

# ### Categorizing Passengers by Age
# Let's try categorizing passengers by age, where a passenger would be considered a child if his/her age is below 16 yrs old. For those passengers whose age is greater than 16 yrs old, they would then be classified as Man or Woman based on his/her gender.

# #### Adding the Age Category column and Categorizing the Passengers

# In[41]:

# Function to Categorize Passengers by Age
def categorize_by_age(df):
    '''This function returns a data frame where it categorizes the Age Category column'''
    if df['Filled Age'] > 16:
        if df['Sex']=='Male':
            df['Age Category'] = 'Man'
        else:
            df['Age Category'] = 'Woman'
    else:
        df['Age Category'] = 'Child'
        
    return df


# In[42]:

# Categorizing each passenger by age
titanic_df = titanic_df.apply(categorize_by_age, axis=1)


# ### How many of the passengers can be considered as a man, a woman, and a child?

# In[43]:

grouped_by_age_cat = titanic_df.groupby('Age Category').count()['PassengerId']
print grouped_by_age_cat

# Plotting
ax = grouped_by_age_cat.plot.pie(figsize=(7,7), autopct='%.0f%%', fontsize=14)

# Plot Customizations
ax.yaxis.set_visible(False)
ax.legend(loc='best', fontsize=11)
plot_customize(ax, title='Passenger Composition by Age Category')

plt.savefig(fig_prefix+'passenger_comp_by_age_cat')


# >**Note: ** From the pie graph, we could see that about 16% of the passengers are children, and majority are men.

# ### What were the survivability from each age category?

# In[44]:

# Plotting the survivability by age category
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

# Plotting & Plot Customizations
ax = sns.barplot(x='Age Category', y='Survived', hue='Survived', data=titanic_df, estimator=lambda x:len(x))
plot_customize(ax, title='Survivability by Age Category', xlabel='Age Category', ylabel='Distribution')
ax.legend(loc='best', fontsize=11)

fig.savefig(fig_prefix+'survivability_by_age_cat')


# >**Analysis: **From the bar plot above, the green bars represents those who survived, while the blue bars for those who didn't. In this graph, we could see that women have higher survival rates compared to the two age categories. For the children, the number of those who survived and didn't survived are almost the same.

# <a id='section2.6' />

# ## 2.6 Looking into Embarked Location
# The RMS Titanic had 3 embarkment location in its voyage from Southampton to New York. It started it maiden voyage from Southampton's White Star Dock and other passengers embarked at Cherbourg, and Queenstown. In this part we are going to look at the 3 different embarkment location and answer the following questions:
#    - How many passengers embarked at Southampton, Cherbourg, and Queenstown?
#    - How many passengers survived from each embarkment location?
#    - How many passengers did not survived from each embarkment location?
#    - What is the distribution of classes from each embarkment location?
#    - What is the distribution of age category (i.e. man, woman, child) from each embarkment location?

# ### How many passengers embarked at Southampton, Cherbourg, and Queenstown?

# In[45]:

grouped_by_embark = titanic_df.groupby('Embarked').count()['PassengerId']
grouped_by_embark


# #### Visualizing the Passenger Composition based on Embarked Location

# In[46]:

# Plotting the passenger composition by embarked location
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

# Plotting
ax = grouped_by_embark.plot.pie(figsize=(7,7), autopct='%.0f%%', fontsize=14, startangle=90)

# Plot Customizations
ax.yaxis.set_visible(False)
ax.legend(loc='best', fontsize=11)
plot_customize(ax, title='Passenger Composition by Embarked Location')

plt.savefig(fig_prefix+'passenger_comp_by_embark')


# >**Analysis: **As we can see in this pie chart, a large percentage of passengers embarked at Southampton. The White Star Dock, where RMS Titanic pulled away to start its maiden voyage, was located at Southampton. Hence, majority of the passengers embarked at this location. Cherbourg was the second embarkment location, and Queenstown was the last since RMS Titanic was not able to complete its voyage towards New York.

# ### How many survived and didn't survived from each embarkment location?

# In[47]:

grouped_by_embark_survival = titanic_df.groupby(['Embarked', 'Survived']).count()['PassengerId']
print grouped_by_embark_survival

# Plotting the survivability by age category
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

# Plotting & Plot Customizations
ax = sns.barplot(x='Embarked', y='Survived', hue='Survived', data=titanic_df, estimator=lambda x:len(x))
plot_customize(ax, title='Survivability by Embarked Location', xlabel='Embarked Location', ylabel='Distribution')
ax.legend(loc='best', fontsize=11)

fig.savefig(fig_prefix+'survivability_by_embark')


# >**Analysis: **From the bar plot above, we could see that a large number of passengers from Southampton didn't survived. But the largest number of survivors, also came from Southampton.

# ### Distribution of Class and Age Category from each Embarkment Location

# In[48]:

g = sns.factorplot(x='PassengerId', y='Pclass', data=titanic_df,
                   hue='Age Category', row='Embarked', order=['Lower Class', 'Middle Class',  'Upper Class'],
                   orient='h', size=3, aspect=4, palette='Set1', legend_out=False,
                   kind="bar", ci=0, estimator=lambda x:len(x))
g.set_xlabels('Distribution')
g.set_ylabels('Socio-economic Status')

plt.tight_layout()

g.savefig(fig_prefix+'dist_class_agecat_by_embark')


# >**Analysis: **From here, we could observe that almost every passenger that embarked at Queenstown belongs to the Lower Class. While for the Upper Class passengers that embarked at Cherbourg, they did not brought their children.

# <a id='section2.7' />

# ## 2.7 Looking into Family Size
# In this part, we are going to look at the survivability of a passenger based on their family size. In the titanic data, the `SibSp` column refers to the number of siblings/spouses aboard, while the `Parch` column refers to the number of parents/children aboard. We are going to add a new column - `Family Size`, that would refer to the number of family members aboard. 
# 
# We are also going to answer the following questions:
#    - What does the distribution of family size looks like?
#    - Is a passenger that is alone more likely to survive than those with families aboard?

# ### Categorizing by Family Size

# #### Creating the Family Size Column

# In[49]:

# Creating a function that would add up the Parch and SibSp, then store the sum in the Family Size column
def get_family_size(df):
    df['Family Size'] = df['Parch'] + df['SibSp']
    
    return df


# In[50]:

titanic_df = titanic_df.apply(get_family_size, axis=1)


# ### What does the distribution of family size looks like?

# In[51]:

# Plotting the distribution of family size
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax = titanic_df['Family Size'].plot.hist()

# Plot Customizations
plot_customize(ax, title='Distribution of Family Size', xlabel='Family Size', ylabel='Distribution')

fig.savefig(fig_prefix+'dist_family_size')


# >**Analysis: **It can be observed that the distribution is somewhat positively skewed. That is, as the family size gets bigger, less and less passengers have this family size.

# ### Is a passenger that is alone more likely to survive than those with families aboard?
# For me to answer this question, I am going to make another column wherein this column would refer to the status wether a passenger is alone or not.

# #### Creating the Family Status Column

# In[52]:

# A function that would create the Family Status Column
def get_family_stat(df):
    if df['Family Size']==0:
        df['Family Status'] = 'Alone'
    else:
        df['Family Status'] = 'Not Alone'
        
    return df


# In[53]:

titanic_df = titanic_df.apply(get_family_stat, axis=1)


# #### Visualizing the Survival by Family Status

# In[54]:

# Plotting
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
sns.barplot(x='Family Status', y='Survived', hue='Survived', data=titanic_df, estimator=lambda x:len(x))

# Plot Customizations
plot_customize(ax, title='Survivability by Family Status', xlabel='Family Status', ylabel='Distribution')
ax.legend(loc='best', fontsize=11)


# > **Analysis: **From the bar graph above, it can be seen that the number of survivors from passengers that were alone and not, are almost the same.

# <a id='section3' />

# # 3 Summary & Conclusion

# As a summary, the titanic data, which was obtained from Udacity, contains **12 columns and 891 entries** about individual passengers. For cleaning the data, under the `Pclass` column, I put the respective socio-economic status value for each numeric value *(1= Upper Class, 2 = Middle Class, 3 = Lower Class)*. For the `Sex` column, I just capitalized each word of the gender. And finally, for the `Embarked` column, I put the respective embark locations *(C = Cherbourg; Q = Queenstown; S = Southampton)*. Upon cleaning the data, we found out that some columns have missing values.
# 
# Of all the columns in the data, only two columns have missing values - the `Age` and `Cabin` column. The `Age` column was filled by creating a new column - `Filled Age`. The `Filled Age` column was created by getting the median of ages by `Unified Title`, and then use this values to fill up the missing age values based on the passenger's title. Meanwhile for the `Cabin` column, due to the huge number of missing values under this column, I have decided to not fill this column; but later found out that most of the passengers with missing values under the cabin column, didn't survived, most are males, and a large number belongs to the lower class.
# 
# For the analysis, we were able to find some of the basic information like only **38% of the passengers survived**, and **58% of the passengers are men**, **30% are women**, and **12% are children**. We also found out that there are numerous factors that could affect wether a passenger would survive or not. Since the RMS Titanic crew followed a maritime tradition to *evacuate women and children first*, it was reflected in the data that **majority of the survivors are female**. It was also seen that a passenger's socio-economic status may also come to play, since a **large number of survivors came from the upper class** even though they only **make up about 24% of the passengers**. We were also able to look at the embarkment locations, and found some interesting informations. One of which was that, **no child belonging to the upper class embarked at Cherbourg**; another is that **most of the passengers that embarked at Queenstown belongs to the lower class**. We were also able to analyze a passenger's family size and found out that the **distribution of family size is positively skewed**.
# 
# Many other approach of analysis could be done with the given data, these are just some of the possible approaches toward the data.

#  ### References:
#  
#  http://www.history.com/this-day-in-history/titanic-sinks
#  
#  https://en.wikipedia.org/wiki/Lifeboats_of_the_RMS_Titanic
#  
#  http://www.discoversouthampton.co.uk/visit/history/titanic
#  
#  https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic
#  
#  https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic
#   
#  http://pandas.pydata.org/pandas-docs/stable/visualization.html
#  
#  https://stanford.edu/~mwaskom/software/seaborn/tutorial/categorical.html

# <a href=#sections><center> Back to Sections </center></a>
