
# coding: utf-8

# # Jupyter Notebook & SQL

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# ### Loading the SQL Extension and DataBase

# In[2]:

# Loading the extension
get_ipython().magic('load_ext sql')


# In[3]:

# Connecting the database
get_ipython().magic('sql sqlite:///db/Chinook_Sqlite.sqlite')


# ### Query using Line Magic and Cell Magic SQL

# #### Line Magic and Converting the Query into a DataFrame

# In[4]:

# Inline Query
invoice_quer = get_ipython().magic('sql SELECT * FROM INVOICE')
invoice_df = invoice_quer.DataFrame()


# In[5]:

invoice_df.head()


# #### Cell Magic

# In[6]:

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM TRACK;')


# ### Exercises

# #### Count the number of songs by U2

# In[7]:

get_ipython().magic("sql SELECT count(name) FROM Track WHERE Composer = 'U2'")


# #### Get the maximum total invoice for the country of Spain

# In[8]:

get_ipython().magic("sql SELECT max(Total) FROM Invoice WHERE BillingCountry = 'Spain'")


# #### Get the job title of the employee/s whose lastname is Johnson

# In[9]:

get_ipython().magic("sql SELECT Title FROM Employee WHERE LastName = 'Johnson'")


# In[10]:

get_ipython().run_cell_magic('sql', '', '')


# In[ ]:



