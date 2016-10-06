
# coding: utf-8

# # Jupyter Notebook & SQL Magic

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# ## About the Data

# The database was obtained from ChinookDatabase. The database schema is provided by this <a href='https://chinookdatabase.codeplex.com/wikipage?title=Chinook_Schema&referringTitle=Home'> link </a>
# 
# <img src='http://lh4.ggpht.com/_oKo6zFhdD98/SWFPtyfHJFI/AAAAAAAAAMc/GdrlzeBNsZM/s800/ChinookDatabaseSchema1.1.png' />

# ## Loading the SQL Extension and DataBase

# In[2]:

# Loading the extension
get_ipython().magic('load_ext sql')


# In[3]:

# Connecting the database
get_ipython().magic('sql sqlite:///db/Chinook_Sqlite.sqlite')


# ## Query using Line Magic and Cell Magic SQL

# ### Line Magic and Converting the Query into a DataFrame

# In[4]:

# Inline Query
invoice_quer = get_ipython().magic('sql SELECT * FROM INVOICE')
invoice_df = invoice_quer.DataFrame()


# In[5]:

invoice_df.head()


# ### Cell Magic

# In[6]:

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM TRACK;')


# ## Exercises

# ### Count the number of songs by U2

# In[7]:

get_ipython().magic("sql SELECT count(name) FROM Track WHERE Composer = 'U2';")


# ### Get the maximum total invoice for the country of Spain

# In[8]:

get_ipython().magic("sql SELECT max(Total) FROM Invoice WHERE BillingCountry = 'Spain';")


# ### Get the job title of the employee/s whose lastname is Johnson

# In[9]:

get_ipython().magic("sql SELECT Title FROM Employee WHERE LastName = 'Johnson'")


# ### Top 10 Composers with the most Songs

# In[14]:

get_ipython().run_cell_magic('sql', '', 'SELECT Composer, COUNT(*)\nFROM Track\nGROUP BY Composer\nORDER BY COUNT(*) DESC\nLIMIT 10;')


# ### Tracks in the dataset that are between 2,500,000 and 2,600,000 milliseconds long?

# In[18]:

get_ipython().run_cell_magic('sql', '', 'SELECT Name, Milliseconds\nFROM Track\nWHERE Milliseconds>=2500000 AND Milliseconds<=2600000\nORDER BY Milliseconds')


# ### List Albums either written by Iron Maiden or Amy Winehouse

# In[30]:

get_ipython().run_cell_magic('sql', '', "SELECT Ar.Name, Al.Title\nFROM Album as Al\nJOIN Artist as Ar\nON Ar.ArtistId = Al.ArtistId\nWHERE Ar.Name='Iron Maiden'\nOR Ar.Name='Amy Winehouse'")


# ## Looking at the Invoice Table

# ### Top 3 countries with the highest number of invoices

# In[38]:

get_ipython().run_cell_magic('sql', '', 'SELECT BillingCountry, COUNT(*)\nFROM Invoice\nGROUP BY BillingCountry\nORDER BY COUNT(*) DESC\nLIMIT 3')


# ### Highest paying customer
# Build a query that returns the person who has the highest sum of all invoices, along with their email, first name, and last name.

# In[46]:

get_ipython().run_cell_magic('sql', '', 'SELECT Customer.FirstName, Customer.LastName, Customer.Email, SUM(Invoice.Total) as Total\nFROM Customer\nJOIN Invoice\nON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.CustomerId\nORDER BY Total DESC\nLIMIT 1')


# ### Collect a list of emails containing each of your Rock Music listeners
# Use your query to return the email, first name, last name, and Genre of all Rock Music listeners!
# 
# Return you list ordered alphabetically by email address starting with A.
# 
# Can you find a way to deal with duplicate email addresses so no one receives multiple emails?

# In[53]:

get_ipython().run_cell_magic('sql', '', "SELECT Customer.Email, Customer.FirstName, Customer.LastName, Genre.Name\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nJOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId\nJOIN Track ON InvoiceLine.TrackId = Track.TrackId\nJOIN Genre ON Track.GenreId = Genre.GenreId\nWHERE Genre.Name = 'Rock'\nGROUP BY Customer.Email\nORDER BY Customer.Email ASC")


# ### City with highest invoice total
# Write a query that returns the 1 city that has the highest sum of invoice totals.
# 
# Return both the city name and the sum of all invoice totals.

# In[84]:

get_ipython().run_cell_magic('sql', '', 'SELECT Invoice.BillingCity, SUM(Invoice.Total)\nFROM Invoice\nGROUP BY Invoice.BillingCity\nORDER BY SUM(Invoice.Total) DESC\nLIMIT 1')


# Write a query that returns the BillingCity, total number of invoices associated with that particular genre, and the genre Name.
# 
# Return the top 3 most popular music genres for the city with the highest invoice total (you found this in the previous quiz!)

# In[91]:

get_ipython().run_cell_magic('sql', '', "SELECT Invoice.BillingCity, SUM(Invoice.Total), Genre.Name\nFROM Invoice\nJOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId\nJOIN Track ON Track.TrackId = InvoiceLine.TrackId\nJOIN Genre ON Genre.GenreId = Track.GenreId\nWHERE Invoice.BillingCity = 'Prague'\nGROUP BY Genre.Name\nORDER BY SUM(Invoice.Total) DESC\nLIMIT 3")


# In[93]:

get_ipython().run_cell_magic('sql', '', 'SELECT Invoice.BillingCity, Count(Invoice.Total), Genre.Name\nFROM Invoice\nJOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId\nJOIN Track ON Track.TrackId = InvoiceLine.TrackId\nJOIN Genre ON Genre.GenreId = Track.GenreId\nWHERE Invoice.BillingCity IN (SELECT Invoice.BillingCity\n                                FROM Invoice\n                                GROUP BY Invoice.BillingCity\n                                ORDER BY SUM(Invoice.Total) DESC\n                                LIMIT 1)\nGROUP BY Genre.Name\nORDER BY SUM(Invoice.Total) DESC\nLIMIT 3\n\n')


# In[ ]:



