
# coding: utf-8

# # Jupyter Notebook & SQL *(sqlite3 library)*

# ### Importing the library

# In[1]:

# import the sqlite3 library
import sqlite3


# ### Connecting to the Database

# In[2]:

# establishing connection
conn = sqlite3.connect('../develop/db/Chinook_Sqlite.sqlite')

# creating a cursor object
c = conn.cursor()


# In[3]:

# Function to execute a query
def execute_query(query):
    c.execute(query)
    # print rows
    for row in c.fetchall():
        print row
        
# Function to close the connection and cursors
def close_conn():
    c.close()
    conn.close()


# ### Exploring the Database

# #### Top 10 artists with the most songs in the  database

# In[4]:

query = '''
SELECT Artist.Name, Count(Artist.Name)
FROM Track
JOIN Album ON Album.AlbumId = Track.AlbumId
JOIN Artist ON Artist.ArtistId = Album.ArtistId
GROUP BY Artist.Name
ORDER BY Count(*) DESC
LIMIT 10
'''

execute_query(query)


# #### Top 10 genres with missing composers

# In[5]:

query = '''
SELECT Genre.Name, COUNT(*)
FROM Track 
JOIN Genre ON Genre.GenreId = Track.GenreId
WHERE Track.Composer IS NULL
GROUP BY Genre.Name
ORDER BY COUNT(*) DESC
LIMIT 10
'''

execute_query(query)


# #### Closing the Connection and Cursor

# In[6]:

close_conn()


# ### Creating a UdaciousMusic Database

# In[7]:

# Creating a new connection
# If the database doesn't exists, it would be created
conn = sqlite3.connect('../develop/db/UdaciousMusic.db')
c = conn.cursor()


# #### Creating the tables

# ### <center> Album Table Schema </center>
# Columns | Data Type | Primary Key | Foreign Key
# :-----: | :-------: | :---------: | :---------:
# AlbumId | INTEGER   | YES         | NO
# Title   | TEXT      | NO          | NO
# ArtistId| INTEGER   | NO          | YES
# UnitPrice| REAL     | NO          | NO
# Quantity| INTEGER   | NO          | NO

# #### Arist Table

# In[38]:

query = '''
CREATE TABLE IF NOT EXISTS Artist(
    ArtistId INTEGER PRIMARY KEY,
    Name TEXT
);
'''

execute_query(query)


# #### Album Table

# In[8]:

query = '''
CREATE TABLE IF NOT EXISTS Album(
    AlbumId INTEGER PRIMARY KEY,
    Title TEXT,
    ArtistId INTEGER,
    FOREIGN KEY (ArtistId) REFERENCES Artist (ArtistId)
);
'''

execute_query(query)


# 

# In[ ]:



