
# coding: utf-8

# # Data Wrangling and JSON

# ## Sections:
#    - <a href='#intro'>Introduction to JSON</a>
#    - <a href='#body1'>Python and JSON</a>

# <a id='intro' />

# ## Introduction to JSON
# 
# ### What is JSON?
# JSON stands for **Java Script Object Notation**, it is a lightweight text-based open standard designed for human-readable data interchange. 
# 
# In JSON, 
# - data is presented in name-value pairs
# - curly brackets hold the objects and each name is followed by a colon
# - the name/value pairs are seperated by a comma
# - square brackets hold arrays and values are seperated by a comma
# 
# ### Uses of JSON:
# - JSON format is used for serializing and transmitting structured data over network connection.
# - It is mainly used to transmit data between a server and web application. 
# - It acts as a placeholder for the data.
# - It can be used with modern programming languages.
# 
# ### Characteristics of JSON:
# - Readability; easy to read and interpret
# - Lightweight
# - Language Independent
# 
# ### Why use JSON over CSVs?
# It is limited to represent some data in tabular format or in CSVs. In some situations, some data have fields and others have additional subfields, if we were to use CSVs, we may find it hard to represent the given data.

# ### Example of JSON

# ```python
# {
#     "book":[
#         {'id':'01',
#         'language':'Java',
#         'edition':'third',
#         'author':'Herbert Shildt'},
#         {'id':'02',
#         'language':'Python',
#         'edition':'first',
#         'author':'Mark Lutz'}
#     ]
# }
# ```
# > As you can see, JSON is similar to a Python's Dictionary, and to a Python's List 

# <a id='body1' />

# ## Python and JSON

# In[ ]:



