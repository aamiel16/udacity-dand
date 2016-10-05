
# coding: utf-8

# # Data Wrangling
# Data analysts spends almost 70% of their time data wrangling. Data Wrangling is the process of **gathering**, **extracting**,**cleaning**, and **storing data**. Only after this process it is reasonable to start with the analysis.

# In[49]:

# Imports and Configurations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# For reading csv
import unicodecsv

# For reading xls
import xlrd

get_ipython().magic('matplotlib inline')
data_dir = '../data/'
fig_prefix = '../figures/10-04-2016-as-data-wrangling-lab-'


# ## Data Extraction Fundamentals

# ### Parsing CSV File

# In[45]:

# File Location
data_file = 'beatles-diskography.csv'
file_dir = os.path.join(data_dir, data_file)


# In[46]:

def parse_file(datafile):
    '''A Function that Returns a the Parsed File as List of Dictionaries'''
    with open(datafile, 'rb') as f_pointer:
        data = list(unicodecsv.DictReader(f_pointer))
    return data[0:10]


# In[48]:

# Using Pandas to read the csv would result an error at line 10 of the CSV file
# data_df = pd.read_csv('../data/beatles-diskography.csv')
data = parse_file(file_dir)


# ### Parsing Excel File

# In[51]:

datafile = os.path.join(data_dir,"2013_ERCOT_Hourly_Load_Data.xls")

def parse_file(datafile):
    workbook = xlrd.open_workbook(datafile)
    sheet = workbook.sheet_by_index(0)

    data = [[sheet.cell_value(r, col) for col in range(sheet.ncols)] for r in range(sheet.nrows)]

    print "\nList Comprehension"
    print "data[3][2]:",
    print data[3][2]

    print "\nCells in a nested loop:"    
    for row in range(sheet.nrows):
        for col in range(sheet.ncols):
            if row == 50:
                print sheet.cell_value(row, col),


    ### other useful methods:
    print "\nROWS, COLUMNS, and CELLS:"
    print "Number of rows in the sheet:", 
    print sheet.nrows
    print "Type of data in cell (row 3, col 2):", 
    print sheet.cell_type(3, 2)
    print "Value in cell (row 3, col 2):", 
    print sheet.cell_value(3, 2)
    print "Get a slice of values in column 3, from rows 1-3:"
    print sheet.col_values(3, start_rowx=1, end_rowx=4)

    print "\nDATES:"
    print "Type of data in cell (row 1, col 0):", 
    print sheet.cell_type(1, 0)
    exceltime = sheet.cell_value(1, 0)
    print "Time in Excel format:",
    print exceltime
    print "Convert time to a Python datetime tuple, from the Excel float:",
    print xlrd.xldate_as_tuple(exceltime, 0)

    return data

data = parse_file(datafile)


# #### Exercise on Excel Files
# Your task is as follows:
# - read the provided Excel file
# - find and return the min, max and average values for the COAST region
# - find and return the time value for the min and max entries
# - the time values should be returned as Python tuples
# 
# Please see the test function for the expected return format

# In[78]:

def parse_file(datafile):
    workbook = xlrd.open_workbook(datafile)
    sheet = workbook.sheet_by_index(0)

#   coast_col = [sheet.cell_value(r,1) for r in range(sheet.nrows)] # one way of doing it
    coast_col = sheet.col_values(1, start_rowx=1, end_rowx=None)
    
    data = {}
    
    # Finding the max, min, avg in coast column
    data['maxvalue'] = max(coast_col)
    data['minvalue'] = min(coast_col)
    data['avgcoast'] = np.mean(coast_col)
    
    # Finding maxtime and mintime
    loc_max = np.argmax(coast_col)
    loc_min = np.argmin(coast_col)
    
    date_max = sheet.cell_value(loc_max+1, 0)
    date_min = sheet.cell_value(loc_min+1, 0)
    
    data['maxtime'] = xlrd.xldate_as_tuple(date_max, 0)
    data['mintime'] = xlrd.xldate_as_tuple(date_min, 0)
    
    print sheet.cell_value(loc_max, 1)
    
    return data

data = parse_file(datafile)
print data


# In[79]:

def test():
    data = parse_file(datafile)

    assert data['maxtime'] == (2013, 8, 13, 17, 0, 0)
    assert round(data['maxvalue'], 10) == round(18779.02551, 10)

test()


# In[85]:

list_dict = [{'name':'aa'}, {'name':'bb'}]
print list_dict[0].'name'


# In[ ]:



