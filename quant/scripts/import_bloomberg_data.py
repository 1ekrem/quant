'''
Created on 22 Jun 2017

@author: wayne
'''
import os
import pandas as pd


def read_data():
    xl = pd.ExcelFile(os.path.expanduser('~/TempWork/scripts/data.xlsx'))
    return dict([(k, xl.parse(k)) for k in xl.sheet_names])