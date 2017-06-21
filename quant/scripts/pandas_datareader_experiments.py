'''
Created on 21 Jun 2017

@author: wayne
'''
import os
import urllib

base_url = "http://ichart.finance.yahoo.com/table.csv?s="
output_path = os.path.expanduser('~/TempWork/scripts')


def make_url(ticker_symbol):
    return base_url + ticker_symbol


def make_filename(ticker_symbol):
    return output_path + "/" + ticker_symbol + ".csv"


def pull_historical_data(ticker_symbol):
    try:
        urllib.urlretrieve(make_url(ticker_symbol), make_filename(ticker_symbol))
    except urllib.ContentTooShortError as e:
        outfile = open(make_filename(ticker_symbol), "w")
        outfile.write(e.content)
        outfile.close()


def download_spx(ticker='^GSPC'):
    pull_historical_data(ticker)