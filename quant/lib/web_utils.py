from quant.lib.main_utils import *
import requests
from bs4 import BeautifulSoup
from googlesearch import search


HEADERS = requests.utils.default_headers()
HEADERS.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})


def get_page(url):
    page = requests.get(url, headers=HEADERS).content
    soup = BeautifulSoup(page)
    return soup


def google(question, max_page=20):
    return search(question, stop=max_page)

