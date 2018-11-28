from quant.lib.main_utils import *
import requests, urllib, json
from bs4 import BeautifulSoup
from googlesearch import search

HEADERS = requests.utils.default_headers()
HEADERS.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0'
})
GOOGLEURL = 'https://www.google.com/search?client=ubuntu&channel=fs&q=%s&ie=utf-8&oe=utf-8&start=%d'
GOOGLEREPLACE = {'%3F': '?', '%3D': '=', '%3A': ':'}
APIKEY = '37ORV4265X088WZ5'
AVFIELDS = {'TIME_SERIES_DAILY_ADJUSTED': 'Time Series (Daily)'}
TIINGO = 'b6e8d5e094ea73542bf882ca1a3869ce6f78741d'


def get_page(url):
    page = requests.get(url, headers=HEADERS).content
    soup = BeautifulSoup(page)
    return soup


def google(question, max_page=20, pause=2):
    return search(question, stop=max_page, pause=pause)


class GoogleSearch(object):
    def __init__(self, question, pages=1):
        self.question = question.replace(' ', '+').replace(':', '%3A')
        self.pages = pages
        self.run()
    
    def get_url(self, page=0):
        return GOOGLEURL % (self.question, 10 * page)
    
    def run(self):
        self._soup = []
        self.results = []
        for p in xrange(self.pages):
            url = self.get_url(p)
            soup = self.get_search_page(url)
            res = self.analyse_results(soup)
            self._soup.append(soup)
            self.results += res

    def get_search_page(self, url):
        source = requests.get(url, headers=HEADERS)
        return BeautifulSoup(source.text, 'lxml')
    
    def get_clean_link(self, link):
        tmp = link.replace('/url?q=', '')
        ans = tmp.split('&')[0]
        for k, v in GOOGLEREPLACE.iteritems():
            ans = ans.replace(k, v)
        return ans
        
    
    def analyse_results(self, s):
        sections = s.find_all(class_='g')
        results = []
        for sec in sections:
            res = sec.find_all(class_='r')
            if len(res) > 0:
                b = res[0]
                content = b.find_all('a')[0]
                text = content.text
                link = self.get_clean_link(content['href'])
                results.append((text, link))
        return results
    

def run_google_search(question):
    try:
        s = google(question)
        return [x for x in s]
    except:
        s = GoogleSearch(question)
        return [x[1] for x in s.results]


def load_alpha_vantage(ticker, function='TIME_SERIES_DAILY_ADJUSTED', output_size='compact'):
    url = 'https://www.alphavantage.co/query?function=%s&symbol=%s&outputsize=%s&apikey=%s' % (function, ticker, output_size, APIKEY)
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    return data.get(AVFIELDS.get(function))


def load_tiingo(ticker, start_date=dt(2018, 1, 1), end_date=dt.today()):
    headers = {
       'Content-Type': 'application/json',
       'Authorization' : 'Token %s' % TIINGO
       }
    url = 'https://api.tiingo.com/tiingo/daily/%s/prices?startDate=%s&endDate=%s' % (ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    response = requests.get(url, headers=headers)
    return response.json()

