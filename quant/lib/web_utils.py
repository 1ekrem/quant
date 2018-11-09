from quant.lib.main_utils import *
import requests
from bs4 import BeautifulSoup
from googlesearch import search


HEADERS = requests.utils.default_headers()
HEADERS.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0'
})
GOOGLEURL = 'https://www.google.com/search?client=ubuntu&channel=fs&q=%s&ie=utf-8&oe=utf-8&start=%d'
GOOGLEREPLACE = {'%3F': '?', '%3D': '=', '%3A': ':'}


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
