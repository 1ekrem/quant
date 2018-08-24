from quant.lib.main_utils import *
import urllib2
import requests
from bs4 import BeautifulSoup

HEADERS = requests.utils.default_headers()
HEADERS.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})


def get_page(url):
    #page = urllib2.urlopen(url).read()
    page = requests.get(url, headers=HEADERS).content
    soup = BeautifulSoup(page)
    return soup


def find_hl_stock_names(soup):
    ans = []
    for x in soup.find_all('tr'):
        txt = x.text
        if 'Deal' in txt:
            y = txt.split('\n')
            ans.append([y[1], y[2]])
    if len(ans) > 0:
        return pd.DataFrame(ans, columns=['Ticker', 'Name'])
    else:
        return None


def find_hl_pages(soup):
    ans = None
    for x in soup.find_all('tr'):
        txt = x.text
        if 'Page:' in txt:
            for v in txt.split(' '):
                try:
                    v = np.int(v)
                    if ans is None:
                        ans = v
                    elif ans < v:
                        ans = v
                except:
                    pass
            break
    return ans


def get_hl_stocks_table(url):
    soup = get_page(url)
    pages = find_hl_pages(soup)
    ans = []
    if pages is not None:
        for p in xrange(pages):
            new_url = url + '?page=%d' % (p+1)
            ss = get_page(new_url)
            tmp = find_hl_stock_names(ss)
            if tmp is not None:
                ans.append(tmp)
    if len(ans) > 0:
        ans = pd.concat(ans, axis=0)
        ans.index = ans.Ticker.str.replace('.', '')
        ans['u'] = ans.Name.str.upper()
        for kw in ['FUND', 'TRUST', 'REIT', 'INVEST', 'INV TST', 'FIDELITY', 'ABERDEEN', 'BH ', 'HENDERSON',
                   'JPMORGAN', 'ALPHA', 'BAILLIE GIFFORD', 'CREDIT', 'REAL ESTATE', 'EQUITY', 'GBP',
                   'CAPITAL', 'INFRASTRUCTURE', 'SYNCONA', 'F&C', 'INTERNATIONAL PUBLIC PARTNERSHIPS',
                   'PERSHING SQUARE']:
            ans = ans.loc[~ans.u.str.contains(kw)]
        return ans.Name
    else:
        return None


def read_smx_stocks():
    url = "https://www.hl.co.uk/shares/stock-market-summary/ftse-small-cap"
    return get_hl_stocks_table(url)


def read_ftse250_stocks():
    url = "https://www.hl.co.uk/shares/stock-market-summary/ftse-250"
    return get_hl_stocks_table(url)


def read_ftse100_stocks():
    url = "https://www.hl.co.uk/shares/stock-market-summary/ftse-100"
    return get_hl_stocks_table(url)

