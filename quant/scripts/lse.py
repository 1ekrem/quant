from quant.lib.main_utils import *
from quant.lib.web_utils import *

LSE_PRICE = 'https://www.londonstockexchange.com/exchange/prices-and-markets/stocks/summary/company-summary/%s.html'


def get_company_code(url):
    ans = None
    if '/company-summary/' in url:
        tmp = url.split('/company-summary/')[-1]
        ans = tmp.split('.')[0]
    elif 'fourWayKey=' in url:
        ans = url.split('fourWayKey=')[-1]
    return ans


def get_price_key(symbol):
    c0 = 'London stock exchange'
    c1 = 'www.londonstockexchange.com'
    txt = '%s share price (%s)' % (c0, symbol)
    res = google(txt)
    url = []
    for x in res:
        if c1 in x:
            tmp = get_company_code(x)
            if tmp is not None:
                url.append(tmp)
    url = list(set(url))
    return url


def get_price_page(codes, symbol):
    ans = None
    for u in codes:
        url = LSE_PRICE % u
        soup = get_page(url)
        if symbol in soup.text:
            ans = soup
            break
    return ans


def _to_float(s):
    return np.float(s.replace(',', ''))


def get_bid_ask_spread(soup):
    spread = None
    for x in soup.find_all('tr'):
        t = x.text
        if 'Bid' in t and 'Offer' in t:
            res = [v.text for v in x.find_all('td')]
            bid = None
            offer = None
            for i in xrange(1, len(res)):
                if res[i-1] == 'Bid':
                    bid = _to_float(res[i])
                elif res[i-1] == 'Offer':
                    offer = _to_float(res[i])
            if bid is not None and offer is not None:
                spread = offer / bid - 1.
                break
    return spread


def load_bid_ask_spread(symbol):
    codes = get_price_key(symbol)
    soup = get_price_page(codes, symbol)
    if soup is not None:
        return get_bid_ask_spread(soup)
    else:
        return None