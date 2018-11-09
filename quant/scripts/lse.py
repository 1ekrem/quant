from quant.lib.main_utils import *
from quant.lib.web_utils import *
import code

LSE_PRICE = 'https://www.londonstockexchange.com/exchange/prices-and-markets/stocks/summary/company-summary/%s.html'


def get_company_code_from_url(url):
    ans = None
    if '/company-summary/' in url:
        tmp = url.split('/company-summary/')[-1]
        ans = tmp.split('.')[0]
    elif 'fourWayKey=' in url:
        ans = url.split('fourWayKey=')[-1]
    return ans


def get_price_page(code):
    url = LSE_PRICE % code
    return get_page(url)


def check_price_page(soup, symbol):
    check = '(%s' % symbol
    return check in soup.title.text


def get_company_code(symbol):
    c1 = 'www.londonstockexchange.com'
    txt = '%s site:%s' % (symbol, c1)
    res = run_google_search(txt)
    ans = None
    for x in res:
        if c1 in x:
            code = get_company_code_from_url(x)
            if code is not None:
                soup = get_price_page(code)
                check = check_price_page(soup, symbol)
                if check:
                    ans = code
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


def load_bid_ask_spread(code):
    soup = get_price_page(code)
    if soup is not None:
        return get_bid_ask_spread(soup)
    else:
        return None