from quant.lib.main_utils import *
from quant.lib.web_utils import *

PROACTIVE = 'www.proactiveinvestors.co.uk'
URL = 'https://www.proactiveinvestors.co.uk/%s/%s/financials/'


def get_proactive_url_name(symbol):
    txt = 'LON:%s proactiveinvestor financials' % symbol
    res = google(txt)
    ticker = None
    comp_name = None
    for x in res:
        if PROACTIVE in x and symbol in x:
            s = x.split('/')
            for i in xrange(len(s) - 1):
                if 'LON:' in s[i]:
                    ticker = s[i]
                    comp_name = s[i + 1]
            break
    return ticker, comp_name


def get_proactive_finantials_page(ticker, comp_name):
    if ticker is None:
        return None
    else:
        new_url = URL % (ticker, comp_name)
        soup = get_page(new_url)
        return soup


def get_proactive_timeline(x):
    ans = []
    for d in x.find_all('td'):
        t = d.text
        if 'Figures in ' not in t:
            ans.append(dt.strptime(t, '%d/%m/%y'))
    return ans


def get_table_data(x):
    ans = []
    for d in x.find_all('td'):
        try:
            ans.append(np.float(d.text))
        except:
            ans.append(np.nan)
    return ans


def get_proactive_financial_data(soup):
    timeline = None
    sales = None
    ebitda = None
    eps = None
    fcf = None
    ebit = None
    profit = None
    l = 1
    for x in soup.find_all('tr'):
        txt = x.text
        if 'Figures in ' in txt:
            s = get_proactive_timeline(x)
            if len(s) > 1:
                timeline = s
                l = len(s)
        if txt.startswith(u'''\nSales'''):
            s = get_table_data(x)
            sales = s[1:] if len(s) > l else s
        elif txt.startswith(u'''\nEBITDA\n'''):
            s = get_table_data(x)
            ebitda = s[1:] if len(s) > l else s
        elif txt.startswith(u'''\nDiluted EPS\n'''):
            s = get_table_data(x)
            eps = s[1:] if len(s) > l else s
        elif txt.startswith(u'''\nNet increase in cash\n'''):
            s = get_table_data(x)
            fcf = s[1:] if len(s) > l else s
        elif txt.startswith(u'''\nEBIT (Operating Profit)\n'''):
            s = get_table_data(x)
            ebit = s[1:] if len(s) > l else s   
        elif txt.startswith(u'''\nProfit Before Tax\n'''):
            s = get_table_data(x)
            profit = s[1:] if len(s) > l else s
    if timeline is not None:
        sales = pd.Series(sales, index=timeline).sort_index().dropna()
        ebitda = pd.Series(ebitda, index=timeline).sort_index().dropna()
        eps = pd.Series(eps, index=timeline).sort_index().dropna()
        fcf = pd.Series(fcf, index=timeline).sort_index().dropna()
        ebit = pd.Series(ebit, index=timeline).sort_index().dropna()
        profit = pd.Series(profit, index=timeline).sort_index().dropna()
    return {'revenue': sales, 'ebitda': ebitda, 'eps': eps, 'fcf': fcf, 'ebit': ebit, 'profit': profit}
        


            