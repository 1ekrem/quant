from quant.lib.main_utils import *
from quant.lib.web_utils import *


def get_proactive_url(symbol):
    t0 = 'LON:%s' % symbol
    txt = 'LON:%s proactiveinvestor' % symbol
    res = google(txt)
    url = None
    for x in res:
        if x.split('/')[-3] == t0:
            url = x
            break
    return url


def get_proactive_finantials_page(url):
    if url is None:
        return None
    else:
        new_url = url + 'financials/'
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
            pass
    return ans


def get_proactive_financial_data(soup):
    timeline = None
    sales = None
    ebitda = None
    eps = None
    fcf = None
    for x in soup.find_all('tr'):
        txt = x.text
        if 'Figures in ' in txt:
            s = get_proactive_timeline(x)
            if len(s) > 1:
                timeline = s
        if txt.startswith(u'''\nSales\n'''):
            s = get_table_data(x)
            if len(s) > 1:
                sales = s
        if txt.startswith(u'''\nEBITDA\n'''):
            s = get_table_data(x)
            if len(s) > 1:
                ebitda = s
        if txt.startswith(u'''\nDiluted EPS\n'''):
            s = get_table_data(x)
            if len(s) > 1:
                eps = s
        if txt.startswith(u'''\nNet increase in cash\n'''):
            s = get_table_data(x)
            if len(s) > 1:
                fcf = s
    if timeline is not None:
        sales = pd.Series(sales, index=timeline).sort_index()
        ebitda = pd.Series(ebitda, index=timeline).sort_index()
        eps = pd.Series(eps, index=timeline).sort_index()
        fcf = pd.Series(fcf, index=timeline).sort_index()
    return {'revenue': sales, 'ebitda': ebitda, 'eps': eps, 'fcf': fcf}
        


            