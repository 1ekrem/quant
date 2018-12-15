from quant.lib.main_utils import *
from quant.lib.web_utils import *


def _to_float(x):
    try:
        return np.float(x.replace(',', ''))
    except:
        return np.nan


def get_reuters_estimates_page(ticker):
    url = 'https://uk.reuters.com/business/stocks/analyst/%s.L' % ticker
    return get_page(url)


def check_reuters_page(s):
    res = s.find_all('table')
    return len(res) > 1


def get_reuters_id(ticker, company_name):
    s = get_reuters_estimates_page(ticker)
    if check_reuters_page(s):
        return ticker
    else:
        res = run_google_search('reuters %s' % company_name)
        ans = None
        for x in res:
            if 'reuters.com/' in x:
                for y in x.split('/'):
                    if str(y).endswith('.L'):
                        ans = y[:-2]
                if ans is not None:
                    break
        return ans


def get_reuters_tables(s):
    rating = None
    trend = None
    res = s.find_all('table')
    for x in res:
        if '1-5 Linear Scale' in x.text:
            rating = x
        elif '1 WeekAgo' in x.text and '1 MonthAgo' in x.text:
            trend = x
    return rating, trend


def get_ratings(rating):
    x = rating.find_all('tr')
    ans = dict(zip(x[0].text.split('\n'), x[-1].text.split('\n')))
    c = 3. - _to_float(ans.get('Current'))
    c1 = 3. - _to_float(ans.get('1 MonthAgo'))
    c3 = 3. - _to_float(ans.get('3 MonthAgo'))
    return c, c - c1, c - c3


def calculate_trends(ans):
    c = -1. * ans.iloc[:, 1:].subtract(ans.Current, axis=0)
    c = c.mul(np.sqrt(np.array([52., 12., 6., 1.])), axis=1)
    vol = c.abs()
    vol = (1. / vol[vol > 0].mean(axis=1)).fillna(0.)
    c = c.mul(vol, axis=0)
    return tuple(c.mean(axis=0))


def get_trends(trend):
    x = trend.find_all('tr')
    cols = x[0].text.split('\n')
    ans = []
    for y in x[1:]:
        z = y.text.split('\n')
        if len(z) == len(cols):
            ans.append([_to_float(v) for v in z])
    ans = pd.DataFrame(ans, columns = cols).loc[:, ['Current', '1 WeekAgo', '1 MonthAgo', '2 MonthAgo', '1 YearAgo']]
    return calculate_trends(ans)


def load_reuters_estimates(ticker):
    s = get_reuters_estimates_page(ticker)
    if s is None:
        return None
    else:
        ans = {}
        rating, trend = get_reuters_tables(s)
        if rating is not None:
            c, c1, c2 = get_ratings(rating)
            if ~np.isnan(c):
                ans['Rating'] = c
            if ~np.isnan(c1):
                ans['C1'] = c1
            if ~np.isnan(c2):
                ans['C3'] = c2
        if trend is not None:
            t, t1, t2, t3 = get_trends(trend)
            if ~np.isnan(t):
                ans['T1'] = t
            if ~np.isnan(t1):
                ans['T4'] = t1
            if ~np.isnan(t2):
                ans['T8'] = t2
            if ~np.isnan(t3):
                ans['T52'] = t3
        if len(ans.keys()) == 0:
            ans = None
        return ans
