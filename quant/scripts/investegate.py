from quant.lib.main_utils import *
from quant.lib.web_utils import *


def get_investegate_finalcials_page(ticker, page=None):
    url = 'https://www.investegate.co.uk/CompData.aspx?code=%s&tab=fundamentals' % ticker
    if page is not None:
        url += '&curr=%d' % page
    return get_page(url)


def get_table_content(x, by='tr'):
    return [d.text.replace('\n', '') for d in x.find_all(by)]


def get_merge_table(soup):
    t = soup.find_all('table')
    contents = [get_table_content(x) for x in t]
    l = np.median([len(x) for x in contents])
    ans = [x for x in contents if len(x) == l]
    ans = pd.DataFrame(ans)
    ans.index = ans.iloc[:, 0]
    ans.index.name = ans.iloc[0, 0]
    ans.columns = ans.iloc[0]
    ans = ans.iloc[1:, 1:]
    for i in xrange(len(ans.columns)):
        if 'Interim Accounts' in ans.columns[i]:
            ans2 = ans.iloc[:, i:]
            ans2.index = ans2.iloc[:, 1]
            ans2 = ans2.iloc[:, 2:]
            ans = ans.iloc[:, :i]
            break
    ans = ans.loc[[x for x in ans.index if len(x) == 10]]
    ans2 = ans2.loc[[x for x in ans2.index if len(x) == 10]]
    ans.index = pd.DatetimeIndex(ans.index)
    ans2.index = pd.DatetimeIndex(ans2.index)
    return ans.sort_index(), ans2.sort_index()


def _to_data(s):
    ans = []
    for x in s.values:
        try:
            ans.append(np.float(x))
        except:
            ans.append(np.nan)
    return pd.Series(ans, index=s.index)


def load_investegate_contents(ticker):
    ans = None
    ans2 = None
    for page in [None, 8, 16]:
        soup = get_investegate_finalcials_page(ticker, page)
        if soup is not None:
            a, a2 = get_merge_table(soup)
            if ans is None:
                ans = a
            else:
                ans = pd.concat([a[a.index < ans.index[0]], ans], axis=0)
            if ans2 is None:
                ans2 = a2
            else:
                ans2 = pd.concat([a2[a2.index < ans2.index[0]], ans2], axis=0)
    data = {}
    if ans is not None and ans2 is not None:
        for t in ['Turnover', 'Operating Profit', 'EPS Diluted']:
            for c in ans.columns:
                if t in c:
                    tmp = _to_data(ans.loc[:, c])
                    tmp.name = ticker
                    data[t] = tmp
            for c in ans2.columns:
                if t in c:
                    tmp = _to_data(ans2.loc[:, c])
                    tmp.name = ticker
                    data['Interim ' + t] =  tmp
    return data