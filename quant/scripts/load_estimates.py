from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import timeseries_utils as tu
from quant.scripts import reuters


def load_reuters_id(ticker, company_name):
    return reuters.get_reuters_id(ticker, company_name)


def load_reuters_ids(missing=True):
    u = stocks.load_uk_universe().max(axis=1)
    if missing:
        existing = stocks.get_universe('Reuters')
        u = u.loc[~u.index.isin(existing.index)]
    ans = pd.Series([])
    i = 0
    for ticker in u.index:
        logger.info('Loading %s' % ticker)
        tmp = load_reuters_id(ticker, u.loc[ticker])
        if tmp is None:
            i += 1
        else:
            i = 0
            ans.loc[ticker] = tmp
        if i >= 5:
            break
    if not ans.empty:
        ans.name = 'Reuters'
        stocks._save_tickers(ans, 'Reuters')


def update_reuters_ids():
    ans = pd.Series([])
    ans.loc['ASAI'] = 'GB00BDFXHW57GBGBXSSMM'
    ans.name = 'Reuters'
    stocks._save_tickers(ans, 'Reuters')


@try_and_check
def load_reuters(ticker):
    return reuters.load_reuters_estimates(ticker)


def load_reuters_data():
    run_date = get_last_business_day()
    u = stocks.get_universe('Reuters')
    for ticker in u.index:
        logger.info('Loading %s' % ticker)
        tmp = load_reuters(u.loc[ticker, 'Reuters'])
        if tmp is not None:
            for k, v in tmp.iteritems():
                ans = pd.Series([v], name=ticker, index=[run_date])
                tu.store_timeseries(ans, stocks.DATABASE_NAME, stocks.UK_ESTIMATES, k)


def main():
    target = 'Reuters'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if target == 'Reuters':
        load_reuters_data()
    elif target == 'ID':
        load_reuters_ids(True)


if __name__ == '__main__':
    main()