from quant.lib.main_utils import *
from quant.lib import timeseries_utils as tu
from quant.data import stocks
from quant.scripts import proactive, investegate, lse


@try_once
def load_proactive_for_ticker(ticker, database=stocks.DATABASE_NAME, table_name=stocks.UK_FINANCIALS):
    logger.info('Loading %s' % ticker)
    t, c = proactive.get_proactive_url_name(ticker)
    if t is not None:
        soup = proactive.get_proactive_finantials_page(t, c)
        data = proactive.get_proactive_financial_data(soup)
        for k, v in data.iteritems():
            if v is not None:
                v.name = ticker
                tu.store_timeseries(v, database, table_name, k)


def load_proactive_financials():
    u = stocks.load_uk_universe()
    for ticker in u.index:
        load_proactive_for_ticker(ticker)


@try_once
def load_investegate_for_ticker(ticker, database=stocks.DATABASE_NAME, table_name=stocks.UK_FINANCIALS):
    logger.info('Loading %s' % ticker)
    data = investegate.load_investegate_contents(ticker)
    for k, v in data.iteritems():
        if v is not None:
            tu.store_timeseries(v, database, table_name, k)
                

def load_investegate_financials():
    u = stocks.load_uk_universe()
    for ticker in u.index:
        load_investegate_for_ticker(ticker)


@try_and_check
def load_lse_id(ticker):
    return lse.get_company_code(ticker)


def load_lse_ids(missing=True):
    u = stocks.load_uk_universe()
    if missing:
        existing = stocks.get_universe('LSE')
        u = u.loc[~u.index.isin(existing.index)]
    ans = pd.Series([])
    i = 0
    for ticker in u.index:
        logger.info('Loading %s' % ticker)
        tmp = load_lse_id(ticker)
        if tmp is None:
            i += 1
        else:
            i = 0
            ans.loc[ticker] = tmp
        if i >= 5:
            break
    if not ans.empty:
        ans.name = 'LSE'
        stocks._save_tickers(ans, 'LSE')


def update_lse_ids():
    ans = pd.Series([])
    ans.loc['ASAI'] = 'GB00BDFXHW57GBGBXSSMM'
    ans.loc['RWI'] = 'GB0007995243GBGBXSSMM'
    ans.loc['HWDN'] = 'GB0005576813GBGBXSTMM'
    ans.loc['INCH'] = 'GB00B61TVQ02GBGBXSTMM'
    ans.loc['NEX'] = 'GB0006215205GBGBXSTMM'
    ans.loc['VCT'] = 'GB0009292243GBGBXSTMM'
    ans.loc['VVO'] = 'GB00BDGT2M75GBGBXSTMM'
    ans.loc['BRK'] = 'GB00B067N833GBGBXAMSM'
    ans.loc['CTH'] = 'GB00B0KWHQ09GBGBXAMSM'
    ans.loc['PAM'] = 'GB00BZB2KR63GBGBXAMSM'
    ans.name = 'LSE'
    stocks._save_tickers(ans, 'LSE')


@try_and_check
def load_lse_bid_ask_spread(code):
    return lse.load_bid_ask_spread(code)


def load_bid_ask_spreads():
    u = stocks.get_universe('LSE')
    today = dt.today()
    today = dt(today.year, today.month, today.day)
    ans = pd.Series([])
    for ticker in u.index:
        logger.info('Loading %s' % ticker)
        tmp = load_lse_bid_ask_spread(u.loc[ticker, 'LSE'])
        if tmp is not None:
            ans.loc[ticker] = tmp
    ans = ans.to_frame().T
    ans.index = [today]
    tu.store_timeseries(ans, stocks.DATABASE_NAME, stocks.UK_STOCKS, 'Spread')



def main():
    target = 'Proactive'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if target == 'Proactive':
        load_proactive_financials()
    elif target == 'Investegate':
        load_investegate_financials()
    elif target == 'LSE':
        load_lse_ids()
    elif target == 'Spread':
        load_bid_ask_spreads()


if __name__ == '__main__':
    main()
    