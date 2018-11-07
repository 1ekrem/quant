from quant.lib.main_utils import *
import time
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


@try_again
def load_lse_bid_ask_spread(ticker):
    return lse.load_bid_ask_spread(ticker)


def load_bid_ask_spreads():
    u = stocks.load_uk_universe()
    today = dt.today()
    today = dt(today.year, today.month, today.day)
    ans = pd.Series([])
    for ticker in u.index:
        logger.info('Loading %s' % ticker)
        tmp = lse.load_bid_ask_spread(ticker)
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
    elif target == 'Spread':
        load_bid_ask_spreads()


if __name__ == '__main__':
    main()
    