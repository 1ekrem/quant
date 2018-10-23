from quant.lib.main_utils import *
from quant.lib import timeseries_utils as tu
from quant.data import stocks
from quant.scripts import proactive


def load_for_ticker(ticker, database=stocks.DATABASE_NAME, table_name=stocks.UK_FINANCIALS):
    logger.info('Loading %s' % ticker)
    url = proactive.get_proactive_url(ticker)
    if url is not None:
        soup = proactive.get_proactive_finantials_page(url)
        data = proactive.get_proactive_financial_data(soup)
        for k, v in data.iteritems():
            if v is not None:
                v.name = ticker
                tu.store_timeseries(v, database, table_name, k)


def load_uk_financials():
    u = stocks.get_ftse_smx_universe()
    u2 = stocks.get_ftse250_universe()
    u3 = stocks.get_ftse_aim_universe()
    u = pd.concat([u, u2.loc[~u2.index.isin(u.index)]], axis=0, sort=False)
    u = pd.concat([u, u3.loc[~u3.index.isin(u.index)]], axis=0, sort=False)
    for ticker in u.index:
        try:
            load_for_ticker(ticker)
        except:
            logger.warn('Failed for %s' % ticker)


def main():
    load_uk_financials()


if __name__ == '__main__':
    main()
    