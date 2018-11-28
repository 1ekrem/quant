from quant.lib.main_utils import *
from quant.lib import timeseries_utils as tu
from quant.data import stocks
from quant.scripts import scrape


def load_smx_universe(override=False):
    logger.info('Loading SMX universe')
    data = scrape.read_smx_stocks()
    if override:
        tu.delete_description(stocks.DATABASE_NAME, stocks.STOCKS_DESCRIPTION, column_list=['SMX'])
    stocks._save_tickers(data, 'SMX')


def load_ftse250_universe(override=False):
    logger.info('Loading FTSE250 universe')
    data = scrape.read_ftse250_stocks()
    if override:
        tu.delete_description(stocks.DATABASE_NAME, stocks.STOCKS_DESCRIPTION, column_list=['FTSE250'])
    stocks._save_tickers(data, 'FTSE250')


def load_ftse100_universe(override=False):
    logger.info('Loading FTSE100 universe')
    data = scrape.read_ftse100_stocks()
    if override:
        tu.delete_description(stocks.DATABASE_NAME, stocks.STOCKS_DESCRIPTION, column_list=['FTSE100'])
    stocks._save_tickers(data, 'FTSE100')


def load_aim_universe(override=False):
    logger.info('Loading AIM universe')
    data = scrape.read_aim_stocks()
    if override:
        tu.delete_description(stocks.DATABASE_NAME, stocks.STOCKS_DESCRIPTION, column_list=['AIM'])
    stocks._save_tickers(data, 'AIM')


def main():
    load_smx_universe()
    load_ftse250_universe()
    load_ftse100_universe()
    load_aim_universe()


if __name__ == '__main__':
    main()
