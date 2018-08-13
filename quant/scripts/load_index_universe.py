from quant.lib.main_utils import *
from quant.data import stocks
from quant.scripts import scrape


def load_smx_universe():
    logger.info('Loading SMX universe')
    data = scrape.read_smx_stocks()
    stocks._save_tickers(data, 'SMX')


def load_ftse250_universe():
    logger.info('Loading FTSE250 universe')
    data = scrape.read_ftse250_stocks()
    stocks._save_tickers(data, 'FTSE250')


def load_ftse100_universe():
    logger.info('Loading FTSE100 universe')
    data = scrape.read_ftse100_stocks()
    stocks._save_tickers(data, 'FTSE100')


def main():
    load_smx_universe()
    load_ftse250_universe()
    load_ftse100_universe()


if __name__ == '__main__':
    main()
