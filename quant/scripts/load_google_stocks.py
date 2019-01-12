from quant.lib.main_utils import *
from quant.data import stocks


def load_yahoo():
    stocks.import_uk_yahoo_prices()


def main():
    target = 'Full'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if target == 'Full':
        load_yahoo()
    elif target == 'Missing':
        stocks.import_uk_yahoo_prices(missing=True)


if __name__ == '__main__':
    main()