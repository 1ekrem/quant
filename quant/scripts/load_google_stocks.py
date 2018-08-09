from quant.data import stocks, alpha


def load_google():
    stocks.import_ftse250_index_prices()
    stocks.import_exchange_rates()
    stocks.import_uk_google_prices()


def load_yahoo():
    stocks.import_ftse250_index_prices_from_yahoo()
    stocks.import_exchange_rates_from_yahoo()
    stocks.import_uk_yahoo_prices()


def calculate_alpha():
    alpha.calculate_uk_alpha(latest=True)


def main():
    load_yahoo()
    calculate_alpha()


if __name__ == '__main__':
    main()