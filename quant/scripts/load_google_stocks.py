from quant.data import stocks, alpha


def main():
    stocks.import_ftse250_index_prices()
    stocks.import_exchange_rates()
    stocks.import_smx_google_prices()
    stocks.import_ftse250_google_prices()
    alpha.calculate_uk_alpha(latest=True)


if __name__ == '__main__':
    main()