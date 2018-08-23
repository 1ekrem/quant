from quant.lib.main_utils import *
from quant.data import stocks
from datetime import datetime as dt, timedelta


def run_check():
    r = stocks.load_google_returns(dt.today() - timedelta(10), dt.today(), data_table=stocks.UK_STOCKS)
    u = stocks.get_ftse_smx_universe()
    u2 = stocks.get_ftse250_universe()
    rtn = r.loc[:, u.index].resample('W').sum().iloc[-1]
    rtn = rtn.dropna().sort_values().to_frame()
    rtn = pd.concat([rtn.iloc[:10], rtn.iloc[-10:]], axis=0)
    rtn['Week'] = ['%.1f%%' % (100. * x) for x in rtn.iloc[:, 0]]
    rtn.index.name = 'Ticker'
    rtn2 = r.loc[:, u2.index].resample('W').sum().iloc[-1]
    rtn2 = rtn2.dropna().sort_values().to_frame()
    rtn2 = pd.concat([rtn2.iloc[:10], rtn2.iloc[-10:]], axis=0)
    rtn2['Week'] = ['%.1f%%' % (100. * x) for x in rtn2.iloc[:, 0]]
    rtn2.index.name = 'Ticker'
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'Market Watch')
    mail.add_date(dt.today())
    mail.add_text('SMX')
    mail.add_table(rtn.loc[:, ['Week']], width=400)
    mail.add_text('FTSE250')
    mail.add_table(rtn2.loc[:, ['Week']], width=400)
    mail.send_email()


def main():
    run_check()


if __name__ == '__main__':
    main()
