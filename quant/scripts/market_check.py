from quant.lib.main_utils import *
from quant.data import stocks
from quant.research import cross
from datetime import datetime as dt, timedelta


def get_week_table(r):
    rtn = r.iloc[-1].dropna().sort_values()
    rtn = (np.exp(rtn) - 1.).to_frame()
    rtn = pd.concat([rtn.iloc[:10], rtn.iloc[-10:]], axis=0)
    rtn.loc[:, 'Week'] = ['%.1f%%' % (100. * x) for x in rtn.iloc[:, 0]]
    rtn.index.name = 'Ticker'
    return rtn[['Week']]


def get_reversal_table(rm, score, p, lookback=4):
    rtn = pd.concat([rm.rolling(lookback, min_periods=1).mean().iloc[-1],
                     rm.rolling(52, min_periods=13).mean().iloc[-lookback-1],
                     score.loc[rm.columns],
                     p.loc[rm.columns]], axis=1)
    rtn.columns = ['Rev', 'Mom', 'EMom', 'Pos']
    rtn = rtn[(rtn.EMom > 0) & (rtn.Rev < 0)].sort_values('Rev', ascending=True).iloc[:20]
    rtn.index.name = 'Ticker'
    return np.round(rtn, 1)


def get_momentum_table(rm, score, score2, p, lookback=4):
    rtn = pd.concat([rm.rolling(lookback, min_periods=1).mean().iloc[-1], 
                     rm.rolling(52, min_periods=13).mean().iloc[-lookback-1],
                     score.loc[rm.columns],
                     score2.loc[rm.columns],
                     p.loc[rm.columns]], axis=1)
    rtn.columns = ['Rev', 'Mom', 'EF', 'ES', 'Pos']
    rtn = rtn[(rtn.ES > 0) & (rtn.Mom > 0)].sort_values('Mom', ascending=False).iloc[:20]
    rtn.index.name = 'Ticker'
    return np.round(rtn, 1)
    

def run_check():
    t4 = stocks.load_google_returns(data_table=stocks.UK_ESTIMATES, data_name='T4').iloc[-1]
    t8 = stocks.load_google_returns(data_table=stocks.UK_ESTIMATES, data_name='T8').iloc[-1]
    t52 = stocks.load_google_returns(data_table=stocks.UK_ESTIMATES, data_name='T52').iloc[-1]
    rtn, rm, vol, _ = cross.get_dataset('SMX')
    p = .2 / vol.iloc[-1]
    table = get_week_table(rtn)
    table2 = get_reversal_table(rm.loc[:, t4.reindex(rm.columns) > 0], t4, p)
    table7 = get_reversal_table(rm.loc[:, t8.reindex(rm.columns) > 0], t8, p, 8)
    table5 = get_momentum_table(rm.loc[:, t52.reindex(rm.columns) > 0], t4, t52, p)
    rtn, rm, vol, _ = cross.get_dataset('FTSE250')
    p = .2 / vol.iloc[-1]
    table3 = get_week_table(rtn)
    table4 = get_reversal_table(rm.loc[:, t4.reindex(rm.columns) > 0], t4, p)
    table8 = get_reversal_table(rm.loc[:, t8.reindex(rm.columns) > 0], t8, p, 8)
    table6 = get_momentum_table(rm.loc[:, t52.reindex(rm.columns) > 0], t4, t52, p)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'Market Watch')
    mail.add_date(dt.today())
    mail.add_text('SMX Week')
    mail.add_table(table, width=300)
    mail.add_text('SMX Reversal')
    mail.add_table(table2, width=350)
    mail.add_text('SMX 2M Reversal')
    mail.add_table(table7, width=350)
    mail.add_text('SMX Momentum')
    mail.add_table(table5, width=400)
    mail.add_text('FTSE250 Week')
    mail.add_table(table3, width=300)
    mail.add_text('FTSE250 Reversal')
    mail.add_table(table4, width=350)
    mail.add_text('FTSE250 2M Reversal')
    mail.add_table(table8, width=350)
    mail.add_text('FTSE250 Momentum')
    mail.add_table(table6, width=400)
    mail.send_email()


def main():
    run_check()


if __name__ == '__main__':
    main()
