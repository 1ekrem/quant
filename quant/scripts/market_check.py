from quant.lib.main_utils import *
from quant.research import cross
from datetime import datetime as dt, timedelta


def get_week_table(r):
    rtn = r.iloc[-1].dropna().sort_values().to_frame()
    rtn = pd.concat([rtn.iloc[:10], rtn.iloc[-10:]], axis=0)
    rtn.loc[:, 'Week'] = ['%.1f%%' % (100. * x) for x in rtn.iloc[:, 0]]
    rtn.index.name = 'Ticker'
    return rtn[['Week']]


def get_reversal_table(rm):
    rtn = pd.concat([rm.rolling(4, min_periods=1).mean().iloc[-1], rm.rolling(52, min_periods=13).mean().iloc[-5]], axis=1)
    rtn.columns = ['Rev', 'Mom']
    rtn = rtn.sort_values('Rev', ascending=True).iloc[:, :20]
    rtn.loc[:, 'Rev'] = ['%.1f%%' % (100. * x) for x in rtn.Rev]
    rtn.loc[:, 'Mom'] = ['%.1f%%' % (100. * x) for x in rtn.Mom]
    rtn.index.name = 'Ticker'
    return rtn
    

def run_check():
    rtn, rm, vol = cross.get_dataset('SMX')
    table = get_week_table(rtn)
    table2 = get_reversal_table(rm)
    rtn, rm, vol = cross.get_dataset('FTSE250')
    table3 = get_week_table(rtn)
    table4 = get_reversal_table(rm)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'Market Watch')
    mail.add_date(dt.today())
    mail.add_text('SMX')
    mail.add_table(table, width=400)
    mail.add_table(table2, width=600)
    mail.add_text('FTSE250')
    mail.add_table(table3, width=400)
    mail.add_table(table4, width=600)
    mail.send_email()


def main():
    run_check()


if __name__ == '__main__':
    main()
