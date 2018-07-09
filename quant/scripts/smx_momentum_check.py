from quant.lib.main_utils import *
from quant.research import equity


def load_smx_model():
    end_date = dt.today()
    start_date = end_date - relativedelta(years=3)
    sim = equity.MomentumSim(start_date, end_date, start_date, 'SMX', 'SMX', load_model=True)
    return sim