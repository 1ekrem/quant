from quant.lib.main_utils import *
import pandas as pd
from sklearn.decomposition import PCA as spca


class PCA(object):
    def __init__(self, rtns, n_components=None):
        self.rtns = rtns
        self.mu = self.rtns.mean(axis=0)
        self._rtns = self.rtns.subtract(self.mu, axis=1).fillna(0.)
        self.n_components = n_components
        self.run()
    
    def run(self):
        self.core = spca(self.n_components)
        self.core.fit(self._rtns)
        self.components = pd.DataFrame(self.core.components_, columns=self.rtns.columns)
        self.factors = pd.DataFrame(np.dot(self.rtns, self.components.T), index=self.rtns.index)
        self.variance_explained = pd.Series(self.core.explained_variance_)
        self.variance_ratio = pd.Series(self.core.explained_variance_ratio_)
