'''
Created on 22 Jun 2017

@author: wayne
'''
import logging
import os
import cPickle as pickle

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger('quant')
logger.setLevel(20)

MODEL_PATH = '/home/wayne/TempWork/models'


def load_pickle(filename):
    ans = None
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            ans = pickle.load(f)
            f.close()
    else:
        logger.warn('Could not find model file %s' % filename)
    return ans


def write_pickle(data, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            f.close()
    except Exception as e:
        logger.warn('Failed to write pickle %s:\n%s' % (filename, str(e)))
