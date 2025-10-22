import numpy as np
import torch as t


def coef(x, x2, mask):
    return 1

def cal_property():


def predict(seq_pad: np.ndarray, n_valid: int):
    # todo: add prediction model here
    # naive predict
    seq_pad[n_valid]=seq_pad[n_valid-1]

    return seq_pad

def process(seq: np.ndarray, n_pred):   
    """
    
    """
    n_prefill=seq.shape[-1]
    x_mean=np.mean(seq,-1)
    # pre process
    x_normal=np.divide( np.subtract(seq, x_mean), x_mean)
    # predict
    x_pad=np.pad(x_normal, (0, n_pred), 'constant', constant_values=(0, 0))
    # print(x_pad)
    nvalid=x_normal.shape[-1]
    for i in range(n_pred):
        predict(x_pad, nvalid+i)   
    # post process
    rec=np.add(np.multiply(x_pad, x_mean), x_mean)
    return rec

data=np.array([ x for x in range (32)])
# print(data)
ret=process(data, 4)
print(ret)