import numpy as np
import Global

def rx_message(c_encoded):
    #before sending the signal must be Modulated BPSK.
    c_Tx=np.where(c_encoded==1,-1,1)
    n=c_encoded.shape[0]
    # 0 is the mean
    # std is the standard deviation
    # n is the number of elements to be generated
    channel_noise=np.random.normal(0,Global.std,(n,Global.n))
    c_Rx=np.add(c_Tx,channel_noise)
    return c_Rx