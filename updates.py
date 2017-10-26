# -*- coding: utf-8 -*-
import theano
from utils import th_floatX

def ada_updates(params, grads, rho=0.95, eps=1e-6):
    '''
    Ada-delta algorithm
    reference: http://www.cnblogs.com/neopenx/p/4768388.html
    '''
    # initialization:
    #   dp    : delta params
    #   dp_sqr: (delta params) ** 2
    #   gr_sqr: gradient ** 2
    running_gr     = [theano.shared(p.get_value() * th_floatX(0.)) for p in params]
    running_dp_sqr = [theano.shared(p.get_value() * th_floatX(0.)) for p in params]
    running_gr_sqr = [theano.shared(p.get_value() * th_floatX(0.)) for p in params]
    # update gr
    gr_updates = [(gr_i, new_gr_i) for gr_i, new_gr_i in zip(running_gr, grads)]
    # update gr_sqr
    gr_sqr_updates = [(gr_sqr_i, rho*gr_sqr_i + (1-rho)*gr_i**2) for gr_sqr_i, gr_i in zip(running_gr_sqr, running_gr)]    
    # calculate (delta params) by RMS
    # NOTE: here dp_sqr is from last time calculation, because dp has not be calculated!
    dp = [-gr_i * (dp_sqr_i + eps)**0.5/(gr_sqr_i + eps)**0.5 for gr_i, dp_sqr_i, gr_sqr_i in zip(running_gr, running_dp_sqr, running_gr_sqr)]
    # update dx_sqr
    dp_sqr_updates = [(dp_sqr_i, rho*dp_sqr_i + (1-rho)*dp_i**2) for dp_sqr_i, dp_i in zip(running_dp_sqr, dp)]
    # update params
    param_updates = [(param_i, param_i + dp_i) for param_i, dp_i in zip(params, dp)]
    
    return gr_updates, gr_sqr_updates, dp_sqr_updates, param_updates
