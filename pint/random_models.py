import numpy as np
import pint.toa
import pint.models
import pint.fitter
import residuals
from pint.phase import Phase
from pint.utils import make_toas, show_cov_matrix
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt

def random(fitter, rs_mean, ledge_multiplier=4, redge_multiplier=4, iter=10, npoints=100):
    params = fitter.get_fitparams_num()
    mean_vector = params.values()
    cov_matrix = (((fitter.resids.unscaled_cov_matrix[0][1:]).T)[1:]).T
    fac = fitter.resids.unscaled_cov_matrix[1][1:]
    
    f_rand = deepcopy(fitter)
    mrand = f_rand.model
    
    #scale by fac
    print('errors', np.sqrt(np.diag(cov_matrix)))
    print('mean vector',mean_vector)
    mean_vector *= fac
    cov_matrix = ((cov_matrix*fac).T*fac).T
    
    minMJD = fitter.toas.get_mjds().min()
    maxMJD = fitter.toas.get_mjds().max()

    x = make_toas(minMJD-((maxMJD-minMJD)*ledge_multiplier),maxMJD+((maxMJD-minMJD)*redge_multiplier),npoints,mrand)
    x2 = make_toas(minMJD,maxMJD,npoints,mrand)
    
    rss=[]
    for i in range(iter):
        #create a set of randomized parameters based on mean vector and covariance matrix
        rparams_num = np.random.multivariate_normal(mean_vector,cov_matrix)
        #scale params back to real units
        for j in range(len(mean_vector)):
            rparams_num[j] /= fac[j]
        rparams = OrderedDict(zip(params.keys(),rparams_num))
        print("randomized parameters",rparams)
        f_rand.set_params(rparams)
        rs = mrand.phase(x,abs_phase=True)-fitter.model.phase(x, abs_phase=True)
        rs2 = mrand.phase(x2, abs_phase=True)-fitter.model.phase(x2, abs_phase=True)
        #from calc_phase_resids in residuals
        #rs -= Phase(rs.int[0],rs.frac[0])
        #rs2 -= Phase(rs2.int[0],rs2.frac[0])
        rs -= Phase(0.0,rs2.frac.mean()-rs_mean)
        rs = ((rs.int+rs.frac).value/fitter.model.F0.value)*10**6
        rss.append(rs)
        #if i < 1:
        #    plt.plot(x.get_mjds(), rs, 'k-', alpha=0.3, label='random' )
        #else:
        #    plt.plot(x.get_mjds(), rs, 'k-', alpha=0.3)
    
    #scale back to actual units
    mean_vector /= fac
    cov_matrix = ((cov_matrix/fac).T/fac)
    
    return x.get_mjds(),rss