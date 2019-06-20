#! /usr/bin/env python
from __future__ import print_function, division
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.models.model_builder as mb
from pint.phase import Phase
from pint.utils import make_toas, show_cov_matrix
import numpy as np
from copy import deepcopy
from collections import OrderedDict

#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import astropy.units as u
import os

datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, 'NGC6440E4.par')
timfile = os.path.join(datadir, 'NGC6440E.tim')

# Define the timing model
m = mb.get_model(parfile)

# Read in the TOAs
t = pint.toa.get_TOAs(timfile)
t0 = pint.toa.get_TOAs(timfile)
# Examples of how to select some subsets of TOAs
# These can be un-done using t.unselect()
#
# Use every other TOA
# t.select(np.where(np.arange(t.ntoas) % 2))

# Use only TOAs with errors < 30 us
# t.select(t.get_errors() < 30 * u.us)

# Use only TOAs from the GBT (although this is all of them for this example)
# t.select(t.get_obss() == 'gbt')
name = 'NGC6440E5'
save = False
iter = 10
tmin = 53400
t.select(t.get_mjds() > tmin * u.d)#t = fit
t.select(t.get_mjds() < 54100 * u.d)
t0.select(t0.get_mjds() > 52000 * u.d)#t0 = graphed
t0.select(t0.get_mjds() < 55000 * u.d)
# Print a summary of the TOAs that we have
t.print_summary()

# These are pre-fit residuals
rs = pint.residuals.resids(t0, m).phase_resids
xt = t0.get_mjds()
plt.plot(xt, rs, 'x', label = 'pre-fit')
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (phase)')
plt.grid()
plt.show()

# Now do the fit
print("Fitting...")
f = pint.fitter.WlsFitter(t, m)
print("BEFORE:",f.get_fitparams())
print(f.fit_toas())

#this is total bs, has to be with post fit, there has to be a better way to choose the subset of t0 resids that are t, etc.
q = list(t0.get_mjds())
index = q.index([i for i in t0.get_mjds() if i > tmin*u.d][0])
rs_mean = pint.residuals.resids(t0,f.model).phase_resids[index:index+len(t.get_mjds())].mean()
print('rs_mean',rs_mean)

#get scaling factor
M, params, units, scale_by_F0 = f.get_designmatrix()
fac = M.std(axis=0)[1:]

#get mean vector
params = f.get_fitparams_num()#OrderedDict
print("params",params)
mean_vector = params.values()#vector

# Print some basic params --> get covariance matrix
#remove first row and column
ucov_mat = (((f.resids.unscaled_cov_matrix[1:]).T)[1:]).T
show_cov_matrix(ucov_mat,params.keys(),"Unscaled Cov Matrix",switchRD=False)
show_cov_matrix((((f.resids.scaled_cov_matrix[1:]).T)[1:]).T,params.keys(),"Scaled Cov Matrix",switchRD=True)
print("Mean vector is", mean_vector)
print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))
print("\n Best model is:")
print(f.model.as_parfile())
print(f.model)
print('-'*100)
if save:
    j = open(name+'.par','w')
    j.write(f.model.as_parfile())
    j.close()

'''
#histograms
print("mean vector", mean_vector)
print("errors", np.sqrt(np.diag(ucov_mat)))
#scale by fac for calculation
mean_vector *= fac
ucov_mat = ((ucov_mat*fac).T*fac).T
nums = [[],[],[],[],[],[]]
for i in range(20000):
    a,b,c,d,e,f = np.random.multivariate_normal(mean_vector,ucov_mat)
    nums[0].append(a)
    nums[1].append(b)
    nums[2].append(c)
    nums[3].append(d)
    nums[4].append(e)
    nums[5].append(f)
#scale back to real units
mean_vector /= fac   
for i in range(6):
    nums[i] /= fac[i]
ucov_mat = ((ucov_mat/fac).T/fac)
for i in range(6):
    data = nums[i]
    mean = np.mean(data)
    std = np.std(data)
    plt.hist(nums[i], bins=400)
    plt.title(params.keys()[i]+" mean: "+str(mean)+" std: "+str(std))
    plt.show()
'''
#create a copy of the fitter object (have to copy the fitter (rather than the model) to use set_params)
f_rand = deepcopy(f)
mrand = f_rand.model

#scale by fac
print('errors', np.sqrt(np.diag(ucov_mat)))
print('mean vector',mean_vector, fac)
mean_vector *= fac
ucov_mat = ((ucov_mat*fac).T*fac).T

for i in range(iter):
    params_rand_num = np.random.multivariate_normal(mean_vector,ucov_mat) #vector of covariant random numbers for parameters (be sure mean_vec and cov are in same parameter order
    #scale params back to real units
    for j in range(len(mean_vector)):
        params_rand_num[j] /= fac[j]
    params_rand = OrderedDict(zip(params.keys(),params_rand_num))
    print("randomized parameters in Odict",params_rand)
    f_rand.set_params(params_rand)
    #rs = pint.residuals.resids(t, mrand).time_resids.to(u.us).value
    #rs = mrand.phase(t)-f.model.phase(t)
    #rs = ((rs.int+rs.frac).value/m.F0.value)*10**6
    minMJD = t.get_mjds().min()
    maxMJD = t.get_mjds().max()
    print(minMJD, maxMJD)
    x = make_toas(minMJD-((maxMJD-minMJD)*1),maxMJD+((maxMJD-minMJD)*1.5),100,mrand)
    x.clock_corr_info.update({'include_bipm':False,'bipm_version':'BIPM2015','include_gps':False})
    x2 = make_toas(minMJD,maxMJD,100,mrand)
    x2.clock_corr_info.update({'include_bipm':False,'bipm_version':'BIPM2015','include_gps':False})
    rs = f_rand.model.phase(x,abs_phase=True)-f.model.phase(x, abs_phase=True)
    #print(x.get_mjds())
    print('rand model TZR',f_rand.model.get_TZR_toa(x).get_mjds(),'orig model TZR',f.model.get_TZR_toa(x).get_mjds())
    rs2 = f_rand.model.phase(x2, abs_phase=True)-f.model.phase(x2, abs_phase=True)
    #from calc_phase_resids in residuals
    #rs -= Phase(rs.int[0],rs.frac[0])
    #rs2 -= Phase(rs2.int[0],rs2.frac[0])
    rs -= Phase(0.0,rs2.frac.mean()-rs_mean)
    
    rs = ((rs.int+rs.frac).value/m.F0.value)*10**6
    if i < 1:
        plt.plot(x.get_mjds(), rs, 'k-', alpha=0.3, label='random' )
    else:
        plt.plot(x.get_mjds(), rs, 'k-', alpha=0.3)
        
#scale mean vector and cov back to real units
mean_vector /= fac
ucov_mat = ((ucov_mat/fac).T/fac)

#plot post fit residuals with error bars
plt.errorbar(xt.value,
             pint.residuals.resids(t0, f.model).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
             t0.get_errors().to(u.us).value, fmt='x', label = 'post-fit')
plt.plot(t.get_mjds(), pint.residuals.resids(t,m).time_resids.to(u.us).value, 'x', label = 'fit points')
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.legend()
plt.grid()
plt.show()
       
rs = pint.residuals.resids(t0, m).phase_resids
