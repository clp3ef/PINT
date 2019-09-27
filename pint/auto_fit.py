from __future__ import print_function, division
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.models.model_builder as mb
from pint.phase import Phase
from pint.utils import make_toas
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
from astropy import log
import pint.random_models
import astropy.units as u
import os

def get_highest_value_group(toas):
    '''given the toa object, return the group with the most toas'''
    if 'groups' not in toas.table.keys():
        log.warn("No group column in the TOA table.")
        return None
    group_list = list(toas.table['groups'])
    num_list = range(max(group_list)+1)
    len_list = [group_list.count(num) for num in num_list]
    print(num_list, len(num_list))
    print(len_list, len(len_list))
    print(np.argmax(len_list))
    return num_list[np.argmax(len_list)]
    
    
    
datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, 'alg_test.par')
timfile = os.path.join(datadir, 'alg_test.tim')

# read in the initial model
m = mb.get_model(parfile)

# Read in the TOAs
t = pint.toa.get_TOAs(timfile)

# Print a summary of the TOAs that we have
t.print_summary()

# These are pre-fit residuals
rs = pint.residuals.resids(t, m).phase_resids
xt = t.get_mjds()
#plt.plot(xt, rs, 'x', label = 'pre-fit')
#plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
#plt.xlabel('MJD')
#plt.ylabel('Residual (phase)')
#plt.grid()
#plt.show()

#read in the tim and parfiles
#start with groups - single observations - choose the highest density group and start there
#select only those toas and fit for f0
#plot the results

#convert jump flags to params
#m.jump_flags_to_params(t)

#select highest density group
print(t.table['groups'])
t.select(0 <= t.get_groups())
t.select(2 > t.get_groups())
print(t.table['groups'])

last_chi2 = 2000000000000000
cont = True
while cont:
    # Now do the fit
    print("Fitting...")
    f = pint.fitter.WlsFitter(t, m)
    print("BEFORE:",f.get_fitparams())
    print(f.fit_toas())
    
    print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
    print("RMS in phase is", f.resids.phase_resids.std())
    print("RMS in time is", f.resids.time_resids.std().to(u.us))
    print("\n Best model is:")
    print(f.model.as_parfile())
    
    rs_mean = pint.residuals.resids(t, f.model, set_pulse_nums=True).phase_resids.mean()
    print(rs_mean)
    f_toas, rss, rmods = pint.random_models.random_models(f, rs_mean, iter=12)
    t_others = pint.toa.get_TOAs(timfile)
    
    
    print(t.table['groups'])
    num = max(t.table['groups'])
    print('num', num)
    #get closest group, t_others is t plus that group
    t_others.select(t_others.get_groups() > num-4)
    t_others.select(t_others.get_groups() < num+2)
    
    print('0 model chi2', f.resids.chi2_reduced)
    print('0 model chi2_ext', pint.residuals.resids(t_others, f.model).chi2_reduced)
    
    if f.resids.chi2_reduced > 1.5*last_chi2:
        print('last chi2 0 model was', last_chi2)
        print('chi2 is going up, add a new parameter?')
        last_chi2 = 20000000000000
        f_params = []
        for param in m.params:
            if getattr(m, param).frozen == False:
                f_params.append(param)
        print(f_params)
        if 'RAJ' not in f_params:
            #add RAJ
            getattr(m, 'RAJ').frozen = False
        elif 'DECJ' not in f_params:
            #add decj
            getattr(m, 'DECJ').frozen = False
        elif 'F1' not in f_params:
            #add F1
            getattr(m, 'F1').frozen = False
        else:
            print('F0, RAJ, DECJ, and F1 all added')
            cont = False
        continue

    for i in range(len(rmods)):
        print('chi2 reduced',pint.residuals.resids(t, rmods[i]).chi2_reduced)
        print('chi2 reduced ext', pint.residuals.resids(t_others, rmods[i]).chi2_reduced)
        plt.plot(f_toas, rss[i], '-', alpha=0.6)
    
    print(f.get_fitparams().keys())
    t.unselect()
    t.unselect()
    
    #plot post fit residuals with error bars
    plt.errorbar(xt.value,
        pint.residuals.resids(t, f.model).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
        t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
    plt.plot(t.get_mjds(), pint.residuals.resids(t,m).time_resids.to(u.us).value, '.r', label = 'pre-fit')
    plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
    plt.xlabel('MJD')
    plt.ylabel('Residual (us)')
    r = pint.residuals.resids(t,m).time_resids.to(u.us).value
    plt.ylim(min(r)-200,max(r)+200)
    plt.legend()
    plt.grid()
    plt.show()
    
    chi2_ext = [pint.residuals.resids(t_others, rmods[i]).chi2_reduced.value for i in range(len(rmods))]
    chi2_dict = dict(zip(chi2_ext, rmods))
    min_chi2 = sorted(chi2_dict.keys())[0]
    if f.resids.chi2_reduced > 0:
        #prevent 2 points - F0 from messing it up
        last_chi2 = f.resids.chi2_reduced
    
    if min_chi2 < pint.residuals.resids(t_others, f.model).chi2_reduced.value:
        #replace current model with better model
        m = chi2_dict[min_chi2]
        t.select(t.get_groups() >= 0)
        t.select(t.get_groups() < num + 2)
    else:
        print('no better models, add a new parameter?')
        last_chi2 = 20000000000000
        f_params = []
        for param in m.params:
            if getattr(m, param).frozen == False:
                f_params.append(param)
        print(f_params)
        if 'RAJ' not in f_params:
            #add RAJ
            getattr(m, 'RAJ').frozen = False
        elif 'DECJ' not in f_params:
            #add decj
            getattr(m, 'DECJ').frozen = False
        elif 'F1' not in f_params:
            #add F1
            getattr(m, 'F1').frozen = False
        else:
            print('F0, RAJ, DECJ, and F1 all added')
            #cont = False

    
    #for key in sorted(chi2_dict.keys()):
    #    m = chi2_dict[key]
    #    plt.plot(t.get_mjds(), pint.residuals.resids(t,m).time_resids.to(u.us).value, '.', label = 'post-fit')
    #    plt.title(key)
    #    plt.xlabel('MJD')
    #    plt.ylabel('Residual (us)')
    #    plt.legend()
    #    plt.grid()
    #    plt.show()
    
                                                            
                                                            
