from __future__ import print_function, division
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.models.model_builder as mb
from pint.phase import Phase
from pint.toa import make_toas
#from pint.utils import Ftest
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
from astropy import log
#import pint.random_models
import rand
import ut
#import psr_utils as pu
import astropy.units as u
import os

def add_phase_wrap(toas, model, selected, phase):
    """
    Add a phase wrap to selected points in the TOAs object
    
    Turn on pulse number tracking in the model, if it isn't already
    
    :param selected: boolean array to apply to toas, True = selected toa
    :param phase: phase diffeence to be added, i.e.  -0.5, +2, etc.
    """
    # Check if pulse numbers are in table already, if not, make the column
    if (
        "pn" not in toas.table.colnames
    ):
        toas.compute_pulse_numbers(model)
    if (
        "delta_pulse_number" not in toas.table.colnames
    ):
        toas.table["delta_pulse_number"] = np.zeros(
            len(toas.get_mjds())
        )
                
    # add phase wrap
    toas.table["delta_pulse_number"][selected] += phase
                                                                                                                                                                                                                                                                                    
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

def get_closest_group(all_toas, fit_toas):
    #take into account fit_toas being at an edge(d_left or d_right = 0)
    fit_mjds = fit_toas.get_mjds()
    d_left = d_right = None
    if min(fit_mjds) != min(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() < min(fit_mjds))
        left_dict = {min(fit_mjds) - mjd:mjd for mjd in all_toas.get_mjds()} 
        d_left = min(left_dict.keys())
    
    all_toas.unselect()
    if max(fit_mjds) != max(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() > max(fit_mjds))
        right_dict = {mjd-max(fit_mjds):mjd for mjd in all_toas.get_mjds()} 
        d_right = min(right_dict.keys())
    
    try:    
        all_toas.unselect()
    except IndexError:
        print("all toas have been included")
        return None
    print(d_left, d_right)
    if d_left == None and d_right == None:
        print("all groups have been included")
        return None
    elif d_left == None or (d_right != None and d_right <= d_left):
        all_toas.select(all_toas.get_mjds() == right_dict[d_right])
        return all_toas.table['groups'][0]    
    else:
        all_toas.select(all_toas.get_mjds() == left_dict[d_left])
        return all_toas.table['groups'][0]    
    
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
rs = pint.residuals.Residuals(t, m).phase_resids
xt = t.get_mjds()


#starting toas, should not break up groups
groups = t.get_groups()
print(groups)
a = np.logical_or(groups == 8, groups == 9)
#a = np.logical_and(groups == 8, groups == 8)
#a = np.logical_and(t.get_mjds() > 53510*u.d, t.get_mjds() < 53520*u.d)#groups == 25, groups == 26)
print(a)
t.select(a)
print(t.table['groups'])

last_chi2 = 20000000000
cont = True
while cont:
    do_ftest = True
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
    
    full_groups = pint.toa.get_TOAs(timfile).table['groups']
    selected = [True if group in t.table['groups'] else False for group in full_groups] 
    rs_mean = pint.residuals.Residuals(pint.toa.get_TOAs(timfile), f.model, set_pulse_nums=True).phase_resids[selected].mean()
    f_toas, rss, rmods = rand.random_models(f, rs_mean, iter=12, ledge_multiplier=1, redge_multiplier=3.5)
    
    t_others = pint.toa.get_TOAs(timfile)
    
    print('rs_mean',rs_mean)
    print(t.table['groups'])
    closest_group = get_closest_group(deepcopy(t_others), deepcopy(t))
    print('closest_group',closest_group, type(closest_group))
    if closest_group == None:
        #end the program
        cont = False
        continue
    #get closest group, t_others is t plus that group
    #right now t_others is just all the toas, so can use as all
    a = np.logical_or(a, t_others.get_groups() == closest_group)
    t_others.select(a)
    print(t_others.table['groups'])
    
    model0 = deepcopy(f.model)
    print('0 model chi2', f.resids.chi2_reduced)
    print('0 model chi2_ext', pint.residuals.Residuals(t_others, f.model).chi2_reduced)
    
    for i in range(len(rmods)):
        print('chi2 reduced',pint.residuals.Residuals(t, rmods[i]).chi2_reduced)
        print('chi2 reduced ext', pint.residuals.Residuals(t_others, rmods[i]).chi2_reduced)
        plt.plot(f_toas, rss[i], '-k', alpha=0.6)
    
    print(f.get_fitparams().keys())
    t.unselect()
    
    #plot post fit residuals with error bars
    plt.errorbar(xt.value,
        pint.residuals.Residuals(t, f.model).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
        t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
    #plt.plot(t.get_mjds(), pint.residuals.resids(t,m).time_resids.to(u.us).value, '.r', label = 'pre-fit')
    plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
    plt.xlabel('MJD')
    plt.ylabel('Residual (us)')
    r = pint.residuals.Residuals(t,m).time_resids.to(u.us).value
    #plt.ylim(-125000,12500)
    #plt.ylim(min(r)-200,max(r)+200)
    width = max(f_toas).value - min(f_toas).value
    #plt.xlim(min(f_toas).value-width/2, max(f_toas).value+width/2)
    #plt.xlim(51000,57500)
    #plt.legend()
    plt.grid()
    plt.show()
    
    #get next model by comparing chi2 for t_others
    chi2_ext = [pint.residuals.Residuals(t_others, rmods[i]).chi2_reduced.value for i in range(len(rmods))]
    chi2_dict = dict(zip(chi2_ext, rmods))
    #append 0model to dict so it can also be a possibility
    chi2_dict[pint.residuals.Residuals(t_others, f.model).chi2_reduced.value] = f.model
    min_chi2 = sorted(chi2_dict.keys())[0]
    
    #m is new model
    m = chi2_dict[min_chi2]
    #a = current t plus closest group, defined above
    t.select(a)
        
    #use Ftest to decide whether to add a parameter to new model
    #new model = m
    #new model + new param = m_plus
    m_plus = deepcopy(m)
    f_params = []
    #TODO: need to take into account if param isn't setup in model yet
    for param in m.params:
        if getattr(m, param).frozen == False:
            f_params.append(param)
    if 'RAJ' not in f_params:
        #add RAJ
        getattr(m_plus, 'RAJ').frozen = False
    elif 'DECJ' not in f_params:
        #add decj
        getattr(m_plus, 'DECJ').frozen = False
    elif 'F1' not in f_params:
        #add F1
        getattr(m_plus, 'F1').frozen = False
    elif 'DM' not in f_params:
        #add DM
        getattr(m_plus, 'DM').frozen = False
    else:
        print('F0, RAJ, DECJ, F1, and DM all added')
        do_ftest = False
        #cont = False
    if do_ftest:
        #actually fit with new param over extended points
        f = pint.fitter.WlsFitter(t, m)
        f.fit_toas()
        f_plus = pint.fitter.WlsFitter(t, m_plus)
        f_plus.fit_toas()
        #compare m and m_plus
        m_rs = pint.residuals.Residuals(t, f.model)
        m_plus_rs = pint.residuals.Residuals(t, f_plus.model)
        print(m_rs.chi2.value, m_rs.dof, m_plus_rs.chi2.value, m_plus_rs.dof)
        Ftest = ut.Ftest(float(m_rs.chi2.value), m_rs.dof, float(m_plus_rs.chi2.value), m_plus_rs.dof)
        print(Ftest)
        if Ftest < 0.0005:
            #say model is model_plus (AKA, add the parameter)
            m = deepcopy(m_plus)
     
    #get to this point, have a best fit model with or wthout a new parameter
    if min_chi2 > last_chi2*5:
        #try phase -3 to 3
        print('len t_others',len(t_others.get_mjds()))
        #selected = np.logical_or(t_others.get_groups == closest_group,t_others.get_groups == closest_group)
        selected = np.zeros(len(t_others.get_mjds()), dtype = bool)
        if closest_group == min(t_others.get_groups()):
            selected[0] = True
        else:
            selected[-1] = True
        print(selected)
        t0 = deepcopy(t_others)
        t3 = deepcopy(t_others)
        print(t3.table['delta_pulse_number'])
        add_phase_wrap(t3, model0, selected, 3)
        print(t3.table['delta_pulse_number'])
        t2 = deepcopy(t_others)
        print(t2.table['delta_pulse_number'])
        add_phase_wrap(t2, model0, selected, 2)
        print(t2.table['delta_pulse_number'])
        t1 = deepcopy(t_others)
        print(t1.table['delta_pulse_number'])
        add_phase_wrap(t1, model0, selected, 1)
        print(t1.table['delta_pulse_number'])
        t1n = deepcopy(t_others)
        print(t1n.table['delta_pulse_number'])
        add_phase_wrap(t1n, model0, selected, -1)
        print(t1n.table['delta_pulse_number'])
        t2n = deepcopy(t_others)
        add_phase_wrap(t2n, model0, selected, -2)
        t3n = deepcopy(t_others)
        add_phase_wrap(t3n, model0, selected, -3)
        
        min_dict = {min_chi2: m}
        print(len(min_dict))
        #rerun the loop for chi2s and make big list
        for t_phase in [t3, t2, t1, t1n, t2n, t3n]:
            #get next model by comparing chi2 for t_others
            chi2_ext_phase = [pint.residuals.Residuals(t_phase, rmods[i]).chi2_reduced.value for i in range(len(rmods))]
            chi2_dict_phase = dict(zip(chi2_ext_phase, rmods))
            print(chi2_dict_phase.keys())
            min_chi2_phase = sorted(chi2_dict_phase.keys())[0]
            m = chi2_dict_phase[min_chi2_phase]
            min_dict[min_chi2_phase] = m
        print('min_dict keys',min_dict.keys())
        print(len(min_dict))
        min_chi2 = sorted(min_dict.keys())[0]
        print(min_chi2)
        if min_chi2 > last_chi2*5:
            ''''''
            #skip the point and act as if it wasn't there
            #make it so the loop never happened
        m = min_dict[min_chi2]
    last_chi2 = min_chi2
