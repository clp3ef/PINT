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

def starting_points(toas):
    '''function that given a toa object, returns list of truth arrays for best places to start trying to fit'''
    a_list = []
    t = deepcopy(toas)
    groups = list(t.table['groups'])
    #for group in list(set(groups)):
    #    if groups.count(group) > 1:
    #        a = np.logical_or(t.get_groups()==group,t.get_groups()==group)
    #        print(a)
    #        a_list.append(a)
    #if len(a_list) >= 10:
    #    return a_list
    mjd_dict = OrderedDict()
    mjd_values = t.get_mjds().value
    for i in np.arange(len(mjd_values)):
        mjd_dict[i] = mjd_values[i]
    sorted_mjd_list = sorted(mjd_dict.items(), key=lambda kv: (kv[1], kv[0]))
    indexes = [a[0] for a in sorted_mjd_list]
    mjds = [a[1] for a in sorted_mjd_list]
    print(indexes)
    gaps = np.diff(mjds)
    values = [(indexes[i], indexes[i+1]) for i in range(len(indexes)-1)]
    gap_dict = OrderedDict(zip(gaps, values))
    count = 10 - len(a_list)
    for key in sorted(gap_dict.keys()):
        count -= 1
        a = np.zeros(t.ntoas, dtype=bool)
        indexes = gap_dict[key]
        a[indexes[0]] = True
        a[indexes[1]] = True
        a_list.append(a)
        print(a)
        if count == 0:
            break
    return a_list

def get_closest_group(all_toas, fit_toas):
    #take into account fit_toas being at an edge(d_left or d_right = 0)
    fit_mjds = fit_toas.get_mjds()
    d_left = d_right = None
    if min(fit_mjds) != min(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() < min(fit_mjds))
        left_dict = {min(fit_mjds) - mjd:mjd for mjd in all_toas.get_mjds()} 
        d_left = min(left_dict.keys())
    
    all_toas = deepcopy(base_TOAs)
    if max(fit_mjds) != max(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() > max(fit_mjds))
        right_dict = {mjd-max(fit_mjds):mjd for mjd in all_toas.get_mjds()} 
        d_right = min(right_dict.keys())
    
    all_toas = deepcopy(base_TOAs)

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

t = pint.toa.get_TOAs(timfile)

for a in starting_points(t):
    
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
#    a = np.logical_or(groups == 15, groups == 16)
    #a = np.logical_and(groups == 8, groups == 8)
#    a = np.logical_and(t.get_mjds() > 56031.4*u.d, t.get_mjds() < 56031.5*u.d)#groups == 25, groups == 26)
    print('a for this attempt',a)
    t.select(a)
    print(t.table['groups'])

    last_model = deepcopy(m)
    last_t = deepcopy(t)
    last_a = deepcopy(a)
    base_TOAs = pint.toa.get_TOAs(timfile)
    #only modify base_TOAs with deletions
    last_chi2 = 20000000000
    cont = True

    while cont:
        do_ftest = True
        t_small = deepcopy(t)
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
        
        full_groups = base_TOAs.table['groups']
        selected = [True if group in t.table['groups'] else False for group in full_groups] 
        rs_mean = pint.residuals.Residuals(base_TOAs, f.model, set_pulse_nums=True).phase_resids[selected].mean()
        f_toas, rss, rmods = rand.random_models(f, rs_mean, iter=12, ledge_multiplier=1, redge_multiplier=3.5)
    
        t_others = deepcopy(base_TOAs)
    
        print('rs_mean',rs_mean)
        print(t.table['groups'])
        closest_group = get_closest_group(deepcopy(t_others), deepcopy(t))
        print('closest_group',closest_group, type(closest_group))
        if closest_group == None:
            #end the program
            #print the final model and toas
            #actually save the latest model as a new parfile
            print(m.as_parfile())
            cont = False
            continue
        #get closest group, t_others is t plus that group
        #right now t_others is just all the toas, so can use as all
        a = np.logical_or(a, t_others.get_groups() == closest_group)
        t_others.select(a)
        print(t_others.table['groups'])
    
        model0 = deepcopy(f.model)
        print('0 model chi2', f.resids.chi2)
        print('0 model chi2_ext', pint.residuals.Residuals(t_others, f.model).chi2)
    
        for i in range(len(rmods)):
            print('chi2',pint.residuals.Residuals(t, rmods[i]).chi2)
            print('chi2 ext', pint.residuals.Residuals(t_others, rmods[i]).chi2)
            plt.plot(f_toas, rss[i], '-k', alpha=0.6)
    
        print(f.get_fitparams().keys())
        print(t.ntoas)
        t = deepcopy(base_TOAs)
        print(t.ntoas)
        
        #plot post fit residuals with error bars
        xt = t.get_mjds()
        plt.errorbar(xt.value,
            pint.residuals.Residuals(t, f.model).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
            t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
        #plt.plot(t.get_mjds(), pint.residuals.Residuals(t,m).time_resids.to(u.us).value, '.r', label = 'pre-fit')
        plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
        plt.xlabel('MJD')
        plt.ylabel('Residual (us)')
        r = pint.residuals.Residuals(t,m).time_resids.to(u.us).value
        #plt.ylim(-125000,12500)
        plt.ylim(min(r)-200,max(r)+200)
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
        
        if min_chi2 > 15000:
            #bad starting point, break and try next
            cont = False
            continue
        
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
        #current best fit chi2 (extended points and actually fit for with maybe new param)
        f = pint.fitter.WlsFitter(t, m)
        f.fit_toas()
        chi2_new_ext = pint.residuals.Residuals(t, f.model).chi2.value
        #get to this point, have a best fit model with or without a new parameter
        #actually fit with extra toa and same model
        print("START 2nd FTEST THING")
        print(t_small.ntoas, t.ntoas)
        f = pint.fitter.WlsFitter(t, m)
        f.fit_toas()
        f_small = pint.fitter.WlsFitter(t_small, m)
        f_small.fit_toas()
        #compare t_small (t_last) and t
        t_small_rs = pint.residuals.Residuals(t_small, f_small.model)
        t_rs = pint.residuals.Residuals(t, f.model)
        print(t_rs.chi2.value, t_rs.dof, t_small_rs.chi2.value, t_small_rs.dof) 
        s = float(t_small_rs.chi2.value)
        if t_small.ntoas == 2:
            s = 0.0001
        Ftest = ut.Ftest(float(t_rs.chi2.value), t_rs.dof, s, t_small_rs.dof)
        print(Ftest)
        if Ftest < 0.000000005:
            #shouldnt add the point - try phase wraps then delete
            #try phase -3 to +3
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
            add_phase_wrap(t2, model0, selected, 2)
            t1 = deepcopy(t_others)
            add_phase_wrap(t1, model0, selected, 1)
            t1n = deepcopy(t_others)
            print(t1n.table['delta_pulse_number'])
            add_phase_wrap(t1n, model0, selected, -1)
            print(t1n.table['delta_pulse_number'])
            t2n = deepcopy(t_others)
            add_phase_wrap(t2n, model0, selected, -2)
            t3n = deepcopy(t_others)
            add_phase_wrap(t3n, model0, selected, -3)
        
            min_dict = {chi2_new_ext: m}
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
            m_maybe = deepcopy(min_dict[min_chi2])
            f = pint.fitter.WlsFitter(t, m_maybe)
            f.fit_toas()
            f_small = pint.fitter.WlsFitter(t_small, m_maybe)
            f_small.fit_toas()
            #compare t_small (t_last) and t
            t_small_rs = pint.residuals.Residuals(t_small, f_small.model)
            t_rs = pint.residuals.Residuals(t, f.model)
            print(t_rs.chi2.value, t_rs.dof, t_small_rs.chi2.value, t_small_rs.dof) 
            s = float(t_small_rs.chi2.value)
            if t_small.ntoas == 2:
                s = 0.0001
            Ftest = ut.Ftest(float(t_rs.chi2.value), t_rs.dof, s, t_small_rs.dof)
            print(Ftest)
            if Ftest < 0.000000005:
                #TODO: what if fails on first one and last_t, etc. havent been defined yet?
                t = deepcopy(last_t)
                m = deepcopy(last_model)
                aq = deepcopy(last_a)
                a = [aq[i] if aq[i] == a[i] else 'remove' for i in range(len(a))]
                a = [x for x in a if x != 'remove']
                print(a, len(a))
                groups = t.get_groups()
                print(closest_group, 'closest group')
                bad_point = np.logical_and(groups == closest_group, groups == closest_group)
                t.table = t.table[~bad_point].group_by("obs")
                groups = base_TOAs.get_groups()
                bad_point = np.logical_and(groups == closest_group, groups == closest_group)
                base_TOAs.table = base_TOAs.table[~bad_point].group_by("obs")
                print(base_TOAs.ntoas, t.ntoas)
                #skip the point and act as if it wasn't there
                #make it so the loop never happened
                print("DELETED A GROUP")
            m = min_dict[min_chi2]
        print('AT THE BOTTOM')
        last_chi2 = deepcopy(min_chi2)
        last_model = deepcopy(m)
        last_t = deepcopy(t)
        last_a = deepcopy(a)

    xt = t.get_mjds()
    plt.errorbar(xt.value,
    pint.residuals.Residuals(t, m).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
    t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
    #plt.plot(t.get_mjds(), pint.residuals.Residuals(t,m).time_resids.to(u.us).value, '.r', label = 'pre-fit')
    plt.title("%s Final Post-Fit Timing Residuals" % m.PSR.value)
    plt.xlabel('MJD')
    plt.ylabel('Residual (us)')
    r = pint.residuals.Residuals(t,m).time_resids.to(u.us).value
    #plt.ylim(-125000,12500)
    plt.ylim(min(r)-200,max(r)+200)
    width = max(f_toas).value - min(f_toas).value
    #plt.xlim(min(f_toas).value-width/2, max(f_toas).value+width/2)
    #plt.xlim(51000,57500)
    #plt.legend()
    plt.grid()
    plt.show()
