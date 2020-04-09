#!/usr/bin/env python
#-W ignore::FutureWarning -W ignore::UserWarning -W ignore:DeprecationWarning
from __future__ import print_function, division
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.models.model_builder as mb
from pint.phase import Phase
import numpy as np
from astropy import log
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
import pint.random_models
import ut
#import psr_utils as pu
import astropy.units as u
import os
import csv 
import operator

log.setLevel("INFO")

__all__ = ["main"]

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
    '''chooses based on closest points together'''
    #TODO: better variable name for the starting list than 'a'
    t = deepcopy(toas)
    mjd_values = t.get_mjds().value
    #iterate over all toas
    #for each toa pair, caluclate the score based on every other toa
    #choose the toas with the highest scores
    
    #make an OrderedDict with (keys = index of each toa) and (items = mjd of each toa)
    mjd_dict = OrderedDict()
    for i in range(len(mjd_values)):
        mjd_dict[i] = mjd_values[i]
    #sort mjd_dict by the mjd values, so indexes are out of order but mjds are chronological (also turns the dict into a list of tuples)
    sorted_mjd_list = sorted(mjd_dict.items(), key=lambda kv: (kv[1], kv[0]))
    
    score_dict = OrderedDict()
    for i in range(1, len(sorted_mjd_list)):
        #iterate through toa pairs, assigning each a score and putting the indexes of the two toas and their score in a new dict
        indexes = tuple([sorted_mjd_list[i-1][0], sorted_mjd_list[i][0]])
        avg_mjd = (sorted_mjd_list[i-1][1] + sorted_mjd_list[i][1])/2
        score = 0
        for j in range(1, len(sorted_mjd_list)):
            #iterate over all other toas, adding up their "moments" to get a final socre for the pair
            if (j == i) or (j == i-1):
                #if either of the points in the pair, skip so dont count self
                continue
            mjd = sorted_mjd_list[j][1]
            dist = np.fabs(avg_mjd - mjd)
            #score is the sum of "moments" of each other toa, so high density area have high scores
            score += 1.0/dist
        #keys = tuples of indexes, items = scores
        score_dict[indexes] = score
    
    sorted_score_list = sorted(score_dict.items(), key=operator.itemgetter(1))
    print(sorted_score_list)
    
    a_list = [] 
    for pair in sorted_score_list[-10:]:
        indexes = pair[0]
        a = np.zeros(len(mjd_values), dtype=bool)
        a[indexes[0]] = True
        a[indexes[1]] = True
        a_list.append(a)
    
    #reverse order so largest score at top
    a_list.reverse()
    #return list of boolean arrays for ten highest score pairs
    return a_list

def get_closest_group(all_toas, fit_toas, base_TOAs):
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

    if d_left == None and d_right == None:
        print("all groups have been included")
        return None, None
    elif d_left == None or (d_right != None and d_right <= d_left):
        all_toas.select(all_toas.get_mjds() == right_dict[d_right])
        return all_toas.table['groups'][0], d_right
    else:
        all_toas.select(all_toas.get_mjds() == left_dict[d_left])
        return all_toas.table['groups'][0], -d_left #sign tracks direction closest point is, so negative seperation means to the left and positive to the right

def Ftest_param(r_model, fitter, param_name):
    '''function to run an Ftest on a model comparing with and without the given parameter, returning the Ftest for that comparison'''
    m_plus_p = deepcopy(r_model)
    toas = deepcopy(fitter.toas)
    getattr(m_plus_p, param_name).frozen = False
    f_plus_p = pint.fitter.WLSFitter(toas, m_plus_p)
    f_plus_p.fit_toas()
    #compare m and m_plus_p
    m_rs = pint.residuals.Residuals(toas, fitter.model)
    m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)
    print(m_rs.chi2.value, m_rs.dof, m_plus_p_rs.chi2.value, m_plus_p_rs.dof)
    Ftest_p = ut.Ftest(float(m_rs.chi2.value), m_rs.dof, float(m_plus_p_rs.chi2.value), m_plus_p_rs.dof)
    print('Ftest'+param_name,Ftest_p)
    return Ftest_p
    
def main(argv=None):
    import argparse
    import sys
    '''required = parfile, timfile'''
    '''optional = starting points, param ranges'''
    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="tim file to read toas from")
    parser.add_argument(
        "--starting_points",
        help="mask array to apply to chose the starting points, groups or mjds",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--maskfile",
        help="csv file of bool array for fit points",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--r_iter", help="Number of predictive models that should be calculated", type=int, default=10
    )
    parser.add_argument(
        "--ledge_multiplier", help="scale factor for how far to plot predictive models to the left of fit points", type=float, default=1.0
    )
    parser.add_argument(
        "--redge_multiplier", help="scale factor for how far to plot predictive models to the right of fit points", type=float, default=3.0
    )
    parser.add_argument(
        "--RAJ_lim", help="minimum time span before Right Ascension (RAJ) can be fit for", type=float, default=7.0
    )
    parser.add_argument(
        "--DECJ_lim", help="minimum time span before Declination (DECJ) can be fit for", type=float, default=20.0
    )
    parser.add_argument(
        "--F1_lim", help="minimum time span before Spindown (F1) can be fit for (default = time for F1 to change residuals by 0.35phase)", type=None, default=None
    )
    parser.add_argument(
        "--Ftest_lim", help="Upper limit for successful Ftest values", type=float, default=0.0005
    )
    parser.add_argument(
        "--RFtest_lim", help="Upper limit for successful RFtest values", type=float, default=0.000000005
    )
    parser.add_argument("--max_wrap", help="how many phase wraps in each direction to try", type=int, default=1)

    args = parser.parse_args(argv)
    
    start_type = None
    if args.starting_points != None:
        start = args.starting_points.split(',')
        try:
            start = [int(i) for i in start]
            start_type = "groups"
        except:
            start = [float(i) for i in start]
            start_type = "mjds"
    
        
    #check that there is a directory to save the algorihtm state in
    if not os.path.exists('alg_saves'):
        os.mkdir('alg_saves')
            
    '''start main program'''
    
    datadir = os.path.dirname(os.path.abspath(str(__file__)))
    parfile = os.path.join(datadir, args.parfile)
    timfile = os.path.join(datadir, args.timfile)
    
    t = pint.toa.get_TOAs(timfile)
    sys_name = str(mb.get_model(parfile).PSR.value)

    print("BEFORE:",args.F1_lim)
    if args.F1_lim == None:
        #adjust F1_lim based on F0 value --> maximum possible F1
        F0 = mb.get_model(parfile).F0.value
        if F0 < 100:
            F1 = 10**-12
        else:
            F1 = 10**-14
        args.F1_lim = np.sqrt(0.35*2/F1)/86400.0    
    print("AFTER:",args.F1_lim)
    #checks there is a directory specific to the system
    if not os.path.exists('alg_saves/'+sys_name):
        os.mkdir('alg_saves/'+sys_name)

    for a in starting_points(t):
        
        # read in the initial model
        m = mb.get_model(parfile)
        
        # Read in the TOAs
        t = pint.toa.get_TOAs(timfile)
        
        # Print a summary of the TOAs that we have
        t.print_summary()
        
        #starting toas
        groups = t.get_groups()
        print(groups)
        if start_type == "groups":
            a = np.logical_or(groups == start[0], groups == start[1])
        elif start_type == "mjds":
            a = np.logical_and(t.get_mjds() > start[0]*u.d, t.get_mjds() < start[1]*u.d)
        
        #use given a from file if exists
        if args.maskfile != None:
            mask_read = open(args.maskfile, 'r')
            data = csv.reader(mask_read)
            a = [bool(int(row[0])) for row in data]
            
        print('a for this attempt',a)
        t.select(a)
        print(t.table['groups'])
        
        last_model = deepcopy(m)
        last_t = deepcopy(t)
        last_a = deepcopy(a)
        base_TOAs = pint.toa.get_TOAs(timfile)
        #only modify base_TOAs with deletions
        cont = True
        iteration = 0
        if args.maskfile != None:
            iteration = int(args.maskfile[-8:].strip('qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHGFDSAMNBVCXZ._'))
            
        while cont:
            iteration += 1
            try_phase = False
            t_small = deepcopy(t)
            # Now do the fit
            print("Fitting...")
            f = pint.fitter.WLSFitter(t, m)
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
            f_toas, rss, rmods = pint.random_models.random_models(f, rs_mean, iter=args.r_iter, ledge_multiplier=args.ledge_multiplier, redge_multiplier=args.redge_multiplier)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            t_others = deepcopy(base_TOAs)
            
            print('rs_mean',rs_mean)
            print(t.table['groups'])
            closest_group, dist = get_closest_group(deepcopy(t_others), deepcopy(t), deepcopy(base_TOAs))
            print('closest_group',closest_group, type(closest_group))
            if closest_group == None:
                #end the program
                #print the final model and toas
                #actually save the latest model as a new parfile
                #print(m.as_parfile())
                cont = False
                continue
                        
            #get closest group, t_others is t plus that group
            #right now t_others is just all the toas, so can use as all
            a = np.logical_or(a, t_others.get_groups() == closest_group)
            t_others.select(a)
            print(t_others.table['groups'])
            #t_others is now the fit toas plus the to be added group of toas, t is just the fit toas
            
            #if closest group is >0.35 phase away in either direction, try both phase wraps
            selected_closest = [True if group == closest_group else False for group in full_groups]
            last_fit_toa_phase = pint.residuals.Residuals(base_TOAs, f.model).phase_resids[selected][-1]
            first_new_toa_phase = pint.residuals.Residuals(base_TOAs, f.model).phase_resids[selected_closest][0]
            diff = first_new_toa_phase - last_fit_toa_phase
            print(last_fit_toa_phase, first_new_toa_phase, diff)
            if np.abs(diff) > 0.35:
                #make loop which goes through and tries every phase wrap given, and spits out a chi2, model, and toas
                #make different toas with closest group phase wrapped to other side and turn on marker
                #make copy of base model and wrap toas
                #choose new model from random models
                #do Ftests
                f_phases = []
                t_phases = []
                t_others_phases = []
                m_phases = []
                chi2_phases = []
                for wrap in range(-args.max_wrap, args.max_wrap+1):
                    #copy models to np.array --> use index -1 because current object will always be one just appended to array (AKA, pos -1)
                    print("Trying phase wrap:", wrap)
                    f_phases.append(deepcopy(f))
                    t_phases.append(deepcopy(base_TOAs))
                    print(t_phases)
                    add_phase_wrap(t_phases[-1], f.model, selected_closest, wrap)
                    t_others_phases.append(deepcopy(t_phases[-1]))
                    t_others_phases[-1].select(a)
                    
                    #plot with phase wrap
                    model0 = deepcopy(f.model)
                    print('0 model chi2', f.resids.chi2)
                    print('0 model chi2_ext', pint.residuals.Residuals(t_others_phases[-1], f.model).chi2)
                    
                    fig, ax = plt.subplots(constrained_layout=True)
            
                    print(rmods[0] == rmods[1])
                    for i in range(len(rmods)):
                        print('chi2',pint.residuals.Residuals(t_phases[-1], rmods[i]).chi2)#t_phases[-1] is full toas with phase wrap
                        print('chi2 ext', pint.residuals.Residuals(t_others_phases[-1], rmods[i]).chi2)#t_others_phases[-1] is sleected toas plus closest group with phase wrap
                        ax.plot(f_toas, rss[i], '-k', alpha=0.6)

                    #print(f.get_fitparams().keys())
                    #print(t_phases[-1].ntoas)
                    #t_phases[-1] = deepcopy(base_TOAs)
                    #t is now a copy of the base TOAs (aka all the toas)
                    #print(t_phases[-1].ntoas)
            
                    #plot post fit residuals with error bars
                    xt = t_phases[-1].get_mjds()
                    ax.errorbar(xt.value,
                        pint.residuals.Residuals(t_phases[-1], model0).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
                        t_phases[-1].get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
                    #plt.plot(t.get_mjds(), pint.residuals.Residuals(t,m).time_resids.to(u.us).value, '.r', label = 'pre-fit')
                    fitparams = ''
                    for param in f.get_fitparams().keys():
                        fitparams += str(param)+' '
                        
                    plt.title("%s Post-Fit Residuals %d P%d | fit params: %s" % (m.PSR.value, iteration, wrap, fitparams))
                    ax.set_xlabel('MJD')
                    ax.set_ylabel('Residual (us)')
                    r = pint.residuals.Residuals(t_phases[-1],model0).time_resids.to(u.us).value
                    #plt.ylim(-800,800)
                    #yrange = (0.5/float(f.model.F0.value))*(10**6)
                    yrange = abs(max(r)-min(r))
                    ax.set_ylim(max(r) + 0.1*yrange, min(r) - 0.1*yrange)
                    width = max(f_toas).value - min(f_toas).value
                    if (min(f_toas).value - 0.1*width) < (min(xt).value-20) or (max(f_toas).value +0.1*width) > (max(xt).value+20):
                        ax.set_xlim(min(xt).value-20, max(xt).value+20)
                    else:
                        ax.set_xlim(min(f_toas).value - 0.1*width, max(f_toas).value +0.1*width)
                    #plt.xlim(53600,54600)
                    #plt.legend()
                    plt.grid()
                    def us_to_phase(x):
                        return (x/(10**6))*f.model.F0.value
                
                    def phase_to_us(y):
                        return (y/f.model.F0.value)*(10**6)
                    
                    secaxy = ax.secondary_yaxis('right', functions=(us_to_phase, phase_to_us))
                    secaxy.set_ylabel("residuals (phase)")
                    
                    plt.savefig('./alg_saves/%s/%s_%03d_P%03d.png'%(sys_name, sys_name, iteration, wrap), overwrite=True)
                    plt.close()
                    #plt.show()
                    #end plotting
                    
                    #repeat model selection with phase wrap f.model should be same as f_phases[-1].model (all f_phases[n] should be the same)
                    chi2_ext_phase = [pint.residuals.Residuals(t_others_phases[-1], rmods[i]).chi2_reduced.value for i in range(len(rmods))]
                    chi2_dict_phase = dict(zip(chi2_ext_phase, rmods))
                    #append 0model to dict so it can also be a possibility
                    chi2_dict_phase[pint.residuals.Residuals(t_others_phases[-1], f.model).chi2_reduced.value] = f.model
                    min_chi2_phase = sorted(chi2_dict_phase.keys())[0]
                    
                    #m is new model
                    m_phases.append(chi2_dict_phase[min_chi2_phase])
                    #a = current t plus closest group, defined above
                    t_phases[-1].select(a)
                
                    #repeat with phase wrap
                    #fit toas with new model
                    f_phases[-1] = pint.fitter.WLSFitter(t_phases[-1], m_phases[-1])
                    f_phases[-1].fit_toas()
                    span = f_phases[-1].toas.get_mjds().max() - f_phases[-1].toas.get_mjds().min()
                    Ftests_phase = dict()
                    f_params_phase = []
                    #TODO: need to take into account if param isn't setup in model yet
                    for param in m_phases[-1].params:
                        if getattr(m_phases[-1], param).frozen == False:
                            f_params_phase.append(param)
                    if 'RAJ' not in f_params_phase and span > args.RAJ_lim*u.d:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        #test RAJ
                        Ftest_R_phase = Ftest_param(m_phases[-1], f_phases[-1], 'RAJ')
                        Ftests_phase[Ftest_R_phase] = 'RAJ'
                    if 'DECJ' not in f_params_phase and span > args.DECJ_lim*u.d:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        #test DECJ
                        Ftest_D_phase = Ftest_param(m_phases[-1], f_phases[-1], 'DECJ')
                        Ftests_phase[Ftest_D_phase] = 'DECJ'
                    if 'F1' not in f_params_phase and span > args.F1_lim*u.d:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        #test F1 Ftest_param(fitter, toas, param_name) -> return ftest value
                        Ftest_F_phase = Ftest_param(m_phases[-1], f_phases[-1], 'F1')
                        Ftests_phase[Ftest_F_phase] = 'F1'
                    print(Ftests_phase.keys())
                    if not bool(Ftests_phase.keys()) and span > 100*u.d:#doesnt actually do anything
                        print("F1, RAJ, DECJ, and F1 have been added. Will only add points from now on")
                        only_add = True
                        #if only_add true, then still check with random models, but instead of rechecking phase wrapped stuff, just choose version fits best and add point with that wrap
                    elif not bool(Ftests_phase.keys()):
                        '''just keep going to next step'''
                    elif min(Ftests_phase.keys()) < args.Ftest_lim:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        add_param = Ftests_phase[min(Ftests_phase.keys())]
                        print('adding param ', add_param, ' with Ftest ',min(Ftests_phase.keys()))
                        getattr(m_phases[-1], add_param).frozen = False

                    #current best fit chi2 (extended points and actually fit for with maybe new param)
                    f_phases[-1] = pint.fitter.WLSFitter(t_phases[-1], m_phases[-1])
                    f_phases[-1].fit_toas()
                    chi2_phases.append(pint.residuals.Residuals(t_phases[-1], f_phases[-1].model).chi2.value)
                    #END INDENT OF FOR LOOP
                
                #have run program on all phase wraps
                #compare chi2 to see which is best and use that one's f, m, and t as the "set" f, m, and t
                print("Comparing phase wraps")
                print(np.column_stack((list(range(-args.max_wrap, args.max_wrap+1)), chi2_phases)))
                i_phase = np.argmin(chi2_phases)
                print("Phase wrap %d won with chi2 %f."%(list(range(-args.max_wrap, args.max_wrap+1))[i_phase], chi2_phases[i_phase]))
                f = deepcopy(f_phases[i_phase])
                m = deepcopy(m_phases[i_phase])
                t = deepcopy(t_phases[i_phase])

                #fit toas just in case 
                f.fit_toas()
                print("LOOOK HEEEERE!!!")
                print(f.get_fitparams().keys())
                #END INDENT FOR RESID > 0.35
            
            else:#if not resid > 0.35, run as normal, but don't want running if resid > 0.35    
                model0 = deepcopy(f.model)
                print('0 model chi2', f.resids.chi2)
                print('0 model chi2_ext', pint.residuals.Residuals(t_others, f.model).chi2)
                
                fig, ax = plt.subplots(constrained_layout=True)
                
                for i in range(len(rmods)):
                    print('chi2',pint.residuals.Residuals(t, rmods[i]).chi2)
                    print('chi2 ext', pint.residuals.Residuals(t_others, rmods[i]).chi2)
                    ax.plot(f_toas, rss[i], '-k', alpha=0.6)
                    
                print(f.get_fitparams().keys())
                print(t.ntoas)
                t = deepcopy(base_TOAs)
                #t is now a copy of the base TOAs (aka all the toas)
                print(t.ntoas)
                
                #plot post fit residuals with error bars
                xt = t.get_mjds()
                ax.errorbar(xt.value,
                    pint.residuals.Residuals(t, model0).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
                    t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
                #plt.plot(t.get_mjds(), pint.residuals.Residuals(t,m).time_resids.to(u.us).value, '.r', label = 'pre-fit')
                fitparams = ''
                for param in f.get_fitparams().keys():
                    fitparams += str(param)+' '
                plt.title("%s Post-Fit Residuals %d | fit params: %s" % (m.PSR.value, iteration, fitparams))
                ax.set_xlabel('MJD')
                ax.set_ylabel('Residual (us)')
                r = pint.residuals.Residuals(t,model0).time_resids.to(u.us).value
                #plt.ylim(-800,800)
                #yrange = (0.5/float(f.model.F0.value))*(10**6)
                yrange = abs(max(r)-min(r))
                ax.set_ylim(max(r) + 0.1*yrange, min(r) - 0.1*yrange)
                width = max(f_toas).value - min(f_toas).value
                if (min(f_toas).value - 0.1*width) < (min(xt).value-20) or (max(f_toas).value +0.1*width) > (max(xt).value+20):
                    ax.set_xlim(min(xt).value-20, max(xt).value+20)
                else:
                    ax.set_xlim(min(f_toas).value - 0.1*width, max(f_toas).value +0.1*width)
                #plt.xlim(53600,54600)
                #plt.legend()
                plt.grid()
                def us_to_phase(x):
                    return (x/(10**6))*f.model.F0.value
                
                def phase_to_us(y):
                    return (y/f.model.F0.value)*(10**6)
                
                secaxy = ax.secondary_yaxis('right', functions=(us_to_phase, phase_to_us))
                secaxy.set_ylabel("residuals (phase)")
                
                plt.savefig('./alg_saves/%s/%s_%03d.png'%(sys_name, sys_name, iteration), overwrite=True)
                plt.close()
                #plt.show()
                #end plotting
            
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
        
                
                #fit toas with new model
                f = pint.fitter.WLSFitter(t, m)
                f.fit_toas()
                span = f.toas.get_mjds().max() - f.toas.get_mjds().min()
                print('span',span)
                Ftests = dict()
                f_params = []
                #TODO: need to take into account if param isn't setup in model yet
                for param in m.params:
                    if getattr(m, param).frozen == False:
                        f_params.append(param)
                if 'RAJ' not in f_params and span > args.RAJ_lim*u.d:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #test RAJ
                    Ftest_R = Ftest_param(m, f, 'RAJ')
                    Ftests[Ftest_R] = 'RAJ'
                if 'DECJ' not in f_params and span > args.DECJ_lim*u.d:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #test DECJ
                    Ftest_D = Ftest_param(m, f, 'DECJ')
                    Ftests[Ftest_D] = 'DECJ'
                if 'F1' not in f_params and span > args.F1_lim*u.d:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #test F1 Ftest_param(fitter, toas, param_name) -> return ftest value
                    Ftest_F = Ftest_param(m, f, 'F1')
                    Ftests[Ftest_F] = 'F1'
                print("LOOOK HEEEEERRE!!! 2")
                print(Ftests.keys(), args.Ftest_lim)
                if not bool(Ftests.keys()) and span > 100*u.d:#doesnt actually do anything
                    print("F1, RAJ, DECJ, and F1 have been added. Will only add points from now on")
                    only_add = True
                    #if only_add true, then still check with random models, but instead of rechecking phase wrapped stuff, just choose version fits best and add point with that wrap
                elif not bool(Ftests.keys()):
                    '''just keep going to next step'''
                elif min(Ftests.keys()) < args.Ftest_lim:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    add_param = Ftests[min(Ftests.keys())]
                    print('adding param ', add_param, ' with Ftest ',min(Ftests.keys()))
                    getattr(m, add_param).frozen = False
        
                #current best fit chi2 (extended points and actually fit for with maybe new param)
                f = pint.fitter.WLSFitter(t, m)
                f.fit_toas()
                chi2_new_ext = pint.residuals.Residuals(t, f.model).chi2.value
                #END INDENT FOR ELSE (RESID < 0.35)
                
            #fit toas just in case 
            f.fit_toas()
            
            #get to this point, have a best fit model with or without a new parameter
            #actually fit with extra toa and same model
            print("START 2nd FTEST THING")
            print(t_small.ntoas, t.ntoas)
            f = pint.fitter.WLSFitter(t, m)
            f.fit_toas()
            f_small = pint.fitter.WLSFitter(t_small, m)
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
            if Ftest < args.RFtest_lim:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                continue #should be no deletion with the test data, so just ignore
                #shouldnt add the point - try phase wraps then delete
                #try phase -3 to +3
                print('len t_others',len(t_others.get_mjds()))
                #selected = np.logical_or(t_others.get_groups == closest_group,t_others.get_groups == closest_group)
                selected = np.zeros(len(t_others.get_mjds()), dtype = bool)
                if closest_group == min(t_others.get_groups()):
                    selected[0] = True
                else:
                    selected[-1] = True
                t0 = deepcopy(t_others)
                t3 = deepcopy(t_others)
                print(t3.table['delta_pulse_number'])
                add_phase_wrap(t3, model0, selected, args.phase_wraps)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                f = pint.fitter.WLSFitter(t, m_maybe)
                f.fit_toas()
                f_small = pint.fitter.WLSFitter(t_small, m_maybe)
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
                if Ftest < args.RFtest_lim:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! same as other still
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

            last_model = deepcopy(m)
            last_t = deepcopy(t)
            last_a = deepcopy(a)
            #write these to a par, tim and txt file, and add a thing at beginning to read in a starting list
            par = open('./alg_saves/'+sys_name+'/'+sys_name+'_'+str(iteration)+'.par','w')
            mask = open('./alg_saves/'+sys_name+'/'+sys_name+'_'+str(iteration)+'.csv','w')
            par.write(m.as_parfile())
            mask_string = ''
            for item in a:
                mask_string += str(int(item))+'\n'
            mask.write(mask_string)#list to string, then something at top to read in if given
            base_TOAs.write_TOA_file('./alg_saves/'+sys_name+'/'+sys_name+'_'+str(iteration)+'.tim', format="TEMPO2")
            par.close()
            mask.close()
            '''naming convention? system name, iteration, own folder in saves folder'''
            
            
            '''for each iteration, save picture, model, toas, and a'''
            
            
            
            
        #try with remaining params and see if better
        print(f.get_fitparams())
        #turn on RAJ, DECJ, and F1, and refit -> if chi2 worse or equal, ignore, if better, show that as final model
        m_plus = deepcopy(m)
        getattr(m_plus, 'RAJ').frozen = False
        getattr(m_plus, 'DECJ').frozen = False
        getattr(m_plus, 'F1').frozen = False
        f_plus = pint.fitter.WLSFitter(t, m_plus)
        f_plus.fit_toas()
        #residuals
        r = pint.residuals.Residuals(t, f.model)
        r_plus = pint.residuals.Residuals(t, f_plus.model)
        print(r_plus.chi2, r.chi2)
        if r_plus.chi2 >= r.chi2:
            #ignore
            '''ignore'''
        else:
            f = deepcopy(f_plus)
            
            
        print(f.model.as_parfile())
        #save as .fin
        fin_name = f.model.PSR.value + '.fin'
        finfile = open('./fake_data/'+fin_name, 'w')
        finfile.write(f.model.as_parfile())
        finfile.close()
        
        xt = t.get_mjds()
        plt.errorbar(xt.value,
        pint.residuals.Residuals(t, f.model).time_resids.to(u.us).value,
        t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
        plt.title("%s Final Post-Fit Timing Residuals" % m.PSR.value)
        plt.xlabel('MJD')
        plt.ylabel('Residual (us)')
        span = (0.5/float(f.model.F0.value))*(10**6)
        #r = pint.residuals.Residuals(t,m).time_resids.to(u.us).value
        #plt.ylim(-125000,12500)
        #plt.ylim(min(r)-200,max(r)+200)
        #width = max(f_toas).value - min(f_toas).value
        #plt.ylim(-span, span)
        #plt.xlim(min(f_toas).value-width/2, max(f_toas).value+width/2)
        #plt.xlim(51000,57500)
        #plt.legend()
        plt.grid()
        plt.show()

        
if __name__ == '__main__':
    main()
