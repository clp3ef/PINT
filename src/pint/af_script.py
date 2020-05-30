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
    """
    Choose which TOAs to start the fit at based on highest density
    
    :param toas: TOAs object of all TOAs
    :return list of boolean arrays to mask startng TOAs:
    """
    #TODO: better variable name for the starting list than 'a'
    #read in all toas
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
        #use the midpoint of the two toas to calculate distance to other toas
        avg_mjd = (sorted_mjd_list[i-1][1] + sorted_mjd_list[i][1])/2
        score = 0
        for j in range(1, len(sorted_mjd_list)):
            #iterate over all other toas, adding up their "moments" to get a final socre for the pair
            if (j == i) or (j == i-1):
                #if either of the points in the pair, skip so dont count self
                continue
            mjd = sorted_mjd_list[j][1]
            dist = np.fabs(avg_mjd - mjd)
            #score is the sum of "moments" of each other toa, so high density areas have high scores
            score += 1.0/dist
        #keys = tuples of indexes, items = scores
        score_dict[indexes] = score

    #sort the pairs from lowest to highest score
    sorted_score_list = sorted(score_dict.items(), key=operator.itemgetter(1))
    
    a_list = [] 
    for pair in sorted_score_list[-10:]:
        #create masks for the ten highest score pairs and add the masks to a_list
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
    """
    find the closest group of TOAs to the given toa(s)
    
    :param all_toas: TOAs object of all TOAs
    :param fit_toas: TOAs object of subset of TOAs that have already been fit
    :param base_TOAs: TOAs object of unedited TOAs as read from the timfile
    :return 
    """ 
   #read in the fit toas
    fit_mjds = fit_toas.get_mjds()
    d_left = d_right = None
    #find distance to closest toa to the fit toas on the left (unless fit toas includes the overall leftmost toa, in which case d_left remains None
    if min(fit_mjds) != min(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() < min(fit_mjds))
        left_dict = {min(fit_mjds) - mjd:mjd for mjd in all_toas.get_mjds()} 
        d_left = min(left_dict.keys())
    
    #redefine all_toas to remove selection from above if satement
    all_toas = deepcopy(base_TOAs)
    #find distance to closest toa to the fit toas on the right (unless fit toas includes the overall rightmost toa, in which case d_right remains None
    if max(fit_mjds) != max(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() > max(fit_mjds))
        right_dict = {mjd-max(fit_mjds):mjd for mjd in all_toas.get_mjds()} 
        d_right = min(right_dict.keys())
    
    #reset all_toas
    all_toas = deepcopy(base_TOAs)
    
    #take into account fit_toas being at an edge(d_left or d_right = None)
    if d_left == None and d_right == None:
        #fit toas includes leftmost and rightmost toas, and no gaps are allowed, so must have fit all toas
        print("all groups have been included")
        return None, None
    elif d_left == None or (d_right != None and d_right <= d_left):
        #if closest toa is to the right, return the group number of the closest group and the distance to that group
        all_toas.select(all_toas.get_mjds() == right_dict[d_right])
        return all_toas.table['groups'][0], d_right
    else:
        #if closest toa is to the left, return the group number of the closest group and the distance to that group
        all_toas.select(all_toas.get_mjds() == left_dict[d_left])
        #sign tracks direction closest point is, so negative seperation means to the left and positive to the right
        return all_toas.table['groups'][0], -d_left 

def Ftest_param(r_model, fitter, param_name):
    """
    do an Ftest comparing a model with and without a particular parameter added
    
    Note: this is NOT a general use function - it is specific to this code and cannot be easily adapted to other scripts
    :param r_model: timing model to be compared 
    :param fitter: fitter object containing the toas to compare on   
    :param param_name: name of the timing model parameter to be compared
    :return 
    """
    #read in model and toas
    m_plus_p = deepcopy(r_model)
    toas = deepcopy(fitter.toas)
    #set given parameter to unfrozen
    getattr(m_plus_p, param_name).frozen = False
    #make a fitter object with the chosen parameter unfrozen and fit the toas using the model with the extra parameter
    f_plus_p = pint.fitter.WLSFitter(toas, m_plus_p)
    f_plus_p.fit_toas()
    #calculate the residuals for the fit with (m_plus_p_rs) and without (m_rs) the extra parameter
    m_rs = pint.residuals.Residuals(toas, fitter.model)
    m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)
    #print exactly what goes into the Ftest
    print(m_rs.chi2.value, m_rs.dof, m_plus_p_rs.chi2.value, m_plus_p_rs.dof)
    #calculate the Ftest, comparing the chi2 and degrees of freedom of the two models
    #The Ftest determines how likely (from 0. to 1.) that improvement due to the new parameter is due to chance and not necessity
    #Ftests close to zero mean the parameter addition is necessary, close to 1 the addition is unnecessary, and NaN means the fit got worse when the parameter was added
    Ftest_p = ut.Ftest(float(m_rs.chi2.value), m_rs.dof, float(m_plus_p_rs.chi2.value), m_plus_p_rs.dof)
    c = 0
    while np.isnan(Ftest_p) and c < 10:
        #if the Ftest returns NaN (fit got worse), iterate the fit until it improves to a max of 10 iterations
        c += 1
        #refit the toas with the same model - may have gotten stuck in a local minima
        f_plus_p.fit_toas()
        m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)
        #recalculate the Ftest
        print(m_rs.chi2.value, m_rs.dof, m_plus_p_rs.chi2.value, m_plus_p_rs.dof, c)
        Ftest_p = ut.Ftest(float(m_rs.chi2.value), m_rs.dof, float(m_plus_p_rs.chi2.value), m_plus_p_rs.dof)
    #print the Ftest for the parameter and return the value of the Ftest
    print('Ftest'+param_name,Ftest_p)
    return Ftest_p

def set_F1_lim(args, parfile):
    #if F1_lim not specified in command line, calculate the minimum span based on general F0-F1 relations from P-Pdot diagram
    if args.F1_lim == None:
        #adjust F1_lim based on F0 value --> assume maximum possible F1
        F0 = mb.get_model(parfile).F0.value
        #for slow pulsars, allow F1 to be up to 1e-12 Hz/s, otherwise, 1e-14 Hz/s (recycled pulsars)
        if F0 < 100:
            F1 = 10**-12
        else:
            F1 = 10**-14
        #rearranged equation [delta-phase = (F1*span^2)/2], span in seconds. calculates span (in days) for delta-phase to reach 0.35 due to F1
        args.F1_lim = np.sqrt(0.35*2/F1)/86400.0    

def readin_starting_points(a, t, start_type, start, args):
        #if given starting points from command line, replace calculated starting points with given starting points (group numbers or mjd values)
        groups = t.get_groups()
        print(groups)
        if start_type == "groups":
            #starting point is the group(s) specified
            a = np.logical_or(groups == start[0], groups == start[1])
        elif start_type == "mjds":
            #starting point is the mjds in the range given (if no mjds in that range, program crashes)
            a = np.logical_and(t.get_mjds() > start[0]*u.d, t.get_mjds() < start[1]*u.d)
        
        #if given a maskfile (csv) of a saved boolean array, use that array to choose the starting subset of toas
        if args.maskfile != None:
            mask_read = open(args.maskfile, 'r')
            data = csv.reader(mask_read)
            a = [bool(int(row[0])) for row in data]
        return a

def calc_resid_diff(closest_group, full_groups, base_TOAs, f, selected):
    #if closest group is >0.35 phase away in either direction, try both phase wraps
    #create mask array for closest group (without fit toas)
    selected_closest = [True if group == closest_group else False for group in full_groups]
    #calculate phase resid of last toa of the fit toas and first toa of the closest group. 
    last_fit_toa_phase = pint.residuals.Residuals(base_TOAs, f.model).phase_resids[selected][-1]
    first_new_toa_phase = pint.residuals.Residuals(base_TOAs, f.model).phase_resids[selected_closest][0]
    #Use difference of edge points as difference between groups as a whole
    diff = first_new_toa_phase - last_fit_toa_phase
    print(last_fit_toa_phase, first_new_toa_phase, diff)
    return selected_closest, diff

def plot_wraps(f, t_others_phases, rmods, f_toas, rss, t_phases, m, iteration, wrap, sys_name):
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
        
    #plot post fit residuals with error bars
    xt = t_phases[-1].get_mjds()
    ax.errorbar(xt.value,
        pint.residuals.Residuals(t_phases[-1], model0).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
        t_phases[-1].get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
    #make string of parameters that have been fit
    fitparams = ''
    for param in f.get_fitparams().keys():
        fitparams += str(param)+' '
    #notate pulsar name, iteration number, phase wrap, and parameters that have been fit
    plt.title("%s Post-Fit Residuals %d P%d | fit params: %s" % (m.PSR.value, iteration, wrap, fitparams))
    ax.set_xlabel('MJD')
    ax.set_ylabel('Residual (us)')
    r = pint.residuals.Residuals(t_phases[-1],model0).time_resids.to(u.us).value
    #set the y limits to just above and below the highest and lowest points
    yrange = abs(max(r)-min(r))
    ax.set_ylim(max(r) + 0.1*yrange, min(r) - 0.1*yrange)
    width = max(f_toas).value - min(f_toas).value
    #if the random lines are within the minimum and maximum toas, scale to the edges of the random models
    if (min(f_toas).value - 0.1*width) < (min(xt).value-20) or (max(f_toas).value +0.1*width) > (max(xt).value+20):
        ax.set_xlim(min(xt).value-20, max(xt).value+20)
    #otherwise scale to include all the toas
    else:
        ax.set_xlim(min(f_toas).value - 0.1*width, max(f_toas).value +0.1*width)
    plt.grid()
    def us_to_phase(x):
        return (x/(10**6))*f.model.F0.value
    
    def phase_to_us(y):
        return (y/f.model.F0.value)*(10**6)
    #include a secondary axis for phase
    secaxy = ax.secondary_yaxis('right', functions=(us_to_phase, phase_to_us))
    secaxy.set_ylabel("residuals (phase)")
                    
    #save the image in alg_saves with the iteration and wrap number
    plt.savefig('./alg_saves/%s/%s_%03d_P%03d.png'%(sys_name, sys_name, iteration, wrap), overwrite=True)
    plt.close()
    #plt.show()
    
def plot_plain(f, t_others, rmods, f_toas, rss, t, m, iteration, sys_name, fig, ax):
    #plot post fit residuals with error bars
    model0 = deepcopy(f.model)
    xt = t.get_mjds()
    ax.errorbar(xt.value,
        pint.residuals.Residuals(t, model0).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
        t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
    fitparams = ''
    #make a string of all the fit parameters
    for param in f.get_fitparams().keys():
        fitparams += str(param)+' '
    #notate the pulsar name, iteration, and fit parameters
    plt.title("%s Post-Fit Residuals %d | fit params: %s" % (m.PSR.value, iteration, fitparams))
    ax.set_xlabel('MJD')
    ax.set_ylabel('Residual (us)')
    r = pint.residuals.Residuals(t,model0).time_resids.to(u.us).value
    #set the y limit to be just above and below the max and min points
    yrange = abs(max(r)-min(r))
    ax.set_ylim(max(r) + 0.1*yrange, min(r) - 0.1*yrange)
    #scale to the edges of the points or the edges of the random models, whichever is smaller
    width = max(f_toas).value - min(f_toas).value
    if (min(f_toas).value - 0.1*width) < (min(xt).value-20) or (max(f_toas).value +0.1*width) > (max(xt).value+20):
        ax.set_xlim(min(xt).value-20, max(xt).value+20)
    else:
        ax.set_xlim(min(f_toas).value - 0.1*width, max(f_toas).value +0.1*width)
    plt.grid()
    def us_to_phase(x):
        return (x/(10**6))*f.model.F0.value
    
    def phase_to_us(y):
        return (y/f.model.F0.value)*(10**6)
    #include secondary axis to show phase
    secaxy = ax.secondary_yaxis('right', functions=(us_to_phase, phase_to_us))
    secaxy.set_ylabel("residuals (phase)")
    
    plt.savefig('./alg_saves/%s/%s_%03d.png'%(sys_name, sys_name, iteration), overwrite=True)
    plt.close()
    #plt.show()
    #end plotting

def do_Ftests(t, m, args):
    #fit toas with new model
    f = pint.fitter.WLSFitter(t, m)
    f.fit_toas()
    #calculate the span of fit toas for comparison to minimum parameter spans
    span = f.toas.get_mjds().max() - f.toas.get_mjds().min()
    print('span',span)
    Ftests = dict()
    f_params = []
    #TODO: need to take into account if param isn't setup in model yet
    #make list of already fit parameters
    for param in m.params:
        if getattr(m, param).frozen == False:
            f_params.append(param)
    #if span is longer than minimum parameter span and parameter hasn't been added yet, do ftest to see if parameter should be added
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
    print(Ftests.keys(), args.Ftest_lim)
    #if no Ftests performed, continue on without change
    if not bool(Ftests.keys()):
        if span > 100*u.d:
            print("F1, RAJ, DECJ, and F1 have been added. Will only add points from now on")
    #if smallest Ftest of those calculated is less than the given limit, add that parameter to the model. Otherwise add no parameters
    elif min(Ftests.keys()) < args.Ftest_lim:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        add_param = Ftests[min(Ftests.keys())]
        print('adding param ', add_param, ' with Ftest ',min(Ftests.keys()))
        getattr(m, add_param).frozen = False
    return m
                    
def do_Ftests_phases(m_phases, t_phases, f_phases, args):
    #calculate the span of the fit toas to compare to minimum spans for parameters
    span = f_phases[-1].toas.get_mjds().max() - f_phases[-1].toas.get_mjds().min()
    Ftests_phase = dict()
    f_params_phase = []
    #TODO: need to take into account if param isn't setup in model yet
    #make a list of all the fit params
    for param in m_phases[-1].params:
        if getattr(m_phases[-1], param).frozen == False:
            f_params_phase.append(param)
    #if a given parameter has not already been fit (in fit_params) and span > minimum fitting span for that param, do an Ftest for that param
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
    #print the Ftest values for all the parameters
    print(Ftests_phase.keys())
    #if nothing in the Ftests list, continue to next step. Print message if long enough span that all params should be added 
    if not bool(Ftests_phase.keys()): 
        if span > 100*u.d:
            print("F1, RAJ, DECJ, and F1 have been added. Will only add points from now on")
    #whichever parameter's Ftest is smallest and less than the Ftest limit gets added to the model. Else no parameter gets added
    elif min(Ftests_phase.keys()) < args.Ftest_lim:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        add_param = Ftests_phase[min(Ftests_phase.keys())]
        print('adding param ', add_param, ' with Ftest ',min(Ftests_phase.keys()))
        getattr(m_phases[-1], add_param).frozen = False
    return m_phases[-1]
                    
def calc_random_models(base_TOAs, f, t, args):
    full_groups = base_TOAs.table['groups']
    #create a mask which produces the current subset of toas
    selected = [True if group in t.table['groups'] else False for group in full_groups] 
    #calculate the average phase resid of the fit toas
    rs_mean = pint.residuals.Residuals(base_TOAs, f.model, set_pulse_nums=True).phase_resids[selected].mean()
    #produce several (r_iter) random models given the fitter object and mean residual. return the random models, their residuals, and evenly spaced toas to plot against
    f_toas, rss, rmods = pint.random_models.random_models(f, rs_mean, iter=args.r_iter, ledge_multiplier=args.ledge_multiplier, redge_multiplier=args.redge_multiplier)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return full_groups, selected, rs_mean, f_toas, rss, rmods

def save_state(m, t, a, sys_name, iteration, base_TOAs):
            last_model = deepcopy(m)
            last_t = deepcopy(t)
            last_a = deepcopy(a)
            #write these to a par, tim and txt file to be saved and reloaded
            par = open('./alg_saves/'+sys_name+'/'+sys_name+'_'+str(iteration)+'.par','w')
            mask = open('./alg_saves/'+sys_name+'/'+sys_name+'_'+str(iteration)+'.csv','w')
            par.write(m.as_parfile())
            mask_string = ''
            for item in a:
                mask_string += str(int(item))+'\n'
            mask.write(mask_string)#list to string
            base_TOAs.write_TOA_file('./alg_saves/'+sys_name+'/'+sys_name+'_'+str(iteration)+'.tim', format="TEMPO2")
            par.close()
            mask.close()
            return last_model, last_t, last_a

                    

def main(argv=None):
    import argparse
    import sys
    #read in arguments from the command line
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
        "--F1_lim", help="minimum time span before Spindown (F1) can be fit for (default = time for F1 to change residuals by 0.35phase)", type=float, default=None
    )
    parser.add_argument(
        "--Ftest_lim", help="Upper limit for successful Ftest values", type=float, default=0.0005
    )
    parser.add_argument(
        "--RFtest_lim", help="Upper limit for successful RFtest values", type=float, default=0.000000005
    )
    parser.add_argument("--check_bad_points", help="whether the algorithm should attempt to identify and ignore bad data", type=str, default='True'
    )
    parser.add_argument("--check_max_resid", help="maximum acceptable goodness of fit for polyfit to identify a bad data point", type=float, default=0.02
    )
    parser.add_argument("--n_check", help="how many TOAs ahead of questionable TOA to fit to confirm a bad data point", type=int, default=3
    )
    parser.add_argument("--try_speed_up", help="whether to try to speed up the process by fitting ahead where polyfit confirms a clear trend", type=str, default='True'
    )
    parser.add_argument("--speed_up_min_span", help="minimum span (days) before allowing speed up attempts", type=float, default=30
    )
    parser.add_argument("--speed_max_resid", help="maximum acceptable goodness of fit for polyfit to allow the speed up to succeed", type=float, default=0.02
    )
    parser.add_argument("--span1_c", help="coefficient for first speed up span (i.e. try polyfit on current span * span1_c)", type=float, default=1.3
    )
    parser.add_argument("--span2_c", help="coefficient for second speed up span (i.e. try polyfit on current span * span2_c)", type=float, default=1.8
    )
    parser.add_argument("--span3_c", help="coefficient for third speed up span (i.e. try polyfit on current span * span3_c)", type=float, default=2.4
    )
    parser.add_argument("--max_wrap", help="how many phase wraps in each direction to try", type=int, default=1
    )
    parser.add_argument("--clear_folder", help="whether to remove all existing files from the system's save folder", type=bool, default=False
    )

    args = parser.parse_args(argv)
    #interpret strings as booleans for check_bad_points and try_speed_up
    args.check_bad_points = [False, True][args.check_bad_points.lower()[0] == 't'] 
    args.try_speed_up = [False, True][args.try_speed_up.lower()[0] == 't'] 
    
    #if given starting points from command line, check if ints (group numbers) or floats (mjd values)
    start_type = None
    start = None
    if args.starting_points != None:
        start = args.starting_points.split(',')
        try:
            start = [int(i) for i in start]
            start_type = "groups"
        except:
            start = [float(i) for i in start]
            start_type = "mjds"
    

            
    '''start main program'''
    #construct the filenames
    datadir = os.path.dirname(os.path.abspath(str(__file__)))
    parfile = os.path.join(datadir, args.parfile)
    timfile = os.path.join(datadir, args.timfile)
    
    #read in the toas
    t = pint.toa.get_TOAs(timfile)
    sys_name = str(mb.get_model(parfile).PSR.value)
    
    #set F1_lim if one not given
    set_F1_lim(args, parfile)
    
    #make paths
    
    #check that there is a directory to save the algorithm state in
    if not os.path.exists('alg_saves'):
        os.mkdir('alg_saves')

    #checks there is a directory specific to the system in alg_saves
    if not os.path.exists('alg_saves/'+sys_name):
        os.mkdir('alg_saves/'+sys_name)
    
    #if told to by user deletes all existing files in the folder for clarity
    #if args.clear_folder:
    #    os.remove('./alg_saves/'+sys_name+'/*')

    for a in starting_points(t):
        #a is a list of 10 boolean arrays, each a mask for the base toas. Iterating through all ten gives ten different pairs of starting points
        # read in the initial model
        m = mb.get_model(parfile)
        
        # Read in the TOAs
        t = pint.toa.get_TOAs(timfile)
        
        # Print a summary of the TOAs that we have
        t.print_summary()
        
        #check has TZR params
        try:
            m.TZRMJD
            m.TZRSITE
            m.TZRFRQ
        except:
            print("Error: Must include TZR parameters in parfile")
            return -1

        if args.starting_points != None or args.maskfile != None:
            a = readin_starting_points(a, t, start_type, start, args)
        
        #print the starting subset for this attempt    
        print('a for this attempt',a)
        #apply the starting mask and print the group(s) the starting points are part of
        t.select(a)
        print(t.table['groups'])
        
        #for first iteration, last model, toas, and starting points is just the base ones read in
        last_model = deepcopy(m)
        last_t = deepcopy(t)
        last_a = deepcopy(a)
        #toas as read in from timfile, should only be modified with deletions
        base_TOAs = pint.toa.get_TOAs(timfile)

        cont = True
        iteration = 0
        #if given a maskfile (csv), read in the iteration we are on from the maskfile filename
        if args.maskfile != None:
            iteration = int(args.maskfile[-8:].strip('qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHGFDSAMNBVCXZ._'))
            
        while cont:
            #main loop of the algorithm, continues until all toas have been included in fit
            iteration += 1
            skip_phases = False
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~basic fit with best fit model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~            
            # fit the toas with the given model as a baseline
            print("Fitting...")
            f = pint.fitter.WLSFitter(t, m)
            print("BEFORE:",f.get_fitparams())
            print(f.fit_toas())
        
            print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
            print("RMS in phase is", f.resids.phase_resids.std())
            print("RMS in time is", f.resids.time_resids.std().to(u.us))
            print("\n Best model is:")
            print(f.model.as_parfile())
            
            #calculate random models and residuals
            full_groups, selected, rs_mean, f_toas, rss, rmods = calc_random_models(base_TOAs, f, t, args)
                            
            #define t_others
            t_others = deepcopy(base_TOAs)
                 #~~~~~~~~~~~~~~~~~~~~~~~calc closest group~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                       
            print('rs_mean',rs_mean)
            print(t.table['groups'])
            #calculate the group closest to the fit toas, pass deepcopies to prevent unintended pass by reference
            closest_group, dist = get_closest_group(deepcopy(t_others), deepcopy(t), deepcopy(base_TOAs))
            print('closest_group',closest_group)
            if closest_group == None:
                #end the program
                #print the final model and toas
                #save the latest model as a new parfile (all done at end of code)
                cont = False
                continue
                        

            #right now t_others is just all the toas, so can use as all
            #redefine a as the mask giving the fit toas plus the closest group of toas
            a = np.logical_or(a, t_others.get_groups() == closest_group)
            #define t_others as the current fit toas pluse the closest group 
            t_others.select(a)
            print(t_others.table['groups'])
            #t_others is now the fit toas plus the to be added group of toas, t is just the fit toas

            #calculate difference in resids between current fit group and closest group
            selected_closest, diff = calc_resid_diff(closest_group, full_groups, base_TOAs, f, selected)
            #if difference in phase is >0.35, try phase wraps t see if point fits better wrapped
            if np.abs(diff) > 0.35 and args.check_bad_points == True:
                #try polyfit on next n data, and if works (has resids < 0.02), just ignore it as a bad data point, and fit the next n data points instead
                if dist > 0:
                    #next point is to the right
                    try_mask = [True if group in t.get_groups() or group in np.arange(closest_group+1, closest_group+1+args.n_check) else False for group in full_groups]
                else:
                    #next point is to the left
                    try_mask = [True if group in t.get_groups() or group in np.arange(closest_group-args.n_check, closest_group) else False for group in full_groups]
                try_t = deepcopy(base_TOAs)
                try_t.select(try_mask)
                try_resids = np.float64(pint.residuals.Residuals(try_t, m).phase_resids)
                try_mjds = np.float64(try_t.get_mjds())
                p, resids, q1, q2, q3 = np.polyfit(try_mjds, try_resids, 3, full=True)
                if resids.size == 0:
                    #means residuals were perfectly 0, shouldnt happen if have enough data
                    resids = [0.0]
                    print("phase resids was empty")
                print('p', p)
                print('resids (phase)', resids)
                
                bad_point_t = deepcopy(base_TOAs)
                bad_point_t.select(bad_point_t.get_mjds() >= min(try_t.get_mjds()))                
                bad_point_t.select(bad_point_t.get_mjds() <= max(try_t.get_mjds()))
                bad_point_r= pint.residuals.Residuals(bad_point_t, m).phase_resids
                index = np.where(bad_point_t.get_groups() == closest_group)
                x = np.arange(min(try_mjds)/u.d ,max(try_mjds)/u.d , 2)
                y = p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
                plt.plot(try_mjds, try_resids, 'b.')
                plt.plot(bad_point_t.get_mjds()[index], bad_point_r[index], 'r.')
                plt.plot(x, y, 'g-')
                plt.grid()
                plt.xlabel('MJD')
                plt.ylabel('phase resids')
                plt.title("Checking Bad Point")
                plt.show()
                                        
                if resids[0] < args.check_max_resid:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #remove/ignore bad point, fit the next n TOAs instead
                    print("Ignoring Bad Data Point, skipping to regular fit")
                    print(t_others.get_groups())
                    print(a)
                    t_others = deepcopy(try_t)
                    a = [True if group in t_others.get_groups() else False for group in full_groups]
                    skip_phases = True
                    print(t_others.get_groups())
                    print(a)

            if np.abs(diff) > 0.35 and skip_phases == False:
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
                #try every phase wrap from -max_wrap to +max_wrap
                for wrap in range(-args.max_wrap, args.max_wrap+1):
                    #copy models to appropriate lists --> use index -1 because current object will always be the one just appended to array (AKA, pos. -1)
                    print("Trying phase wrap:", wrap)
                    #append the current fitter and toas to the appropriate lists 
                    f_phases.append(deepcopy(f))
                    t_phases.append(deepcopy(base_TOAs))
                    print(t_phases)
                    #add the phase wrap to the closest group
                    add_phase_wrap(t_phases[-1], f.model, selected_closest, wrap)
                    #append the wrapped toas to t_others and select the fit toas and closest group as normal
                    t_others_phases.append(deepcopy(t_phases[-1]))
                    t_others_phases[-1].select(a)
                    
                    #plot data
                    plot_wraps(f, t_others_phases, rmods, f_toas, rss, t_phases, m, iteration, wrap, sys_name)            
                    
                    #repeat model selection with phase wrap. f.model should be same as f_phases[-1].model (all f_phases[n] should be the same)
                    chi2_ext_phase = [pint.residuals.Residuals(t_others_phases[-1], rmods[i]).chi2_reduced.value for i in range(len(rmods))]
                    chi2_dict_phase = dict(zip(chi2_ext_phase, rmods))
                    #append 0model to dict so it can also be a possibility
                    chi2_dict_phase[pint.residuals.Residuals(t_others_phases[-1], f.model).chi2_reduced.value] = f.model
                    min_chi2_phase = sorted(chi2_dict_phase.keys())[0]
                    
                    #m_phases is list of best models from each phase wrap
                    m_phases.append(chi2_dict_phase[min_chi2_phase])
                    #a = current t plus closest group, defined above
                    t_phases[-1].select(a)
                    
                    #fit toas with new model
                    f_phases[-1] = pint.fitter.WLSFitter(t_phases[-1], m_phases[-1])
                    f_phases[-1].fit_toas()
                    
                    #do Ftests with phase wraps
                    m_phases[-1] = do_Ftests_phases(m_phases, t_phases, f_phases, args)
                    
                    #current best fit chi2 (extended points and actually fit for with maybe new param)
                    f_phases[-1] = pint.fitter.WLSFitter(t_phases[-1], m_phases[-1])
                    f_phases[-1].fit_toas()
                    chi2_phases.append(pint.residuals.Residuals(t_phases[-1], f_phases[-1].model).chi2.value)
                
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
                print(f.get_fitparams().keys())
                #END INDENT FOR RESID > 0.35
                
            else:#if not resid > 0.35, run as normal, but don't want running if resid > 0.35    
                #do speed-up check
                #t is the current fit toas, t_others is the current fit toas plus the closest group, and a is the same as t_others
                minmjd, maxmjd = (min(t_others.get_mjds()), max(t_others.get_mjds()))
                print("PRESPAN is", maxmjd-minmjd) 
                print(args.try_speed_up)
                if (maxmjd-minmjd) > args.speed_up_min_span * u.d and args.try_speed_up == True:#!!!!!!!!!!!!!!!!!!!!!!!!!
                    try:
                        try_span1 = args.span1_c*(maxmjd-minmjd)#!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        print("TRYSPAN1 is",try_span1)
                        new_t = deepcopy(base_TOAs)
                        if dist > 0:
                            #next data is to the right
                            new_t.select(new_t.get_mjds() > maxmjd)
                            new_t.select(new_t.get_mjds() < minmjd + try_span1)
                        else:
                            #next data is to the left
                            new_t.select(new_t.get_mjds() < minmjd)
                            new_t.select(new_t.get_mjds() > maxmjd - try_span1)
                        #try_t now includes all the TOAs to be fit by polyfit but are not included in t_others
                        try_mask = [True if group in t_others.get_groups() or group in new_t.get_groups() else False for group in full_groups]
                        try_t = deepcopy(base_TOAs)
                        try_t.select(try_mask)
                        try_resids = np.float64(pint.residuals.Residuals(try_t, m).phase_resids)
                        try_mjds = np.float64(try_t.get_mjds())
                        p, resids, q1, q2, q3 = np.polyfit(try_mjds, try_resids, 3, full=True)
                        if resids.size == 0:
                            #shouldnt happen if make it wait until more than a week of data
                            resids = [0.0]
                            print("resids was empty")
                        print('p', p)
                        print('resids', resids)
                        
                        x = np.arange(min(try_mjds)/u.d ,max(try_mjds)/u.d , 2)
                        y = p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
                        plt.plot(try_mjds, try_resids, 'b.')
                        plt.plot(x, y, 'g-')
                        plt.grid()
                        plt.xlabel('MJD')
                        plt.ylabel('phase resids')
                        plt.show()
                        
                        if resids[0] < args.speed_max_resid:#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            #go ahead and fit on all those days
                            #try with even bigger span
                            try_span2 = args.span2_c*(maxmjd-minmjd)
                            print("TRYSPAN2 is",try_span2)
                            new_t2 = deepcopy(base_TOAs)
                            if dist > 0:
                                #next data is to the right
                                new_t2.select(new_t2.get_mjds() > maxmjd)
                                new_t2.select(new_t2.get_mjds() < minmjd + try_span2)
                            else:
                                #next data is to the left
                                new_t2.select(new_t2.get_mjds() < minmjd)
                                new_t2.select(new_t2.get_mjds() > maxmjd - try_span2)
                            #try_t now includes all the TOAs to be fit by polyfit but are not included in t_others
                            try_mask2 = [True if group in t_others.get_groups() or group in new_t2.get_groups() else False for group in full_groups]
                            try_t2 = deepcopy(base_TOAs)
                            try_t2.select(try_mask2)
                            
                            try_resids2 = np.float64(pint.residuals.Residuals(try_t2, m).phase_resids)
                            try_mjds2 = np.float64(try_t2.get_mjds())
                            p, resids2, q1, q2, q3 = np.polyfit(try_mjds2, try_resids2, 3, full=True)
                            if resids2.size == 0:
                                #shouldnt happen if make it wait until more than a week of data
                                resids2 = [0.0]
                                print("resids was empty")
                            print('p', p)
                            print('resids2', resids2)
                            
                            x = np.arange(min(try_mjds2)/u.d ,max(try_mjds2)/u.d , 2)
                            y = p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
                            plt.plot(try_mjds2, try_resids2, 'b.')
                            plt.plot(x, y, 'k-')
                            plt.grid()
                            plt.xlabel('MJD')
                            plt.ylabel('phase resids')
                            plt.show()
                            
                            if resids2[0] < args.speed_max_resid:
                                #go ahead and fit on all those days
                                #try with even bigger span
                                try_span3 = args.span3_c*(maxmjd-minmjd)
                                print("TRYSPAN3 is", try_span3)
                                new_t3 = deepcopy(base_TOAs)
                                if dist > 0:
                                    #next data is to the right
                                    new_t3.select(new_t3.get_mjds() > maxmjd)
                                    new_t3.select(new_t3.get_mjds() < minmjd + try_span3)
                                else:
                                    #next data is to the left
                                    new_t3.select(new_t3.get_mjds() < minmjd)
                                    new_t3.select(new_t3.get_mjds() > maxmjd - try_span3)
                                #try_t now includes all the TOAs to be fit by polyfit but are not included in t_others
                                try_mask3 = [True if group in t_others.get_groups() or group in new_t3.get_groups() else False for group in full_groups]
                                try_t3 = deepcopy(base_TOAs)
                                try_t3.select(try_mask3)
                                
                                try_resids3 = np.float64(pint.residuals.Residuals(try_t3, m).phase_resids)
                                try_mjds3 = np.float64(try_t3.get_mjds())
                                p, resids3, q1, q2, q3 = np.polyfit(try_mjds3, try_resids3, 3, full=True)
                                if resids3.size == 0:
                                    #shouldnt happen if make it wait until more than a week of data
                                    resids3 = [0.0]
                                    print("resids was empty")
                                print('p', p)
                                print('resids3', resids3)
                                
                                x = np.arange(min(try_mjds3)/u.d ,max(try_mjds3)/u.d , 2)
                                y = p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
                                plt.plot(try_mjds3, try_resids3, 'b.')
                                plt.plot(x, y, 'm-')
                                plt.grid()
                                plt.xlabel('MJD')
                                plt.ylabel('phase resids')
                                plt.show()
                                
                                if resids3[0] < args.speed_max_resid:
                                    print("Fitting points from", minmjd, "to", minmjd+try_span3)
                                    print(t_others.get_groups())
                                    print(a)
                                    t_others = deepcopy(try_t3)
                                    a = [True if group in t_others.get_groups() else False for group in full_groups]
                                    print(t_others.get_groups())
                                    print(a)
                                else:
                                    print("Fitting points from", minmjd, "to", minmjd+try_span2)
                                    print(t_others.get_groups())
                                    print(a)
                                    t_others = deepcopy(try_t2)
                                    a = [True if group in t_others.get_groups() else False for group in full_groups]
                                    print(t_others.get_groups())
                                    print(a)
                            else:
                                #and repeat all above until get bad resids, then do else and the below
                                print("Fitting points from", minmjd, "to", minmjd+try_span1)
                                print(t_others.get_groups())
                                print(a)
                                t_others = deepcopy(try_t)
                                a = [True if group in t_others.get_groups() else False for group in full_groups]
                                print(t_others.get_groups())
                                print(a)
                    except:
                        print("an error occued while trying to do speed up. Continuing On")
                #calculate chi2 and reduced chi2 for base model
                model0 = deepcopy(f.model)
                print('0 model chi2', f.resids.chi2)
                print('0 model chi2_ext', pint.residuals.Residuals(t_others, f.model).chi2)
                
                fig, ax = plt.subplots(constrained_layout=True)
    
                #calculate chi2 and reduced chi2 for the random models
                for i in range(len(rmods)):
                    print('chi2',pint.residuals.Residuals(t, rmods[i]).chi2)
                    print('chi2 ext', pint.residuals.Residuals(t_others, rmods[i]).chi2)
                    ax.plot(f_toas, rss[i], '-k', alpha=0.6)
                    
                print(f.get_fitparams().keys())
                print(t.ntoas)
                t = deepcopy(base_TOAs)
                #t is now a copy of the base TOAs (aka all the toas)
                print(t.ntoas)

                #plot data
                plot_plain(f, t_others, rmods, f_toas, rss, t, m, iteration, sys_name, fig, ax)            
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~choose next model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #get next model by comparing chi2 for t_others
                chi2_ext = [pint.residuals.Residuals(t_others, rmods[i]).chi2_reduced.value for i in range(len(rmods))]
                chi2_dict = dict(zip(chi2_ext, rmods))
                #append 0model to dict so it can also be a possibility
                chi2_dict[pint.residuals.Residuals(t_others, f.model).chi2_reduced.value] = f.model
                min_chi2 = sorted(chi2_dict.keys())[0]
                
                #the model with the smallest chi2 is chosen as the new best fit model
                m = chi2_dict[min_chi2]
                #a = current t plus closest group, defined above
                t.select(a)
                
                #do Ftests 
                m = do_Ftests(t, m, args)                

                #current best fit chi2 (extended points and actually fit for with maybe new param)
                f = pint.fitter.WLSFitter(t, m)
                f.fit_toas()
                chi2_new_ext = pint.residuals.Residuals(t, f.model).chi2.value
                #END INDENT FOR ELSE (RESID < 0.35)
                
            #fit toas just in case 
            f.fit_toas()
            
            #save current state in par, tim, and csv files
            last_model, last_t, last_a = save_state(m, t, a, sys_name, iteration, base_TOAs)
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
        plt.grid()
        plt.show()

        
if __name__ == '__main__':
    main()
