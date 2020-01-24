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
                                                                                                                                                                                                                                                                                    
def add_jump(model, all_toas, selected_toas, selected):
    """
    jump the toas selected or unjump them if already jumped
    
    :param selected: boolean array to apply to toas, True = selected toa
    """
    if "PhaseJump" not in model.components:
        # if no PhaseJump component, add one
        print("PhaseJump component added")
        a = pint.models.jump.PhaseJump()
        a.setup()
        model.add_component(a)
        model.components['PhaseJump']._parent = model
        model.remove_param("JUMP1")
        param = pint.models.parameter.maskParameter(
        name="JUMP", index=1, key="jump", key_value=1, value=0.0, units="second"
        )
        model.add_param_from_top(param, "PhaseJump")
        getattr(model, param.name).frozen = False
        for dict1, dict2 in zip(
            all_toas.table["flags"][selected],
            selected_toas.table["flags"],
        ):
            dict1["jump"] = 1
            dict2["jump"] = 1
        return param.name
    model.components['PhaseJump']._parent = model
    # if gets here, has at least one jump param already
    # if doesnt overlap or cancel, add the param
    jump_nums = [
    int(dict["jump"]) if "jump" in dict.keys() else np.nan
    for dict in all_toas.table["flags"]
    ]
    for num in range(1, int(np.nanmax(jump_nums) + 1)):
        num = int(num)
        jump_select = [num == jump_num for jump_num in jump_nums]
        if np.array_equal(jump_select, selected):
            print('removing a jump')
            # if current jump exactly matches selected, remove it
            model.remove_param("JUMP" + str(num))
            for dict1, dict2 in zip(
                all_toas.table["flags"][selected],
                selected_toas.table["flags"],
            ):
                if "jump" in dict1.keys() and dict1["jump"] == num:
                    del dict1["jump"]  # somehow deletes from both
            nums_subset = range(num + 1, int(np.nanmax(jump_nums) + 1))
            for n in nums_subset:
                # iterate through jump params and rename them so that they are always in numerical order starting with JUMP1
                n = int(n)
                print(n)
                for dict in all_toas.table["flags"]:
                    if "jump" in dict.keys() and dict["jump"] == n:
                        dict["jump"] = n - 1
                param = pint.models.parameter.maskParameter(
                    name="JUMP",
                    index=int(n - 1),
                    key="jump",
                    key_value=int(n - 1),
                    value=getattr(model, "JUMP" + str(n)).value,
                    units="second",
                )
                model.add_param_from_top(param, "PhaseJump")
                getattr(model, param.name).frozen = getattr(
                    model, "JUMP" + str(n)
                ).frozen
                model.remove_param("JUMP" + str(n))
                #if self.fitted:
                #    self.postfit_model.add_param_from_top(param, "PhaseJump")
                #    getattr(self.postfit_model, param.name).frozen = getattr(
                #        self.postfit_model, "JUMP" + str(n)
                #    ).frozen
                #    self.postfit_model.remove_param("JUMP" + str(n))
            if "JUMP1" not in model.params:
                # remove PhaseJump component if no jump params
                comp_list = getattr(model, "PhaseComponent_list")
                for item in comp_list:
                    if isinstance(item, pint.models.jump.PhaseJump):
                        comp_list.remove(item)
                        break
                #model.setup_components(comp_list)
            else:
                model.components["PhaseJump"].setup()
            print("removed param", "JUMP" + str(num))
            return jump_select
        elif True in [a and b for a, b in zip(jump_select, selected)]:
            # if current jump overlaps selected, raise and error and end
            print("The selected toa(s) overlap an existing jump. Remove all interfering jumps before attempting to jump this point")
            return None
    
    # if here, then doesn't overlap or match anything
    for dict1, dict2 in zip(
        all_toas.table["flags"][selected], selected_toas.table["flags"]
    ):
        dict1["jump"] = int(np.nanmax(jump_nums)) + 1
        dict2["jump"] = int(np.nanmax(jump_nums)) + 1
    param = pint.models.parameter.maskParameter(
        name="JUMP",
        index=int(np.nanmax(jump_nums)) + 1,
        key="jump",
        key_value=int(np.nanmax(jump_nums)) + 1,
        value=0.0,
        units="second",
        aliases=["JUMP"],
    )
    model.add_param_from_top(param, "PhaseJump")
    getattr(model, param.name).frozen = False
    model.components["PhaseJump"].setup()
    return param.name
            
def jump_all_groups(model, all_toas, selected_toas, selected):
    # jump all groups except the one(s) selected, or jump all groups if none selected
    groups = list(all_toas.table["groups"])
    # jump each group, check doesn't overlap with existing jumps and selected
    print(np.arange(max(groups)+1))
    for num in np.arange(max(groups) + 1):
        group_bool = [num == group for group in all_toas.table["groups"]]
        if True in  [a and b for a, b in zip(group_bool, selected)]:
            print('continue')
            continue
        selected_toas = deepcopy(all_toas)
        selected_toas.select(group_bool)
        jump_name = add_jump(model, all_toas, selected_toas, group_bool)
        print(jump_name)
        
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

    #print(d_left, d_right)
    if d_left == None and d_right == None:
        print("all groups have been included")
        return None, None
    elif d_left == None or (d_right != None and d_right <= d_left):
        all_toas.select(all_toas.get_mjds() == right_dict[d_right])
        return d_right, all_toas.table['groups'][0]    
    else:
        all_toas.select(all_toas.get_mjds() == left_dict[d_left])
        return d_left, all_toas.table['groups'][0]    
    
datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, 'Ter5N.par.1')
timfile = os.path.join(datadir, 'Ter5N.tim')

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
a = np.logical_or(groups == 3, groups == 3)
#a = np.logical_and(groups == 8, groups == 8)
#a = np.logical_and(t.get_mjds() > 55829*u.d, t.get_mjds() < 55831*u.d)#groups == 25, groups == 26)
print(a)
t_sub = deepcopy(t)
t_sub.select(a)
print(t.table['groups'])


base_TOAs = pint.toa.get_TOAs(timfile)
#only modify base_TOAs with deletions and JUMPs
last_chi2 = 20000000000
cont = True
add_points = True
#jump all groups other than the ones given
jump_all_groups(m, base_TOAs, t, a)
been_in = False

while cont:
    print("AT THE START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    try_fit = True
    add_points = True
    #binary JUMP stuff
    #combine any jumps within a week of each other and with residuals differing by less than 0.05
    #go through all the jumps and (using get_closest_group) produce a dict with each jumps closest group and distance to that group
    #startning from smallest distance, find difference of residuals of edge points and if less than 0.05, combine into one jump
    #repeat until smallest distance > 1 week or residuals > 0.05
    resids = pint.residuals.Residuals(base_TOAs, m)
    resids = resids.phase_resids
    combination_dict = dict()
    jumps = [int(dict["jump"]) if "jump" in dict.keys() else np.nan for dict in base_TOAs.table["flags"]]
    print('jumps\n',jumps)
    unfit_toas = deepcopy(base_TOAs)
    nan_jumps = [True if np.isnan(item) else False for item in jumps]
    unfit_toas.select(nan_jumps)
    all_toas = deepcopy(base_TOAs)
    for jump_num in range(1, int(np.nanmax(jumps))):
        jump_select = [jump_num == item for item in jumps]
        if not any(jump_select):
            print('no group', jump_num)
            continue
        jumped_toas = deepcopy(base_TOAs)
        jumped_toas.select(jump_select)
        dist, closest_group = get_closest_group(all_toas, jumped_toas)
        if closest_group in unfit_toas.get_groups():
            dist = 10*u.d
            #make dist = 10 days so ignores that group and doesn't combine it
        combination_dict[dist] = [deepcopy(jumped_toas), closest_group, jump_num]
    
    for distance in sorted(combination_dict.keys()):
        if distance > 7*u.d: 
            #skip the rest
            break
        #starting from minimum sepertion, look at residual of edge points
        closest_toas = deepcopy(base_TOAs)
        closest_group = combination_dict[distance][1]
        b = np.logical_or(closest_toas.get_groups() == closest_group, closest_toas.get_groups() == closest_group)
        closest_toas.select(b)
        jumped_toas = combination_dict[distance][0]
        if closest_toas.get_mjds()[0] < jumped_toas.get_mjds()[0]:#closest group is to the left
            #find max toa of closest group and min of jumped group and compare their residuslas
            #if less than 0.05, combine closest group and jumped group into one jump
            print('left closest group')
            closest_mjd = max(closest_toas.get_mjds())
            jumped_mjd = min(jumped_toas.get_mjds())
            i1 = np.where(base_TOAs.get_mjds() == closest_mjd)
            i2 = np.where(base_TOAs.get_mjds() == jumped_mjd)
            r_diff = abs(resids[i1] - resids[i2])
            print(r_diff)
            if r_diff.value <= 0.05:
                jump_num = combination_dict[distance][2]
                jump_select = [jump_num == item for item in jumps]
                add_jump(m, base_TOAs, jumped_toas, jump_select)
                closest_jump = base_TOAs.table['flags'][list(base_TOAs.table['groups']).index(closest_group)]['jump']
                print('closest jump', closest_jump)
                jump_select1 = [closest_jump == item for item in jumps]
                jumped_toas1 = deepcopy(base_TOAs)
                jumped_toas1.select(jump_select1)
                add_jump(m, base_TOAs, jumped_toas1, jump_select1)
                jump_select2 = [a or b for a,b in zip(jump_select, jump_select1)]
                jumped_toas2 = deepcopy(base_TOAs)
                jumped_toas2.select(jump_select2)
                add_jump(m, base_TOAs, jumped_toas2, jump_select2)
                try_fit = False
                add_points = False
                been_in = True
                break
        elif closest_toas.get_mjds()[0] > jumped_toas.get_mjds()[0]:#closest group is to the right
            #find min toa of closest group and max of jumped group and compare their residuslas
            #if less than 0.05, combine closest group and jumped group into one jump
            print('right closest group')
            closest_mjd = min(closest_toas.get_mjds())
            jumped_mjd = max(jumped_toas.get_mjds())
            i1 = np.where(base_TOAs.get_mjds() == closest_mjd)
            i2 = np.where(base_TOAs.get_mjds() == jumped_mjd)
            r_diff = abs(resids[i1] - resids[i2])
            print(r_diff)
            if r_diff.value <= 0.05:
                jump_num = combination_dict[distance][2]
                closest_jump = base_TOAs.table['flags'][list(base_TOAs.table['groups']).index(closest_group)]['jump']
                print('closest jump', closest_jump)
                jump_select1 = [closest_jump == item for item in jumps]
                jumped_toas1 = deepcopy(base_TOAs)
                jumped_toas1.select(jump_select1)
                add_jump(m, base_TOAs, jumped_toas1, jump_select1)
                jump_select = [jump_num == item for item in jumps]
                add_jump(m, base_TOAs, jumped_toas, jump_select)
                jump_select2 = [a or b for a,b in zip(jump_select, jump_select1)]
                jumped_toas2 = deepcopy(base_TOAs)
                jumped_toas2.select(jump_select2)
                add_jump(m, base_TOAs, jumped_toas2, jump_select2)
                try_fit = False
                add_points = False
                been_in = True
                break
    
    print("CAT")
    print('try fit', try_fit, 'been in', been_in)
    if try_fit and been_in:
        been_in = False
        #dobule fit
        fj = pint.fitter.WlsFitter(base_TOAs, m)
        m = fj.model
        fj = pint.fitter.WlsFitter(base_TOAs, m)
        m = fj.model
        xt = base_TOAs.get_mjds()
        plt.plot(xt, pint.residuals.Residuals(base_TOAs, m).phase_resids, '.')
        plt.title("Try fit")
        plt.show()
        continue
        #fit twice with given toas and model and then do jumps again
        #if try to combine jumps again and none to be combined, run actual stuff
        
    if add_points:
        been_in = False
        # regular isolated stuff -- except always fitting all points, and not always adding closest group to unjumped points
        print(m.as_parfile())
        t = deepcopy(base_TOAs)
        t_others = deepcopy(base_TOAs)
        
        dist, closest_group = get_closest_group(deepcopy(t_others), deepcopy(t_sub))
        print('closest_group',closest_group)
        if closest_group == None:
            #end the program
            #maybe redo fit so actually fits with new model?
            print(m.as_parfile())
            cont = False
            continue
        #get closest group, t_others is t plus that group
        #right now t_others is just all the toas, so can use as all
        #need to figure out which ump closest group is in, add that entire jump to a, and then unjump it and apply it
        #from group get jump
        closest_jump = base_TOAs.table['flags'][list(base_TOAs.table['groups']).index(closest_group)]['jump']
        #from all jumps versus this jump get truth array
        jumps = [int(dict["jump"]) if "jump" in dict.keys() else np.nan for dict in base_TOAs.table["flags"]]
        b = [jump == closest_jump for jump in jumps] 
        #combine that array with a 
        a = [c or d for c,d in zip(a, b)]
        #remove jump in question
        b_toas = deepcopy(base_TOAs)
        b_toas.select(b)
        add_jump(m, base_TOAs, b_toas, b)
        t_others.select(a)
                
        do_ftest = True
        # Now do the fit
        print("Fitting...")
        f = pint.fitter.WlsFitter(base_TOAs, m)
        print("BEFORE:",f.get_fitparams())
        print(f.fit_toas())
        
        print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
        print("RMS in phase is", f.resids.phase_resids.std())
        print("RMS in time is", f.resids.time_resids.std().to(u.us))
        print("\n Best model is:")
        print(f.model.as_parfile())

        full_groups = base_TOAs.table['groups']
        selected = [True if group in t_sub.table['groups'] else False for group in full_groups] 
        rs_mean = pint.residuals.Residuals(base_TOAs, f.model, set_pulse_nums=True).phase_resids[selected].mean()
        f_toas, rss, rmods = rand.random_models(f, rs_mean, iter=12, ledge_multiplier=0, redge_multiplier=0)
        
        print('rs_mean',rs_mean)

        model0 = deepcopy(f.model)
        print('0 model chi2', f.resids.chi2)
        print('0 model chi2_ext', pint.residuals.Residuals(t_others, f.model).chi2)
        
        for i in range(len(rmods)):
            print('chi2',pint.residuals.Residuals(t_sub, rmods[i]).chi2)
            print('chi2 ext', pint.residuals.Residuals(t_others, rmods[i]).chi2)
            plt.plot(f_toas, rss[i], '-k', alpha=0.6)
            
        print(f.get_fitparams().keys())
        print(t_sub.ntoas)
        t_sub = deepcopy(base_TOAs)
        print(t_sub.ntoas)
        #plot post fit residuals with error bars
        xt = t.get_mjds()
#        mj = deepcopy(m)
#        comp_list = getattr(mj, "PhaseComponent_list")
#        for item in comp_list:
#            if isinstance(item, pint.models.jump.PhaseJump):
#                comp_list.remove(item)
#                break
#        mj.setup_components(comp_list)
#        print(mj.params)
        plt.errorbar(xt.value,
            pint.residuals.Residuals(t, f.model).time_resids.to(u.us).value,#f.resids.time_resids.to(u.us).value,
            t.get_errors().to(u.us).value, fmt='.b', label = 'post-fit')
    #    plt.plot(t.get_mjds(), pint.residuals.Residuals(t,m).time_resids.to(u.us).value, '.r', label = 'pre-fit')
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
        
        #m is new model
        m = chi2_dict[min_chi2]
        #a = current t plus closest group, defined above
        t_sub.select(a)
        
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
            if Ftest < 0.0008:
                #say model is model_plus (AKA, add the parameter)
                m = deepcopy(m_plus)
        print('can basically ignore after this ----------------------------------------------------------------------')    
    """
    #    f = pint.fitter.WlsFitter(t, model0)
    #    f.fit_toas()
    #    chi2_0_ext = pint.residuals.Residuals(t, f.model).chi2.value
        f = pint.fitter.WlsFitter(t, m)
        f.fit_toas()
        chi2_new_ext = pint.residuals.Residuals(t, f.model).chi2.value
        print(chi2_new_ext, 100*last_chi2)
        #get to this point, have a best fit model with or wthout a new parameter
        if chi2_new_ext > 100*last_chi2:
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
            if min_chi2 > 100*last_chi2:
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
        """
        #print('AT THE BOTTOM')
        #last_chi2 = deepcopy(min_chi2)
        #last_model = deepcopy(m)
        #last_t = deepcopy(t)
        #last_a = deepcopy(a)

    
