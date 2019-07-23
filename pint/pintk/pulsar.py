'''
A wrapper around pulsar functions for pintkinter to use. This object will be shared
between widgets in the main frame and will contain the pre/post fit model, toas, 
pre/post fit residuals, and other useful information.
'''
from __future__ import print_function
from __future__ import division
import os, sys

# Numpy etc.
import numpy as np
import astropy.units as u
from astropy.time import Time
from enum import Enum
import copy

#Pint imports
import pint.models
import pint.toa
import pint.fitter
import pint.residuals
import pint.random_models


plot_labels = ['pre-fit', 'post-fit', 'mjd', 'year', 'orbital phase', 'serial', \
    'day of year', 'frequency', 'TOA error', 'rounded MJD']

# Some parameters we do not want to add a fitting checkbox for:
nofitboxpars = ['PSR', 'START', 'FINISH', 'POSEPOCH', 'PEPOCH', 'DMEPOCH', \
    'EPHVER', 'TZRMJD', 'TZRFRQ', 'TRES', 'PLANET_SHAPIRO']

class Fitters(Enum):
    POWELL = 0
    WLS = 1
    GLS = 2

class Pulsar(object):
    '''
    Wrapper class for a pulsar. Contains the toas, model, residuals, and fitter
    '''

    def __init__(self, parfile=None, timfile=None, ephem=None):
        super(Pulsar, self).__init__()
        
        print('STARTING LOADING OF PULSAR %s' % str(parfile))
        
        if parfile is not None and timfile is not None:
            self.parfile = parfile
            self.timfile = timfile
        else:
            raise ValueError("No valid pulsar to load")

        self.prefit_model = pint.models.get_model(self.parfile)
        print("prefit_model.as_parfile():")
        print(self.prefit_model.as_parfile())

        if ephem is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=ephem, planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=ephem, planets=True)
            self.prefit_model.EPHEM.value = ephem
        elif getattr(self.prefit_model, 'EPHEM').value is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
        else:
            self.toas = pint.toa.get_TOAs(self.timfile,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile,planets=True)
        self.toas.print_summary()
        
        self.prefit_resids = pint.residuals.resids(self.toas, self.prefit_model)
        print("RMS PINT residuals are %.3f us\n" % \
              self.prefit_resids.time_resids.std().to(u.us).value)
        self.fitter = Fitters.WLS
        self.fitted = False
        self.track_added = False
        
    @property
    def name(self):
        return getattr(self.prefit_model, 'PSR').value

    def __getitem__(self, key):
        try:
            return getattr(self.prefit_model, key)
        except AttributeError:
            print('Parameter %s was not found in pulsar model %s' % (key, self.name))
            return None

    def __contains__(self, key):
        return key in self.prefit_model.params
   
    def reset_model(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.update_resids()

    def reset_TOAs(self):

        if getattr(self.prefit_model, 'EPHEM').value is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
        else:
            self.toas = pint.toa.get_TOAs(self.timfile,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile,planets=True)
            
        if self.track_added:
            self.prefit_model.TRACK.value = ''
            if self.fitted:
                self.postfit_model.TRACK.value = ''
            self.track_added = False
        self.update_resids()

    def resetAll(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False

        if getattr(self.prefit_model, 'EPHEM').value is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
        else:
            self.toas = pint.toa.get_TOAs(self.timfile,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            
        self.update_resids()
   
    def update_resids(self):
        self.prefit_resids = pint.residuals.resids(self.fulltoas, self.prefit_model)
        if self.fitted:
            self.postfit_resids = pint.residuals.resids(self.fulltoas, self.postfit_model)

    def orbitalphase(self):
        '''
        For a binary pulsar, calculate the orbital phase. Otherwise, return
        an array of zeros
        '''
        if 'PB' not in self:
            print("WARNING: This is not a binary pulsar")
            return np.zeros(len(self.toas))

        toas = self.toas.get_mjds()

        if 'T0' in self and not self['T0'].quantity is None:
            tpb = (toas.value - self['T0'].value) / self['PB'].value
        elif 'TASC' in self and not self['TASC'].quantity is None:
            tpb = (toas.value - self['TASC'].value) / self['PB'].value
        else:
            print("ERROR: Neither T0 nor TASC set")
            return np.zeros(len(toas))
        
        phase = np.modf(tpb)[0]
        phase[phase < 0] += 1
        return phase * u.cycle

    def dayofyear(self):
        '''
        Return the day of the year for all the TOAs of this pulsar
        '''
        t = Time(self.toas.get_mjds(), format='mjd')
        year = Time(np.floor(t.decimalyear), format='decimalyear')
        return (t.mjd - year.mjd) * u.day

    def year(self):
        '''
        Return the decimal year for all the TOAs of this pulsar
        '''
        t = Time(self.toas.get_mjds(), format='mjd')
        return (t.decimalyear) * u.year
    
    def write_fit_summary(self):
        '''
        Summarize fitting results
        '''
        if self.fitted:
            chi2 = self.postfit_resids.chi2
            wrms = np.sqrt(chi2 / self.toas.ntoas)
            print('Post-Fit Chi2:\t\t%.8g us^2' % chi2)
            print('Post-Fit Weighted RMS:\t%.8g us' % wrms)
            print('%19s  %24s\t%24s\t%16s  %16s  %16s' % 
                  ('Parameter', 'Pre-Fit', 'Post-Fit', 'Uncertainty', 'Difference', 'Diff/Unc'))
            print('-' * 132)
            fitparams = [p for p in self.prefit_model.params 
                         if not getattr(self.prefit_model, p).frozen]
            for key in fitparams:
                line = '%8s ' % key
                pre = getattr(self.prefit_model, key)
                post = getattr(self.postfit_model, key)
                line += '%10s  ' % ('' if post.units is None else str(post.units))
                if post.quantity is not None:
                    line += '%24s\t' % pre.print_quantity(pre.quantity)
                    line += '%24s\t' % post.print_quantity(post.quantity)
                    try:
                        line += '%16.8g  ' % post.uncertainty.value
                    except:
                        line += '%18s' % ''
                    try:
                        diff = post.value - pre.value
                        line += '%16.8g  ' % diff
                        if pre.uncertainty is not None:
                            line += '%16.8g' % (diff / pre.uncertainty.value)
                    except:
                        pass
                print(line)
        else:
            print('Pulsar has not been fitted yet!')

    def add_phase_wrap(self, selected, phase):
        """
        Add a phase wrap to selected points in the TOAs object
        Turn on pulse number tracking in the model, if it isn't already
        """
        #Check if pulse numbers are in table already
        if 'pn' not in self.fulltoas.table.colnames:
            self.fulltoas.compute_pulse_numbers(self.prefit_model)
            self.toas.compute_pulse_numbers(self.prefit_model)
        if 'delta_pulse_numbers' not in self.fulltoas.table.colnames:
            self.fulltoas.table['delta_pulse_numbers'] = np.zeros(len(self.fulltoas.get_mjds()))
            self.toas.table['delta_pulse_numbers'] = np.zeros(len(self.toas.get_mjds()))
            
        self.fulltoas.table['delta_pulse_numbers'][selected] += phase
        
        #Turn on pulse number tracking
        if self.prefit_model.TRACK.value != '-2':
            self.track_added = True
            self.prefit_model.TRACK.value = '-2'
            if self.fitted:
                self.postfit_model.TRACK.value = '-2'

        self.update_resids()
        
    def add_jump(self, selected):
        """
        jump the toas selected (add a new parameter to the model) or unjump them (remove the jump parameter) if already jumped
        """
        mjds = self.fulltoas.table['mjd_float'][selected]
        minmjd = min(mjds)
        maxmjd = max(mjds)
        if "PhaseJump" not in self.prefit_model.components:
            print("No PhaseJump component until now")
            a = pint.models.jump.PhaseJump()
            a.setup()
            self.prefit_model.add_component(a)
            self.prefit_model.remove_param("JUMP1")
            param = pint.models.parameter.maskParameter(name = 'JUMP', index=1, key='mjd', key_value = [minmjd, maxmjd], frozen = False, value = 0.0, units = 'second')
            self.prefit_model.add_param_from_top(param, "PhaseJump")
            if self.fitted:
                self.postfit_model.add_component(a)
            return None
        
        ranges = []
        for param in self.prefit_model.params:
            if param.startswith("JUMP"):
                ranges.append(getattr(self.prefit_model,param).key_value+[getattr(self.prefit_model,param)])
        print(ranges)
        nums = []
        for r in ranges:
            print(r[0],r[1])
            print(minmjd == r[0], maxmjd == r[1])
            nums.append(int(r[2].name[4:]))
            if minmjd == r[0] and maxmjd == r[1]:
                self.prefit_model.remove_param(r[2].name)
                ranges_subset = ranges[ranges.index(r):]
                print(ranges_subset)
                c = True
                for rr in ranges_subset:
                    if c:#skip first loop
                        c = False
                        continue
                    param = pint.models.parameter.maskParameter(name = 'JUMP', index=int(rr[2].name[4:])-1, key='mjd', key_value = [rr[0], rr[1]], 
                    frozen = rr[2].frozen, value = rr[2].value, units = 'second')
                    self.prefit_model.add_param_from_top(param, 'PhaseJump')
                    self.prefit_model.remove_param(rr[2].name)
                if "JUMP1" not in self.prefit_model.params:
                    comp_list = getattr(self.prefit_model, 'PhaseComponent_list')
                    print(comp_list)
                    for item in comp_list:
                        if isinstance(item, pint.models.jump.PhaseJump):
                            comp_list.remove(item)
                            break
                    print(comp_list)
                    self.prefit_model.setup_components(comp_list)
                    if self.fitted:
                        self.postfit_model.setup_components(comp_list)
                else:
                    self.prefit_model.components["PhaseJump"].setup()
                print("removed param", r[2].name)
                print(self.prefit_model.params)
                return None#end the function call
            elif (r[0] <= minmjd and minmjd <= r[1]) or (r[0] <= maxmjd and maxmjd <= r[1]):
                print("Cannot JUMP toas that have already been jumped, check for overlap.")
                return None#end the function call
        #if doesn't overlap or cancel, add it to the model
        if nums == []:
            param = pint.models.parameter.maskParameter(name = 'JUMP', index=1, key='mjd', key_value = [minmjd, maxmjd], frozen = False, value = 0.0, units = 'second')
            self.prefit_model.add_param_from_top(param, "PhaseJump")
            return None
        
        param = pint.models.parameter.maskParameter(name = 'JUMP', index=max(nums)+1, key='mjd', key_value = [minmjd, maxmjd], frozen = False, value = 0.0, units = 'second')
        self.prefit_model.add_param_from_top(param, "PhaseJump")
        print(self.prefit_model.params)
        self.prefit_model.components["PhaseJump"].setup()
        #return None
    
        #pfile = open(self.parfile, 'r')
        #ranges = []
        #text = []
        #count = 0
        #for line in pfile:
        #    text.append(line)
        #    if line.startswith("JUMP"):
        #        line = line.split()
        #        ranges.append((count,line[0],float(line[2]),float(line[3])))
        #    count += 1
        #pfile.close()
        #for r in ranges:
        #    print(r[2],r[3])
        #    print(minmjd == r[2], maxmjd == r[3])
        #    if minmjd == r[2] and maxmjd == r[3]:
        #        del text[r[0]]#delete the jump line 
        #        print('whole text file',''.join(text))
        #        pfile = open(self.parfile, 'w')
        #        pfile.write(''.join(text))
        #        pfile.close()
        #        self.prefit_model = pint.models.get_model(self.parfile)
        #        return None#end the function call
        #    elif (r[2] <= minmjd and minmjd <= r[3]) or (r[2] <= maxmjd and maxmjd <= r[3]):
        #        print("Cannot JUMP toas that have already been jumped, check for overlap.")
        #        return None#end the function call
        #
        ##if min and max match an existing jump, then delete the jump
        ##elif min and max overlap an existing jump, raise an error
        ##else, add the new jump to the file
        #nums = []
        #for param in self.prefit_model.params:
        #    if 'JUMP' in param:
        #        nums.append(int(param[4:]))
        #try:
        #    jumpnum = str(max(nums)+1)
        #except:
        #    jumpnum = '1'
        #line = 'JUMP'+jumpnum+' mjd '+str(minmjd)+' '+str(maxmjd)+' 0.0 1\n'
        #pfile = open(self.parfile, 'a')
        #pfile.write(line)
        #pfile.close()
        #self.prefit_model = pint.models.get_model(self.parfile)
            
    def fit(self, iters=1):
        '''
        Run a fit using the specified fitter
        '''
        if self.fitted:
            self.prefit_model = self.postfit_model
            self.prefit_resids = self.postfit_resids
            
        """JUMP check, put in fitter?"""
        if "PhaseJump" in self.prefit_model.components:
            mjds = self.toas.table['mjd_float']
            mjds_copy = list(copy.deepcopy(mjds))
            minmjd = min(mjds)
            maxmjd = max(mjds)
            
            prefit_save = copy.deepcopy(self.prefit_model)
            for param in self.prefit_model.params:
                if param.startswith("JUMP") and getattr(self.prefit_model, param).frozen == False:
                    minmax = getattr(self.prefit_model,param).key_value
                    #checks if selected toas are all jumped and returns error if they all are
                    if minmax[0] in mjds_copy:
                        if minmax[1] in mjds_copy:
                            mjds_copy[mjds_copy.index(minmax[0]):mjds_copy.index(minmax[1])+1] = []
                        else:
                            setattr(getattr(self.prefit_model, param), 'key_value', [minmax[0],maxmjd])
                            mjds_copy[mjds_copy.index(minmax[0]):] = []
                    elif minmax[1] in mjds_copy:
                        setattr(getattr(self.prefit_model, param), 'key_value', [minmjd, minmax[1]])
                        mjds_copy[:mjds_copy.index(minmax[1])+1]
                    elif minmax[0] < minmjd and minmax[1] > maxmjd:
                        #dont bother resizing the jump range because its going to reset anyways
                        mjds_copy = []
                    else:
                        #if being fit for but jump entirely outside range, uncheck it
                        print("outside range")
                        setattr(getattr(self.prefit_model, param), 'frozen', True)
                    print(minmax[0],minmax[1])
                    print(mjds_copy)
                    if mjds_copy == []:
                        self.prefit_model = prefit_save
                        print("toas being fit must not all be jumped. Remove or uncheck at least one jump in the selected toas before fitting.")
                        return None
        
        if self.fitter == Fitters.POWELL:
            fitter = pint.fitter.PowellFitter(self.toas, self.prefit_model)
        elif self.fitter == Fitters.WLS:
            fitter = pint.fitter.WlsFitter(self.toas, self.prefit_model)
        elif self.fitter == Fitters.GLS:
            fitter = pint.fitter.GLSFitter(self.toas, self.prefit_model)
        chi2 = self.prefit_resids.chi2
        wrms = np.sqrt(chi2 / self.toas.ntoas)
        print('Pre-Fit Chi2:\t\t%.8g us^2' % chi2)
        print('Pre-Fit Weighted RMS:\t%.8g us' % wrms)
        
        fitter.fit_toas(maxiter=1)
        self.postfit_model = fitter.model
        self.postfit_resids = pint.residuals.resids(self.fulltoas, self.postfit_model, set_pulse_nums = True)
        self.fitted = True
        self.write_fit_summary()
        

        q = list(self.fulltoas.get_mjds())
        index = q.index([i for i in self.fulltoas.get_mjds() if i > self.toas.get_mjds().min()][0])
        rs_mean = pint.residuals.resids(self.fulltoas,fitter.model, set_pulse_nums=True).phase_resids[index:index+len(self.toas.get_mjds())].mean()
        if len(fitter.get_fitparams()) < 3:
            redge = ledge = 30
            npoints = 400
        else:
            redge = ledge = 2.5
            npoints = 100
        f_toas, rs = pint.random_models.random(fitter, rs_mean=rs_mean, redge_multiplier=redge, ledge_multiplier=ledge, iter=10, npoints=npoints)
        self.random_resids = rs
        self.fake_toas = f_toas
            
