from __future__ import absolute_import, print_function, division
import astropy.units as u
from astropy import log
import numpy as np
from .phase import Phase
from pint import dimensionless_cycles
import scipy.linalg as sl
from copy import deepcopy

class resids(object):
    """resids(toa=None, model=None)"""

    def __init__(self, toas=None, model=None, weighted_mean=True, set_pulse_nums=False):
        self.toas = toas
        self.model = model
        if toas is not None and model is not None:
            self.phase_resids = self.calc_phase_resids(weighted_mean=weighted_mean, set_pulse_nums=set_pulse_nums)
            self.time_resids = self.calc_time_resids(weighted_mean=weighted_mean)
            #self.pulse_numbers = self.calc_pulse_numbers()
            self.chi2 = self.calc_chi2()
            self.dof = self.get_dof()
            self.chi2_reduced = self.chi2 / self.dof
        else:
            self.phase_resids = None
            self.time_resids = None

    def calc_phase_resids(self, weighted_mean=True, set_pulse_nums=False):
        """Return timing model residuals in pulse phase."""
        rs = self.model.phase(self.toas)
        rs -= Phase(rs.int[0],rs.frac[0])
        try:
            delta_pulse_numbers = Phase(self.toas.table['delta_pulse_numbers'])
        except:
            self.toas.table['delta_pulse_numbers'] = np.zeros(len(self.toas.get_mjds()))
            delta_pulse_numbers = Phase(self.toas.table['delta_pulse_numbers'])
        if set_pulse_nums:
            self.toas.table['delta_pulse_numbers'] = np.zeros(len(self.toas.get_mjds()))
            delta_pulse_numbers = Phase(self.toas.table['delta_pulse_numbers'])
        full = Phase(np.zeros_like(rs.frac),rs.frac) + delta_pulse_numbers
        full = full.int + full.frac
        
        if not weighted_mean:
            rs -= Phase(0.0,(full).mean())
        else:
        # Errs for weighted sum.  Units don't matter since they will
        # cancel out in the weighted sum.
            if np.any(self.toas.get_errors() == 0):
                raise ValueError('TOA errors are zero - cannot calculate residuals')
            w = 1.0/(np.array(self.toas.get_errors())**2)
            wm = ((full)*w).sum() / w.sum()
            rs -= Phase(0.0,wm)
        return full

    def calc_time_resids(self, weighted_mean=True):
        """Return timing model residuals in time (seconds)."""
        if self.phase_resids is None:
            self.phase_resids = self.calc_phase_resids(weighted_mean=weighted_mean)
        with u.set_enabled_equivalencies(dimensionless_cycles):
            return (self.phase_resids.to(u.Unit("")) / self.get_PSR_freq()).to(u.s)

    def get_PSR_freq(self, modelF0=True):
        if modelF0:
            """Return pulsar rotational frequency in Hz. model.F0 must be defined."""
            if self.model.F0.units != 'Hz':
                ValueError('F0 units must be Hz')
            # All residuals require the model pulsar frequency to be defined
            F0names = ['F0', 'nu'] # recognized parameter names, needs to be changed
            nF0 = 0
            for n in F0names:
                if n in self.model.params:
                    F0 = getattr(self.model, n).value
                    nF0 += 1
            if nF0 == 0:
                raise ValueError('no PSR frequency parameter found; ' +
                                 'valid names are %s' % F0names)
            if nF0 > 1:
                raise ValueError('more than one PSR frequency parameter found; ' +
                                 'should be only one from %s' % F0names)
            return F0 * u.Hz
        return self.model.d_phase_d_toa(self.toas)

    def calc_chi2(self):
        """Return the weighted chi-squared for the model and toas."""
        # Residual units are in seconds. Error units are in microseconds.
        if (self.toas.get_errors()==0.0).any():
            return np.inf
        else:
            # The self.time_resids is in the unit of "s", the error "us".
            # This is more correct way, but it is the slowest.
            #return (((self.time_resids / self.toas.get_errors()).decompose()**2.0).sum()).value

            # This method is faster then the method above but not the most correct way
            #return ((self.time_resids.to(u.s) / self.toas.get_errors().to(u.s)).value**2.0).sum()

            # This the fastest way, but highly depend on the assumption of time_resids and
            # error units.
            return ((self.time_resids / self.toas.get_errors().to(u.s))**2.0).sum()

    def get_dof(self):
        """Return number of degrees of freedom for the model."""
        dof = self.toas.ntoas
        for p in self.model.params:
            dof -= bool(not getattr(self.model, p).frozen)
        return dof
    
    def get_reduced_chi2(self):
        """Return the weighted reduced chi-squared for the model and toas."""
        return self.calc_chi2() / self.get_dof()
    
    def update(self, weighted_mean=True):
        """Recalculate everything in residuals class
        after changing model or TOAs"""
        if self.toas is None or self.model is None:
            self.phase_resids = None
            self.time_resids = None
        if self.toas is None:
            raise ValueError('No TOAs provided for residuals update')
        if self.model is None:
            raise ValueError('No model provided for residuals update')

        self.phase_resids = self.calc_phase_resids(weighted_mean=weighted_mean)
        self.time_resids = self.calc_time_resids(weighted_mean=weighted_mean)
        self.chi2 = self.calc_chi2()
        self.dof = self.get_dof()
        self.chi2_reduced = self.chi2 / self.dof
