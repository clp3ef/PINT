import numpy as np
import pint.models.model_builder as mb
import os

#read in two models. Print the differences between their values, and the sigma difference given one is "solution"

datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile_1 = os.path.join(datadir, 'fake_data/fake_186.sol')
parfile_2 = os.path.join(datadir, 'fake_data/fake_186.fin')

m_1 = mb.get_model(parfile_1)
m_2 = mb.get_model(parfile_2)


print(m_1.as_parfile())
print(m_2.as_parfile())

F0_1 = float(m_1.F0.value)
F0_err_1 = float(m_1.F0.uncertainty.value)
RAJ_1 = float(m_1.RAJ.value)
RAJ_err_1 = float(m_1.RAJ.uncertainty.value)
DECJ_1 = float(m_1.DECJ.value)
DECJ_err_1 = float(m_1.DECJ.uncertainty.value)
F1_1 = float(m_1.F1.value)
F1_err_1 = float(m_1.F1.uncertainty.value) 

F0_2 = float(m_2.F0.value)
F0_err_2 = float(m_2.F0.uncertainty.value)
RAJ_2 = float(m_2.RAJ.value)
RAJ_err_2 = float(m_2.RAJ.uncertainty.value)
DECJ_2 = float(m_2.DECJ.value)
DECJ_err_2 = float(m_2.DECJ.uncertainty.value)
F1_2 = float(m_2.F1.value)
F1_err_2 = float(m_2.F1.uncertainty.value)

diff_F0 = F0_1 - F0_2
diff_RAJ = RAJ_1 - RAJ_2
diff_DECJ = DECJ_1 - DECJ_2
diff_F1 = F1_1 - F1_2

reldiff_F0 = diff_F0/np.sqrt(F0_err_1**2 + F0_err_2**2)
reldiff_RAJ = diff_RAJ/np.sqrt(RAJ_err_1**2 + RAJ_err_2**2)
reldiff_DECJ = diff_DECJ/np.sqrt(DECJ_err_1**2 + DECJ_err_2**2)
reldiff_F1 = diff_F1/np.sqrt(F1_err_1**2 + F1_err_2**2)
print(reldiff_F1)
print("F0  % 2.8e  % 2.1f  " %(diff_F0, reldiff_F0))
print("RAJ  % 2.8e  % 2.1f  " %(diff_RAJ, reldiff_RAJ))
print("DECJ  % 2.8e  % 2.1f  " %(diff_DECJ, reldiff_DECJ))
print("F1  % 2.8e  % 2.1f  " %(diff_F1, reldiff_F1))

