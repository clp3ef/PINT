import numpy as np
import matplotlib.pyplot as plt
import os
import pint.fitter
from pint.models import model_builder as mb

datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, 'NGC6440Etest.par')
timfile = os.path.join(datadir, 'NGC6440Etest1.tim')


# Define the timing model
m = mb.get_model(parfile)
t = pint.toa.get_TOAs(timfile)

print("Fitting...")
f = pint.fitter.WlsFitter(t, m)
print(f.fit_toas())
params = f.get_fitparams_num()
# Print some basic params
#remove first row and column of cov matrix
ucov_mat = (((f.unscaled_cov_matrix[1:]).T)[1:]).T
mean_vector = params.values()
fac = f.fac[1:]



#assume given mean vector, covariance matrix, and fac
print("mean vector", mean_vector)
print("errors", np.sqrt(np.diag(ucov_mat)))
#scale by fac for calculation
mean_vector *= fac
ucov_mat = ((ucov_mat*fac).T*fac).T
nums = [[],[],[]]
for i in range(200000):
        a,b,c = np.random.multivariate_normal(mean_vector,ucov_mat)
        nums[0].append(a)
        nums[1].append(b)
        nums[2].append(c)
#scale back to real units
mean_vector /= fac
for i in range(3):
    nums[i] /= fac[i]
ucov_mat = ((ucov_mat/fac).T/fac)
for i in range(3):
    data = nums[i]
    mean = np.mean(data)
    std = np.std(data)
    print(params.keys()[i]+" mean: "+str(mean)+" std: "+str(std))
    plt.hist(nums[i], bins=250, color='grey')
    plt.title(params.keys()[i])
    plt.show()
    
