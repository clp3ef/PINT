import pint.toa
import pint.models
import pint.utils
import os
import numpy.random as r
import numpy as np
import astropy.units as u
from copy import deepcopy

#write 3 files
#fake toas - unevenly distributed, randomized errors
#fake perfect parfile - solution to the fake system
#fake starting parfile - starting parfile smeared/dispersed by a random amount
#save the files in fake_data folder
print(os.listdir('./fake_data/'))
try:
    maxnum = max([int(filename[:-4][5:]) for filename in os.listdir('./fake_data/')])
except ValueError:
    #TODO: this error is also raised when max breaks, ie, not all the files are of the format ____#___ whatever
    maxnum = 0
    print('no files in the directory')


iter = 5
for num in range(maxnum+1, maxnum+1+iter):
    sol_name = 'fake_'+str(num)+'.sol'
    par_name = 'fake_'+str(num)+'.par'
    tim_name = 'fake_'+str(num)+'.tim'
    
    solfile = open('./fake_data/'+sol_name, 'w')
    h = r.randint(0,24)    
    m = r.randint(0,60)
    s = r.uniform(0,60)
    #error is whatever the precision on np.random is
    raj = (str(h)+':'+str(m)+':'+str(s),0.0000000001)
    d = r.randint(-89,90)
    arcm = r.randint(0,60)
    arcs = r.uniform(0,60)
    decj = (str(d)+':'+str(arcm)+':'+str(arcs),0.0000000001)
    f0 = (r.uniform(0.2,3), 0.0000000001)
    #20 between 0.2 and 3 and 20 between 3 and 100
    if f0[0] < 1000 and f0[0] > 100:
        f1 = (10**(r.randint(-16,-14)), 0.0000000001)
    elif f0[0] < 100 and f0[0] > 10:
        f1 = (10**(r.randint(-16,-15)), 0.0000000001)
    elif f0[0] < 10 and f0[0] > 0.1:
        f1 = (10**(r.randint(-16,-11)), 0.0000000001)
    else:
        f1 = (10**(-16), 0.0000000001)
    dm = (r.uniform(5,70), 0.0000001)
    pepoch = 56000
    tzrmjd = 56000
    tzrfrq = 1400
    tzrsite = 'GBT'
    
    
    solfile.write('PSR\t'+sol_name[:-4]+'\n')
    solfile.write('RAJ\t'+str(raj[0])+'\t\t1\t'+str(raj[1])+'\n')
    solfile.write('DECJ\t'+str(decj[0])+'\t\t1\t'+str(decj[1])+'\n')
    solfile.write('F0\t'+str(f0[0])+'\t\t1\t'+str(f0[1])+'\n')
    solfile.write('F1\t'+str(f1[0])+'\t\t1\t'+str(f1[1])+'\n')
    solfile.write('DM\t'+str(dm[0])+'\t\t1\t'+str(dm[1])+'\n')
    solfile.write('PEPOCH\t'+str(pepoch)+'\n')
    solfile.write('TZRMJD\t'+str(tzrmjd)+'\n')
    solfile.write('TZRFRQ\t'+str(tzrfrq)+'\n')
    solfile.write('TZRSITE\t'+tzrsite)
    
    solfile.close()

    print("start")
    print(raj, decj, f0, f1, sep='\n')
    raj = [float(s),]
    decj = [float(arcs),]
    start = [deepcopy(param[0]) for param in [raj, decj, f0, f1]]
    print(start)
    
    parfile = open('./fake_data/'+par_name, 'w')
    rblur = 1.3*r.standard_normal()
    raj = (str(h)+':'+str(m)+':'+str(s+rblur),0.01)
    dblur = 5*r.standard_normal()
    decj = (str(d)+':'+str(arcm)+':'+str(arcs+dblur),0.01)
    #2e-4 HZ roughly error from 1000 sec obsv
    f0blur = 4e-4*r.standard_normal()#0.0000005*r.standard_normal()
    f0 = (f0[0]+f0blur, 0.000001)
    f1 = (0.0, 0.0)
    dmblur = 0#2*r.standard_normal()
    dm = (dm[0]+dmblur, 0.0)
    pepoch = 56000
    tzrmjd = 56000
    tzrfrq = 1400
    tzrsite = 'GBT'
    
    parfile.write('PSR\t'+par_name[:-4]+'\n')
    parfile.write('RAJ\t'+str(raj[0])+'\t0\t'+str(raj[1])+'\n')
    parfile.write('DECJ\t'+str(decj[0])+'\t0\t'+str(decj[1])+'\n')
    parfile.write('F0\t'+str(f0[0])+'\t1\t'+str(f0[1])+'\n')
    parfile.write('F1\t'+str(f1[0])+'\t\t\t0\t'+str(f1[1])+'\n')
    parfile.write('DM\t'+str(dm[0])+'\t0\t'+str(dm[1])+'\n')
    parfile.write('PEPOCH\t'+str(pepoch)+'\n')
    parfile.write('TZRMJD\t'+str(tzrmjd)+'\n')
    parfile.write('TZRFRQ\t'+str(tzrfrq)+'\n')
    parfile.write('TZRSITE\t'+tzrsite)
    
    parfile.close()
    
    print("end")
    print(raj, decj, f0, f1, sep='\n')
    raj = [float(s+rblur),]
    decj = [float(arcs+dblur),]
    end = [deepcopy(param[0]) for param in [raj, decj, f0, f1]]
    
    for i in range(len(start)):
        per_diff = 100*(start[i]-end[i])/start[i]
        print('param '+str(i)+': % diff '+str(float(per_diff)))
        
    #use zima to write the parfile into a timfile
    #ntoas, duration, error?, startmjd, fuzzdays?
    #duration - 300 to 1200 days
    #density from toa every 0.004 days (6 min) to 0.02 days (30 min)
    density = r.uniform(0.004, 0.02)
    duration = int(r.uniform(200,700))
    ntoas = int(duration/density)
    #1 observation is a set of anywhere from 1 to 8 consecutive toas
    #2 obs on 1 day, then obs 3 of 5 days, then 2 of next 10 days, then 1 a week later, then monthly
    #have to convert from days to indexes - 1 day = (1/density) toas-indexes = day/density 
    #randomize: length of obsv, exact time between observations, 
    
    #number of observations in each group
    #length between each observation for each group
    #length of each observation
    # 2,            2-4,          2-4,      1-3,    monthly until end
    # 0.1 - 0.9 d, 0.8 - 2.2 d, 4 - 7 d, 6 - 14 d, 20-40 d
    # 1-10 toas
    d1 = [int(r.uniform(0.1,0.9)/density) for i in range(2)]
    d2 = [int(r.uniform(0.8,2.2)/density) for i in range(r.randint(2,4))]
    d3 = [int(r.uniform(4,7)/density) for i in range(r.randint(2,4))]
    d4 = [int(r.uniform(6,14)/density) for i in range(r.randint(1,3))]
    print(d1)
    print(d2)
    print(d3)
    print(d4)
    distances = d1 + d2 + d3 + d4
    print(distances)
    mask = np.zeros(ntoas, dtype = bool)
    i = 0
    for distance in distances:
        obs_length = r.randint(1,8)
        mask[i:i+obs_length] = ~mask[i:i+obs_length]
        i = i + obs_length + distance
    print('^'*100)
    while i < ntoas:
        obs_length = r.randint(1,8)
        distance = int(r.uniform(20,40)/density)
        mask[i:i+obs_length] = ~mask[i:i+obs_length]
        i = i + obs_length + distance

    #error = 10
    #startmjd - 56000
    print('zima ./fake_data/' + sol_name + ' ./fake_data/' + tim_name + ' --ntoa '+ str(ntoas) + ' --duration ' + str(duration)+' --error 50')
    os.system('zima ./fake_data/' + sol_name + ' ./fake_data/' + tim_name + ' --ntoa '+ str(ntoas) + ' --duration ' + str(duration)+' --error 50')
    
    t = pint.toa.get_TOAs('./fake_data/'+tim_name)
    t.table = t.table[mask].group_by("obs")
    #a = percent of the data to remove
    #a = r.uniform(0.2, 0.45)
    #print(a)
    #for i in range(int(ntoas*a)):
    #    remove = np.zeros(t.ntoas,dtype = bool)
    #    j = r.randint(0,len(remove))
    #    remove[j] = True
    #    t.table = t.table[~remove].group_by("obs")
    
    #mjds = t.get_mjds()
    #minmjd = min(mjds).value
    #maxmjd = max(mjds).value
    #n_centers = r.randint(200, 500)
    #a = r.uniform(1.2, 3)
    #print(a)
    #avg_diam = a*(maxmjd-minmjd)/n_centers
    #print(n_centers)
    #print(avg_diam)
    #for i in range(n_centers):
    #    center = r.uniform(minmjd, maxmjd)
    #    diam = avg_diam*((0.8*r.standard_normal())+0.9)
    #    lower_mjd = center - (diam/2)
    #    upper_mjd = center + (diam/2)
    #    q = np.logical_and(t.get_mjds() < upper_mjd*u.d, t.get_mjds() > lower_mjd*u.d)
    #    t.table = t.table[~q].group_by("obs")

    print(t.table['groups'][:10])
    print("groups" in t.table.columns)
    del t.table['groups']
    print("groups" in t.table.columns)
    t.table['groups'] = t.get_groups()
    print(t.table['groups'][:10])
    t.write_TOA_file('./fake_data/'+tim_name, format = 'TEMPO2')
    #save timfile
    
    #then remove random parts using random number of centers and random circumferences (scaling with numbrof ceners)
    #re-get groups so not one big group




