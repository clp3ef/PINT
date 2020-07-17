#!/usr/bin/env python
import pint.toa
import pint.models
import pint.utils
import os
import numpy.random as r
import numpy as np
import astropy.units as u
from copy import deepcopy


def write_solfile(args, sol_name):
    '''docstring'''
    solfile = open('./fake_data/'+sol_name, 'w')
    h = r.randint(0,24)
    m = r.randint(0,60)
    s = r.uniform(0,60)
    #randomly assign values in appropriate ranges to RAJ
    if args.RAJ_value == None:              
        raj = (str(h)+':'+str(m)+':'+str(s),args.RAJ_error)
    else:
        raj = (args.RAJ_value, args.RAJ_error)
    d = r.randint(-89,90)
    arcm = r.randint(0,60)
    arcs = r.uniform(0,60)
    #randomly assign values in appropriate ranges to DECJ
    if args.DECJ_value == None:
        decj = (str(d)+':'+str(arcm)+':'+str(arcs),args.DECJ_error)
    else:
        decj = (args.DECJ_value, args.DECJ_error)
    #randomly assign values in apporiate range to F0 (100-800 Hz is millisecond pulsar range. Slow pulsars are 2-20 Hz range)
    if args.F0_value == None:    
        f0 = (r.uniform(100,800), args.F0_error)
    else:
        f0 = (args.F0_value, args.F0_error)
        
    #assign F1 based on general ranges in F-Fdot diagram (see ****)
    if f0[0] < 1000 and f0[0] > 100:
        f1 = (10**(r.randint(-16,-14)), args.F1_error)
    elif f0[0] < 100 and f0[0] > 10:
        f1 = (10**(r.randint(-16,-15)), args.F1_error)
    elif f0[0] < 10 and f0[0] > 0.1:
        f1 = (10**(r.randint(-16,-11)),  args.F1_error)
    else:
        f1 = (10**(-16),  args.F1_error)
        
    if type(args.F1_value) == float:
        f1 = (args.F1_value, args.F1_error)
    
    #assign DM param to be zero
    dm = (0., 0.)
    
    #assign positional parameters
    pepoch = args.PEPOCH
    tzrmjd = args.TZRMJD
    tzrfrq = args.TZRFRQ
    tzrsite = args.TZRSITE
    
    #save the value of F0 for later use
    f0_save = deepcopy(f0[0])
    
    #write the lines to the solution parfile in TEMPO2 format 
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
    
    return f0_save, h, m, s, d, arcm, arcs, f0, f1, dm
    
def write_timfile(args, f0_save, tim_name, sol_name):
    '''docstring'''
    
    #use zima to write the parfile into a timfile
    #duration - 300 to 1200 days
    #density from toa every 0.004 days (6 min) to 0.02 days (30 min)
    density = r.uniform(args.density_range[0], args.density_range[1])
    duration = int(r.uniform(args.span[0], args.span[1]))
    ntoas = int(duration/density)
    #1 observation is a set of anywhere from 1 to 8 consecutive toas
    #2 obs on 1 day, then obs 3 of 5 days, then 2 of next 10 days, then 1 a week later, then monthly
    #n_obs per timespan  2, 2-4, 2-4, 1-3, monthly until end
    #length between each obsevration for each timespan 0.1 - 0.9 d, 0.8 - 2.2 d, 4 - 7 d, 6 - 14 d, 20-40 d
    # 1-8 toas
    d1 = [int(r.uniform(0.1,0.9)/density) for i in range(2)]
    d2 = [int(r.uniform(0.8,2.2)/density) for i in range(r.randint(2,4))]
    d3 = [int(r.uniform(4,7)/density) for i in range(r.randint(2,4))]
    d4 = [int(r.uniform(6,14)/density) for i in range(r.randint(1,3))]
    distances = d1 + d2 + d3 + d4
    
    #make a mask which only allows TOAs to exist on those spans specified by the distances above
    mask = np.zeros(ntoas, dtype = bool)
    i = 0
    count = 0
    for distance in distances:
        count += 1
        if count <= 2:
            #for first two observations, allow 3 to 8 TOAs per observation
            obs_length = r.randint(3,8)
            ntoa2 = obs_length
        else:
            obs_length = r.randint(1,8)
        mask[i:i+obs_length] = ~mask[i:i+obs_length]
        i = i + obs_length + distance
    #once distance list is used up, continue adding observations ~monthly until end of TOAs is reached
    while i < ntoas:
        obs_length = r.randint(1,8)
        distance = int(r.uniform(20,40)/density)
        mask[i:i+obs_length] = ~mask[i:i+obs_length]
        i = i + obs_length + distance
            
    #maximum possible residual based on F0
    max_resid = (0.5/f0_save)*10**6
    #randomly chosen error for TOAs
    percent = r.uniform(0.0003,0.03)
    #error = percent*max resid, scale error relevant to possible residual difference
    error = int(max_resid * percent)
    #startmjd = 56000, always
    
    #run zima with the parameters given, this may take a long time is the number of TOAs is high (i.e. over 20000)
    print('zima ./fake_data/' + sol_name + ' ./fake_data/' + tim_name + ' --ntoa '+ str(ntoas) + ' --duration ' + str(duration)+' --error '+str(error))
    os.system('zima ./fake_data/' + sol_name + ' ./fake_data/' + tim_name + ' --ntoa '+ str(ntoas) + ' --duration ' + str(duration)+' --error '+str(error))
    
    #turn the TOAs into a TOAs object and use the mask to remove all TOAs not in the correct ranges
    t = pint.toa.get_TOAs('./fake_data/'+tim_name)
    t.table = t.table[mask].group_by("obs")
    
    #reset the TOA table group column
    print(t.table['groups'][:10])
    print("groups" in t.table.columns)
    del t.table['groups']
    print("groups" in t.table.columns)
    t.table['groups'] = t.get_groups()
    print(t.table['groups'][:10])
    #save timfile
    t.write_TOA_file('./fake_data/'+tim_name, format = 'TEMPO2')
    return ntoa2, density
    
def write_parfile(args, par_name, h, m, s, d, arcm, arcs, f0, f1, dm, ntoa2, density):
    '''docstring'''
    #write parfile as a skewed version of the solution file, the same way real data is a corrupted or blurred version of the "true" nature of the distant pulsar
    parfile = open('./fake_data/'+par_name, 'w')
    #read in argument blurring factors, or chose them from scaled gaussian distributions
    if args.rblur != None:
        rblur = args.rblur
    else:
        rblur = args.rblur_coeff*r.standard_normal()
    #blur RAJ by the amount rblur (there is a nonzero chance this makes the value for RAJ unusable, i.e. RAJ = 6:2:89.9, Rblur=+1.1 --> RAJ=6:2:91.0. just rerun the script)
    raj = (str(h)+':'+str(m)+':'+str(s+rblur),0.01)
    
    if args.dblur != None:
        dblur = args.dblur
    else:
        dblur = args.dblur_coeff*r.standard_normal()
    #blur DECJ by the amount dblur (see note above about possible unusbale values)
    decj = (str(d)+':'+str(arcm)+':'+str(arcs+dblur),0.01)
    
    #the length of the observation
    Tobs = ((ntoa2*density)*24*60*60)
    if args.f0blur != None:
        f0blur = args.f0blur
    else:
        #f0blur = (~0.1)/length_obs(in s)
        f0blur = r.uniform(args.f0blur_range[0], args.f0blur_range[1])/((ntoa2*density)*24*60*60)

    #blur F0 by amount f0blur
    f0 = (f0[0]+f0blur, 0.000001)
    
    #blur F1 if given a blurring factor, otherwise set F1 to zero, as is the case in most starting parfiles
    if args.f1blur != None:
        f1 = (f1[0]+args.f1blur, 0.0)
    else:
        f1 = (0.0, 0.0)
        
    #set positional parameters
    pepoch = args.PEPOCH
    tzrmjd = args.TZRMJD
    tzrfrq = args.TZRFRQ
    tzrsite = args.TZRSITE
    
    #write the parfile
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
    #end write parfile
    

def main(argv=None):
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument("--parfile", help="par file to read model from")
    parser.add_argument("--timfile", help="tim file to read toas from")
    parser.add_argument(
        "--iter",
        help="number of pulsar systems to produce",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--name",
        help="name for the pulsar, output files will be of format <name>.par, etc.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--F0_value", help="value of F0 (Hz)",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--F0_error", help="error of F0 (Hz)",
        type=float, 
        default=0.0000000001,
    )
    parser.add_argument(
        "--f0blur", help="how much to skew the known F0 value by (Hz)",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--f0blur_range", help="range of uniform random phases to skew F0 by (phase)",
        type=str, 
        default='0.05, 0.25',
    )
    parser.add_argument(
        "--RAJ_value", help="value of RAJ (degrees)",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--RAJ_error", help="error of RAJ (degrees)",
        type=float, 
        default=0.0000000001,
    )
    parser.add_argument(
        "--rblur", help="how much to skew the known value of RAJ by (degrees)",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--rblur_coeff", help="coefficient in front of Gaussian distribution to randomly skew RAJ",
        type=float, 
        default=1.3,
    )
    parser.add_argument(
        "--DECJ_value", help="value of DECJ (degrees)",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--DECJ_error", help="error of DECJ (degrees)",
        type=float, 
        default=0.0000000001,
    )
    parser.add_argument(
        "--dblur", help="how much to skew the known value of DECJ by (degrees)",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--dblur_coeff", help="coefficient in front of Gaussian distribution to randomly skew DECJ",
        type=float, 
        default=5.,
    )
    parser.add_argument(
        "--F1_value", help="value of F1 (1/s^2)",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--F1_error", help="error of F1 (1/s^2)",
        type=float, 
        default=0.0000000001,
    )
    parser.add_argument(
        "--f1blur", help="how much to skew the known value of F1 by ()",
        type=float, 
        default=None,
    )
    parser.add_argument(
        "--PEPOCH", help="period epoch for pulsar (MJD)",
        type=float, 
        default=56000,
    )
    parser.add_argument(
        "--TZRFRQ", help="Frequency (Hz) of observation",
        type=float, 
        default=1400,
    )
    parser.add_argument(
        "--TZRMJD", help="Observation start time (MJD)",
        type=float, 
        default=56000,
    )
    parser.add_argument(
        "--TZRSITE", help="observation site code",
        type=str, 
        default='GBT',
    )
    parser.add_argument(
        "--density_range", help="range of toa densities to choose from (days)",
        type=str, 
        default='0.004, 0.02',
    )
    parser.add_argument(
        "--span", help="range of time spans to choose from (days)",
        type=str, 
        default='200,700',
    )
    #parse comma-seperated pairs
    args = parser.parse_args(argv)
    args.span = [float(i) for i in args.span.split(',')]
    args.f0blur_range = [float(i) for i in args.f0blur_range.split(',')]
    args.density_range = [float(i) for i in args.density_range.split(',')]
    

    #write 3 files
    #fake toas - unevenly distributed, randomized errors
    #fake perfect parfile - solution to the fake system
    #fake starting parfile - starting parfile smeared/dispersed by a random amount
    #save the files in fake_data folder
    
    print(os.listdir('./fake_data/'))
    #determine highest number system from files in fake_data
    try:
        maxnum = max([int(filename[:-4][5:]) for filename in os.listdir('./fake_data/')])
    except ValueError:
        #TODO: this error is also raised when max breaks, ie, not all the files are of the format ____#___ whatever
        maxnum = 0
        print('no files in the directory')


    iter = args.iter
    for num in range(maxnum+1, maxnum+1+iter):
        if args.name == None:
            sol_name = 'fake_'+str(num)+'.sol'
            par_name = 'fake_'+str(num)+'.par'
            tim_name = 'fake_'+str(num)+'.tim'
        else:
            sol_name = args.name+'.sol'
            par_name = args.name+'.par'
            tim_name = args.name+'.tim'
            
        #write solfile
        f0_save, h, m, s, d, arcm, arcs, f0, f1, dm = write_solfile(args, sol_name)
        
        #wrie timfile
        ntoa2, density = write_timfile(args, f0_save, tim_name, sol_name)

        #write parfile
        write_parfile(args, par_name, h, m, s, d, arcm, arcs, f0, f1, dm, ntoa2, density)

        
if __name__ == '__main__':
    main()


