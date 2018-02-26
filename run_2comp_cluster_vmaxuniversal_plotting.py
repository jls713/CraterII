import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('external/')
import m2m_tools as m2m
import nbody_tools
from subprocess import call
import json
from sim_setup import *
from orbit import *
import itertools
import subprocess
import tempfile
import shutil
import os

from run_2comp_cluster_vmaxuniversal import *


segregation = [2., 5.]  # I think 10 is too ambitious
segregation = [2.]
shapes = [1., 0.3]  # , 0.3] ## Just spherical for the moment
# shapes = [0.3]
propermotions = [0.05, 0.1, 0.17, 0.25]  # , [0., 0.], [0., 0.17]]
# propermotions = [0.05,0.1, 0.17, 0.25]
segregation = [0.5]
shapes = [1.]
# propermotions = [0.05, 0.1,
propermotions = [0.1, 0.25]  # 0.17, 0.25]

segregation = [0.5]  # [1., 2.]
shapes = [0.3]
propermotions = [0.05, 0.1, 0.17, 0.25]

# segregation = [2.]
# shapes = [1.]
# propermotions = [0.05]

if __name__ == '__main__':

    for p in list(itertools.product(*[segregation, propermotions, shapes])):

        simproperties = generate_simproperties(SegregationParameter=p[0],
                                               propermotionmag=p[1],
                                               flattening=p[2],
                                               Nparticles=400000,
                                               output_file=False,
                                               old_version=False)

        times = nbody_tools.get_times(simproperties['simFileGRAVOUTPUT'])

        # data = grab_snapshot(simproperties, times[times>5.][0])
        # exit()

        dataset = all_evolution(simproperties)
        dataset.to_csv(simproperties['sim_folder'] +
                       simproperties['sim_name'] + '_properties.csv')
        continue
        for t in times:
            R = shape_profile_dehnen(simproperties, t)
            df = pd.DataFrame(np.vstack((R[:5])).T,
                              columns=['radii_st', 'ba_st',
                                       'ca_st', 'align_st', 'dens_st'])
            df.to_csv(simproperties['sim_folder'] + 'flattening_profiles/' +
                      simproperties['sim_name'] +
                      '_flattening_profile_stars_%0.3f.csv' % t)
            df = pd.DataFrame(np.vstack((R[8:13])).T,
                              columns=['radii_dm', 'ba_dm',
                                       'ca_dm', 'align_dm', 'dens_dm'])
            df.to_csv(simproperties['sim_folder'] + 'flattening_profiles/' +
                      simproperties['sim_name'] +
                      '_flattening_profile_dm_%0.3f.csv' % t)

        # rot = [[84., 208., 0.]]
        # compare_1d_profiles_2(simproperties,
        #                       simproperties['sim_name'] + '1dprof.pdf',
        #                       times=[0., times[-1]])
        # compare_1d_sigmar_profiles_2(simproperties,
        #                              simproperties['sim_name'] +
        #                              '1dsrprof.pdf',
        #                              times=[0., times[-1]], sigmayaxis=100.)

        # shape_evolution(simproperties,
        #                 simproperties['sim_name'] + 'shape.pdf',
        #                 max_time=times[-1])
        # bound_mass_evolution(simproperties,
        #                      simproperties['sim_name'] + 'mass.pdf',
        #                      max_time=times[-1])

        # generate_movie(simproperties, simproperties['sim_name'], rot=rot[0],
        #                select_range='all')
        # generate_movie(simproperties, simproperties['sim_name'] + '_dm',
        #                rot=rot[0],
        #                select_range='dm')
        # generate_movie(simproperties, simproperties['sim_name'] + '_st',
        #                rot=rot[0],
        #                select_range='stars')

    # simproperties['simFileGRAVOUTPUT']=simproperties['simFile']+'_nopot.snp'
    # shape_evolution(simproperties,
    #                     simproperties['sim_name']+'_nopot_shape.pdf')
    # compare_2d_profiles_2(simproperties,
    #                     simproperties['sim_name']+'2dprof.pdf',
    #                     times=[0.,simproperties['IntTime']])
