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

from run_2comp_cluster_differenslopes import *

segregation = [1.]
shapes = [1.]
propermotions = [0.1]
slopes = [0.7, 1.0]

if __name__ == '__main__':

    for p in list(itertools.product(*[segregation, propermotions, shapes, slopes])):

        simproperties = generate_simproperties(SegregationParameter=p[0],
                                               propermotionmag=p[1],
                                               flattening=p[2],
                                               slope=p[3],
                                               Nparticles=400000,
                                               output_file=False)

        times = nbody_tools.get_times(simproperties['simFileGRAVOUTPUT'])

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
