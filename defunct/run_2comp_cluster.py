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


def fit_veldisp(target_vel_disp, simproperties):
    def veldisp(rs):
        simproperties['rs'] = rs
        # # As Walter uses abc=1
        # simproperties['rs'] *= np.power(simproperties['ca'], 1. / 3.)
        simproperties['Rscale'] = simproperties['rs'] * \
            simproperties['SegregationParameter']

        MDM_rs = nbody_tools.M_NFW_rs_c(simproperties['rs'],
                                        simproperties['Rscale'],
                                        simproperties['c'])
        return np.sqrt(MDM_rs * G / simproperties['rs'] / 2.5) - target_vel_disp
    return brentq(veldisp, 0.001, 1000.)


def generate_simproperties(SegregationParameter=2.,
                           propermotion=caldwell_propermotion,
                           flattening=1.,
                           fixed_veldisp=None):
    # SegregationParameter = 2.  # Ratio of r_dm/r_star
    # If fixed veldisp, choose r_s that recovers veldisp
    # Otherwise use r_s=1 -- a bit like Crater II
    # Simulation properties
    simproperties = {}

    # Names
    simproperties['sim_folder'] = '/data/jls/m2m/CraterII/fixed_veldisp_10_c20/'
    simproperties['sim_name'] = '2comp_cuspplum_nfw_SP_%i_PM_%i_%i_ca_%02d' % (
        SegregationParameter, propermotion[0] * 100.,
        propermotion[1] * 100., flattening * 10.)
    simproperties['simFile'] = simproperties['sim_folder'] + \
        simproperties['sim_name']
    simproperties['simFileDM'] = simproperties['sim_folder'] + \
        simproperties['sim_name'] + '_dm'
    simproperties['simFileS'] = simproperties['sim_folder'] + \
        simproperties['sim_name'] + '_stars'
    simproperties['simFileDMGRAVINPUT'] = simproperties['simFileDM'] + \
        '_init.ini'
    simproperties['simFileSGRAVINPUT'] = simproperties['simFileS'] + \
        '_init.ini'
    simproperties['simFileGRAVINPUT_scaled'] = simproperties['simFile'] + \
        '_init_sc.ini'
    simproperties['simFileGRAVOUTPUT'] = simproperties['simFile'] + '.snp'
    simproperties['simFileGRAVLOG'] = simproperties['simFile'] + '.log'

    # Values
    # 1. Properties of halo
    simproperties['NDM'] = 100000
    simproperties['NStars'] = 100000
    simproperties['MtoL'] = 30
    simproperties['ba'] = 1.
    simproperties['ca'] = flattening
    simproperties['SegregationParameter'] = SegregationParameter
    simproperties['c'] = 40.

    if fixed_veldisp is not None:
        simproperties['rs'] = fit_veldisp(fixed_veldisp, simproperties)
    else:
        simproperties['rs'] = 1.  # Size of Crater II ~ 1 kpc
        # As Walter uses abc=1
        simproperties['rs'] *= np.power(flattening, 1. / 3.)
    simproperties['Rscale'] = simproperties['rs'] * SegregationParameter
    print 'Scale radius =', simproperties['rs']

    # in units solar mass, kpc/Gyr kpc
    simproperties['G'] = G * nbody_tools.kms2kpcGyr**2

    # 2. Orbital properties
    Eq = np.array([crater2_eq()[0], crater2_eq()[1], distance, vlos,
                   propermotion[0], propermotion[1]])
    R, T = integrate_orbits_backwards(Eq)
    XV = R[-1]
    XV[-3:] *= -1.
    simproperties['rtidal'] = tidal_radius(
        np.sqrt(np.sum(XV[:3]**2)),
        simproperties['Rscale'], c=simproperties['c'])
    print 'Tidal radius = ', simproperties['rtidal']

    simproperties['Mscale'] = nbody_tools.M_NFW_rs_c(
        simproperties['rtidal'],
        simproperties['Rscale'],
        simproperties['c'])
    simproperties['tunit'] = 2. * np.pi / \
        (np.sqrt(G *
                 simproperties['Mscale'] / simproperties['Rscale']**3))
    # Mass within rs_S = MtoL*2
    MDM_rs = nbody_tools.M_NFW_rs_c(simproperties['rs'],
                                    simproperties['Rscale'],
                                    simproperties['c'])
    simproperties['Ms'] = 2. * MDM_rs / simproperties['MtoL']
    print 'Velocity dispersion = ', \
        np.sqrt(MDM_rs * G / simproperties['rs'] / 2.5)

    simproperties['XV'] = list(XV)
    simproperties['IntTime'] = T[-1]

    simproperties['Omega'] = 0.
    simproperties['tfac'] = 0.2
    simproperties['pot'] = 'GalPot'
    simproperties['potpars'] = None
    simproperties['potfile'] = '/home/jls/work/code/Torus/pot/PJM16_best.Tpot'
    simproperties['exptype'] = 1.  # Lilley expansion # NFW expansion

    simlog = json.dumps(simproperties)
    with open(simproperties['sim_folder'] + simproperties['sim_name'] +
              '.simlog',
              'w') as outfile:
        json.dump(simlog, outfile)

    return simproperties


def generate_images_glnemo(simproperties, output, rot=[0., 0., 0.], select_range='all'):
    zoom = -400.
    if select_range == 'all':
        select = '%i:%i,0:0,0:0,0:0,%i:%i' % (simproperties['NDM'],
                                              simproperties['NDM'] +
                                              simproperties['NStars'],
                                              0,
                                              simproperties['NDM'] - 1)
    elif select_range == 'dm':
        select = '0:0,0:0,0:0,0:0,%i:%i' % (0,
                                            simproperties['NDM'] - 1)
    elif select_range == 'stars':
        select = '%i:%i' % (simproperties['NDM'],
                            simproperties['NDM'] +
                            simproperties['NStars'])
    else:
        print 'select_range must be all, dm or stars.'
    command = "/home/jls/bin/glnemo2 in=%s zoom=%i select=%s times=all bestzoom=f screenshot=%s grid=f xrot=%0.2f yrot=%0.2f zrot=%0.2f com=f texture_a=0.2 psize=0. play=t shot_ext=png" % (
        simproperties['simFileGRAVOUTPUT'], zoom,
        select,
        output,
        rot[0], rot[1], rot[2])
    subprocess.call(command, shell=True)


def generate_movie(simproperties, output_name, select_range='all', rot=[0., 0., 0.]):
    cwd = os.getcwd()
    temp_path = tempfile.mkdtemp(dir=cwd)
    os.chdir(temp_path)
    generate_images_glnemo(simproperties, 'tmp',
                           select_range=select_range, rot=rot)
    subprocess.call('ffmpeg -r 1.5 -i tmp.%05d.png ../' +
                    output_name + '.gif', shell=True)
    os.chdir('../')
    shutil.rmtree(temp_path)


segregation = [2., 5.]
shapes = [1., 0.3]
propermotions = [caldwell_propermotion, [0., 0.], [0., 0.17]]
rot = [[84., 208., 0.]]
veldisp = 10.

if __name__ == '__main__':

    for p in list(itertools.product(*[segregation, propermotions, shapes])):
        print p
        simproperties = generate_simproperties(SegregationParameter=p[0],
                                               propermotion=p[1],
                                               flattening=p[2],
                                               fixed_veldisp=veldisp)

        generate_plum_in_nfw(simproperties, nmax=20, int_step_frac=50.)

        times = nbody_tools.get_times(simproperties['simFileGRAVOUTPUT'])

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

        dataset = all_evolution(simproperties)
        dataset.to_csv(simproperties['sim_name'] + '_properties.csv')

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
