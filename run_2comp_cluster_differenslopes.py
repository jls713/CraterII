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


def stellar_mass_from_mass_to_light_relation(Mdm):
    ''' Dark matter mass inside r_s '''
    return np.power(2. * Mdm / (200. * np.power(1e5, 0.4)), 1. / 0.6)


def mass_to_light(Mdm, Mst):
    ''' Dark matter mass inside r_s '''
    return 2. * Mdm / Mst


def generate_simproperties(SegregationParameter=2.,
                           slope=1.,
                           propermotionmag=caldwell_pmmag,
                           flattening=1.,
                           without_potential=False,
                           Nparticles=100000,
                           output_file=True):
    # SegregationParameter = 2.  # Ratio of r_dm/r_star
    # Simulation properties
    simproperties = {}

    # Names
    simproperties['sim_folder'] = \
        '/data/gaia-eso/jls/m2m/CraterII/vmax_universal/'
    simproperties['sim_name'] = '2comp_SP_%02d_slope_%02d_PM_%i_ca_%02d' % \
        (SegregationParameter * 10., slope * 10.,
         propermotionmag * 100., flattening * 10.)
    if without_potential:
        simproperties['sim_name'] += '_nogalpot'
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
    simproperties['NDM'] = Nparticles
    simproperties['NStars'] = Nparticles
    simproperties['Rscale'] = 1.45  # Produces v_max=20 km/s
    # simproperties['MtoL'] = 30
    simproperties['ba'] = 1.
    simproperties['ca'] = flattening
    simproperties['SegregationParameter'] = SegregationParameter
    simproperties['slope'] = slope

    simproperties['rs'] = simproperties['Rscale'] / SegregationParameter
    print 'Scale radius =', simproperties['rs']
    simproperties['epsilon'] = nbody_tools.optimal_softening(
        simproperties['NDM'], r_m2=1.) * simproperties['Rscale']
    print 'Softening =', simproperties['epsilon']
    # in units solar mass, kpc/Gyr kpc
    simproperties['G'] = G * nbody_tools.kms2kpcGyr**2

    propermotion = np.array([propermotionmag, 0.]) + crater2_solarreflex()

    # 2. Orbital properties
    Eq = np.array([crater2_eq()[0], crater2_eq()[1], distance, vlos,
                   propermotion[0], propermotion[1]])
    R, T = integrate_orbits_backwards(Eq)
    XV = R[-1]
    XV[-3:] *= -1.
    print XV

    MDM_rs = nbody_tools.truncated_double_power_mass(
        1.305 * simproperties['rs'], simproperties['Rscale'],
        1.,
        [1., 3., simproperties['slope']],
        0.)
    MDM_rs_NFW = nbody_tools.M_NFW_rs_c(
        1.305 * simproperties['rs'], simproperties['Rscale'],
        c=20., Delta=101.1)
    rho0 = MDM_rs_NFW / MDM_rs

    simproperties['rtidal'] = tidal_radius_double_power(
        np.sqrt(np.sum(XV[:3]**2)),
        simproperties['Rscale'],
        rho0,
        [1., 3., simproperties['slope']])
    print 'Tidal radius = ', simproperties['rtidal']

    simproperties['Mscale'] = nbody_tools.truncated_double_power_mass(
        np.inf,
        simproperties['Rscale'],
        rho0,
        [1., 3., simproperties['slope']],
        simproperties['rtidal'])
    simproperties['tunit'] = 2. * np.pi / \
        (np.sqrt(G * simproperties['Mscale'] /
                 simproperties['Rscale']**3)) / nbody_tools.kms2kpcGyr
    print 'Time unit: ', simproperties['tunit']

    # Mass within 1.305 rs_S = MtoL*2
    print simproperties['NStars']

    MDM_rs = nbody_tools.truncated_double_power_mass(
        1.305 * simproperties['rs'], simproperties['Rscale'],
        rho0,
        [1., 3., simproperties['slope']],
        simproperties['rtidal'])

    simproperties['Ms'] = stellar_mass_from_mass_to_light_relation(MDM_rs)
    simproperties['MtoL'] = mass_to_light(MDM_rs, simproperties['Ms'])
    print 'Velocity dispersion = ', \
        np.sqrt(MDM_rs * G / simproperties['rs'] / 4.)
    print 'Mass-to-light', simproperties['MtoL']
    simproperties['XV'] = list(XV)
    simproperties['IntTime'] = T[-1] / nbody_tools.kms2kpcGyr

    fraction = simproperties['IntTime'] / simproperties['tunit']
    simproperties['tunit'] = simproperties['IntTime'] / int(fraction)
    print simproperties['tunit']

    simproperties['Omega'] = 0.
    simproperties['tfac'] = 0.2
    simproperties['pot'] = 'GalPot'
    simproperties['potpars'] = None
    simproperties['potfile'] = '/home/jls/work/code/Torus/pot/PJM16_best.Tpot'
    simproperties['exptype'] = 2.  # 0. # 2.  # Lilley expansion # NFW expansion

    if without_potential:
        simproperties['pot'] = None
        simproperties['potfile'] = None
        simproperties['XV'] = [0.] * 6

    if(output_file):
        simlog = json.dumps(simproperties)
        with open(simproperties['sim_folder'] + simproperties['sim_name'] +
                  '.simlog',
                  'w') as outfile:
            json.dump(simlog, outfile)

    return simproperties


def run_progenitor_orbit(simproperties):
    XV = np.copy(simproperties['XV'])
    for i in range(3, 6):
        XV[i] *= nbody_tools.kms2kpcGyr
    nbody_tools.run_orbit(XV, simproperties['IntTime'],
                          simproperties['pot'],
                          simproperties['simFile'] + '_orbit',
                          potfile=simproperties['potfile'])


segregation = [1.]
shapes = [1.]  # , 0.3] ## Just spherical for the moment
propermotions = [0.1]
slopes = [0.1]#, 0.5, 0.7, 1.]

if __name__ == '__main__':

    for p in list(itertools.product(*[segregation, propermotions,
                                      shapes, slopes])):
        print p
        simproperties = generate_simproperties(SegregationParameter=p[0],
                                               slope=p[3],
                                               propermotionmag=p[1],
                                               flattening=p[2],
                                               Nparticles=400000)

        generate_plum_in_nfw(simproperties, nmax=30,
                             stellar_inner_cusp=p[3],
                             dark_inner_cusp=p[3],
                             beta_stars=-0.2,
                             int_step_frac=50.)

        # times = nbody_tools.get_times(simproperties['simFileGRAVOUTPUT'])

        # dataset = all_evolution(simproperties)
        # dataset.to_csv(simproperties['sim_folder'] +
        #                simproperties['sim_name'] + '_properties.csv')

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
