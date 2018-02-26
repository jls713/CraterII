from run_2comp_cluster_vmaxuniversal import *

# simproperties = generate_simproperties(SegregationParameter=5.,
#                                        propermotionmag=0.1,
#                                        flattening=1.,
#                                        without_potential=True,
#                                        Nparticles=400000)

# generate_plum_in_nfw(simproperties, nmax=20, int_step_frac=50.)


simproperties = generate_simproperties(SegregationParameter=2.,
                                       propermotionmag=0.1,
                                       flattening=1.,
                                       without_potential=True,
                                       Nparticles=400000)

generate_plum_in_nfw(simproperties, nmax=20, int_step_frac=50.)

# times = nbody_tools.get_times(simproperties['simFileGRAVOUTPUT'])

# compare_1d_profiles_2(simproperties,
#                       simproperties['sim_name'] + '1dprof.pdf',
#                       times=[0., times[-1]],
#                       annotation=r'$N_\mathrm{DM}=N_\star=4\times10^5$',
#                       vline=simproperties['rtidal'],
#                       vline_annotation=r'$r_t$')

# compare_1d_sigmar_profiles_2(simproperties,
#                              simproperties['sim_name'] +
#                              '1dsrprof.pdf',
#                              times=[0., times[-1]], sigmayaxis=100.)
