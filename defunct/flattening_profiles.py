from run_2comp_cluster_vmaxuniversal import *
s,pmmag,ca = 2.,0.05,0.3
simprops = generate_simproperties(SegregationParameter=s, propermotionmag=pmmag, flattening=ca,
				  Nparticles=4e5,output_file=False)
flat_props = shape_profile_dehnen(simprops,0.)
print flat_props
