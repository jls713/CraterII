import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import seaborn
import linecache
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import MaxNLocator
from subprocess import call
from os import popen
from StringIO import StringIO
from itertools import izip
from numpy.linalg import eigvals
import pynbody
import threading
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# Run nemo/m2m commands in C shell
# =============================================================================

data_folder='/data/jls/m2m/'
log_folder='/data/jls/m2m/log_files/'
actions_folder='/home/jls/work/code/genfunc/new_struct/'

def run_csh_command(command,use_v2=False):
	cmd_str = "/bin/csh -c 'source /home/jls/.cshrc && source $NEMO/nemo.rc && setenv FALCON $NEMO/usr/dehnen/falcON.P && source $FALCON/falcON_restart && "+command+"'"
	if(use_v2==True):
		cmd_str = "/bin/csh -c 'source /home/jls/.cshrc && source $NEMO/nemo.rc && setenv FALCON $NEMO/usr/dehnen/falcON2 && source $FALCON/falcON_restart && "+command+"'"
	print '\n'+cmd_str+'\n'
	proc = call([cmd_str],shell=True) #, stdout=open(os.devnull, 'wb'))

def make_beta_string(beta0=0,betainf=0.75,r_s=1,eta=4./9.,nradial_bins=100,c_over_a=0.7,b_over_a=0.5):
	'''
		Returns the relevant string for anisotropic models
		inner anisotropy = beta0
		outer anisotropy = beta1
		scale radius = r_s
		eta = beta\propto r^eta
		c/a = c_over_a
		b/a = b_over_a
		number of radial bins = nradial_bins
	'''
	return str(beta0)+","+str(betainf)+","+str(r_s)+","+str(eta)+","+str(nradial_bins)+","+str(c_over_a)+","+str(b_over_a)


def prepareM2M(output_file, nbody=1000000, nsample=10, c_over_a=0.5, b_over_a=0.7,scale=1,nmax=20,lmax=12,inner=7./9.,outer=31./9.,rs=1.,eta=4./9.,r_t=10.,b=0.,lopside=1,Omegap=0.,potfile=None,tracer=False,M=1.,alpha=None,seed=None,twonu=2.,r_a=0.):
	'''
		Prepare a snapshot file for M2M code.
		Sample nbody particles nsample times to generate potential expansion
		plus errors
		Triaxial profile with b/a=b_over_a and c/a=c_over_a such that
			q^2=x^2/a^2+y^2/b^2+z^2/c^2 and abc=1 and scale radius rs
		lopside is the factor to compress the negative x direction by to
		produce a lopsided model
		Potential has scale radius scale and nmax radial expansion coeffs and lmax spherical harmonic coeffs
		Density profile:
			 \rho(r)=(q/rs)^{-inner} [(q/rs)^\eta+1]^-(outer-inner)/\eta
		r_t is a truncation radius multiplying density profile by sech(q/r_t)
		b is the anisotropy beta = 1-\sigma_\theta^2/\sigma_r^2
		if r_a!=0 beta transitions from b at centre to 1 at inf over scale r_a
		Omegap is the pattern speed
		potfile is an external PotExp file
		M is total mass
		twonu -- 2*nu (nu=0 NFW, nu=1/2 Lilley, nu=1 Zhao)
	'''
	# if os.path.isfile(output_file+'.ini'):
	# 	reply = raw_input(output_file+".ini exists. Delete file "+output_file+"? [y/[n]] ")
 #        if reply=='y':
 #            os.system('rm '+output_file+'.ini '+output_file+'.Sn '+output_file+'.An ')
 #        else:
 #            print "Aborting..."
 #            return 0
	csh_command="$FALCON/bin/./prepareM2M  out="+output_file+" nbody="+str(nbody)+" nsample="+str(nsample)+" zx="+str(c_over_a)+" yx="+str(b_over_a)+" scale="+str(scale)+" nmax="+str(nmax)+" lmax="+str(lmax)+" inner="+str(inner)+" outer="+str(outer)+" r_s="+str(rs)+" eta="+str(eta)+" r_t="+str(r_t)+" b="+str(b)+" lopside="+str(lopside)+" Omega="+str(Omegap)+" M="+str(M)+" twonu="+str(twonu)+" r_a="+str(r_a)
	if(alpha!=None):
		csh_command+=" alpha="+str(alpha)
	if(potfile!=None):
		csh_command+=" potname=NormPotExp potfile="+potfile+".An"
	if(tracer==True):
		csh_command+=" M=1e-6"
	csh_command+=" verbose=t;"
	if(seed!=None):
		csh_command+=" seed="+str(seed)
	run_csh_command(csh_command)

class disc_params:
	def __init__(self,ndisk=100000,hdisk=0.14,rsolar=3.07,Qtoomre=1.4):
		self.ndisk=ndisk
		self.hdisk=hdisk
		self.rsolar=rsolar
		self.Qtoomre=Qtoomre
	def params_list(self):
		return "ndisk="+str(self.ndisk)+" hdisk="+str(self.hdisk)+" rsolar="+str(self.rsolar)+" Qtoomre="+str(self.Qtoomre)

class halo_params:
	def __init__(self,mhalo,rhalo,nhalo=100000,qhalo=0.8,type='LH'):
		self.nhalo=nhalo
		self.mhalo=mhalo
		self.rhalo=rhalo
		self.qhalo=qhalo
		self.type=type
	def params_list(self):
		return "nhalo="+str(self.nhalo)+" mhalo="+str(self.mhalo)+" rhalo="+str(self.rhalo)+" qhalo="+str(self.qhalo)+" type="+self.type

def build_magalie_galaxy(outputfile,disc_params,halo_params):
	csh_command="$NEMO/bin/magalie out="+outputfile+" "+disc_params.params_list()+" "+halo_params.params_list()
	run_csh_command(csh_command)

class mkgal_disc_params:
	def __init__(self,ndisk=200000,zdisk=0.1,Qtoomre=1.2,RSig=0.):
		self.ndisk=ndisk
		self.zdisk=zdisk
		self.Qtoomre=Qtoomre
		self.RSig=RSig
	def params_list(self):
		return "Nd="+str(self.ndisk)+" Zd="+str(self.zdisk)+" Q="+str(self.Qtoomre)+" Rsig="+str(self.RSig)

class mkgal_bulge_params:
	def __init__(self,mass=0.2):
		self.mass=mass
	def params_list(self):
		return "Mb="+str(self.mass)

class mkgal_halo_params:
	def __init__(self,nhalo=1200000,mass=24.,inner_slope=1,outer_slope=3,scaleradius=6.,truncradius=60.):
		self.nhalo=nhalo
		self.mass=mass
		self.ih=inner_slope
		self.oh=outer_slope
		self.rh=scaleradius
		self.rt=truncradius
	def params_list(self):
		return "Nh="+str(self.nhalo)+" Mh="+str(self.mass)+" innerh="+str(self.ih)+" outerh="+str(self.oh)+" Rh="+str(self.rh)+" Rth="+str(self.rt)

def build_mkgalaxy_galaxy(outputfile,disc_params,bulge_params,halo_params):
	csh_command="$NEMO/bin/mkgalaxy name="+outputfile+" "+disc_params.params_list()+" "+bulge_params.params_list()+" "+halo_params.params_list()
	run_csh_command(csh_command)

def runM2M(input_file,output_file,log_file,tstop=200,dT=1,mmmax=4,mu=100.,epsilon=0.5,eta=0.5, m2m_constraints=[["RhoNLM","",""]], parallel_threads=None,debug=None,live_plot=False,other=None,Omegap=None,withgdb=False,accfile=None,dT_rs=10,tfac=0.0025,sfrac=0.05):
	'''
		Run M2M code with initial conditions in input_file and output sim to output_file every dT until final time tstop
		Resample if m_max/m_min>mmmax (= 4 for default)
		mu multiples entropy in cost function so makes weights smoother
		epsilon controls how fast weights evolve
		eta controls how the smoothing (eta=0 no smoothing)
		if parallel_threads number given it will run M2MP which is an mpi implementation
		if debug is not None it will run the code in debug mode (printing out statements to cout)
	'''
	# if os.path.isfile(output_file):
	# 	reply = raw_input(output_file+" exists. Delete file "+output_file+"? [y/[n]] ")
 #        if reply=='y':
 #            os.system('rm '+output_file)
 #        else:
 #            print "Aborting..."
 #            return 0
 	acc_file = accfile
 	if(accfile==None):
 		acc_file=input_file

	csh_command="$FALCON/bin/./M2M in="+input_file+".ini out="+output_file+\
	" give=mxvktdr tstop="+str(tstop)+" dT="+str(dT)+" mmmax="+str(mmmax)+\
	" logfile="+log_file+" mu="+str(mu)+" epsilon="+str(epsilon)+\
	" eta="+str(eta)+" dT_rs="+str(dT_rs)+" tfac="+str(tfac)+" sfrac="+str(sfrac)+" accname=NormPotExp accfile="+acc_file+".An m2mname="

	if(parallel_threads):
		mpi_com='mpirun -np '+str(parallel_threads)+' '
		if(withgdb):
			mpi_com=mpi_com+'xterm -e gdb --args '
		csh_command=mpi_com+csh_command[:17]+'P'+csh_command[17:]

	for con in range(len(m2m_constraints)):
		if(m2m_constraints[con][0]=="RhoNLM" and m2m_constraints[con][1]==""):
			m2m_constraints[con][1]=input_file
		if(con>0):
			csh_command+=","
		csh_command+=m2m_constraints[con][0]
	csh_command+=" m2mfile=\""
	for con in range(len(m2m_constraints)):
		if(con>0):
			csh_command+=";"
		csh_command+=m2m_constraints[con][1]
	csh_command+="\" m2mpars=\""
	for con in range(len(m2m_constraints)):
		if(con>0):
			csh_command+=";"
		csh_command+=m2m_constraints[con][2]
	csh_command+="\""
	if(debug):
		csh_command+=" debug="+str(debug)
	if(Omegap):
		csh_command+=" Omegap="+str(Omegap)
	if(other):
		csh_command+=" "+other
	csh_command+=";"
	if(live_plot):
		call(['touch',log_file])
		draw_thread=threading.Thread(target=live_plotter,args=[log_file,csh_command])
	if(withgdb and not parallel_threads):
		csh_command="gdb --args "+csh_command
	run_csh_command(csh_command)

def phasemixM2M(input_file,output_file,tstop=200,dT=1, parallel_threads=None):
	return runM2M(input_file,output_file,'',tstop=200,dT=1,mmmax=4,mu=100.,epsilon=0.,eta=0.5, m2m_constraints=[["RhoNLM","",""]], parallel_threads=None,debug=None,live_plot=False,other=None)

def extract_snapshot(input_file,output_file,time):
	'''
		Extract snapshot from input_file corresponding to time
	'''
	call(['rm',output_file])
	csh_command="s2s in="+input_file+" out="+output_file+" times="+str(time-0.01)+":"+str(time+0.01)+" time=0;"
	run_csh_command(csh_command)

def extract_ascii_snapshot(input_file,output_file,time):
	'''
		Extract snapshot at time as ascii from input_file
	'''
	call(['rm',output_file])
	csh_command="s2a in="+input_file+" out="+output_file+" times="+str(time-0.01)+":"+str(time+0.01)+";"
	run_csh_command(csh_command)

def write_nparray_to_snapshot(data,output_file):
	'''
		Write numpy array to file and then convert to snp
	'''
	if os.path.isfile(output_file):
		call(['rm',output_file])
	np.savetxt(output_file+'_tmp',data)
	N = len(data)
	csh_command="a2s in="+output_file+"_tmp read=mxv N="+str(N)+" out="+output_file+";"
	run_csh_command(csh_command)
	call(['rm',output_file+'_tmp'])

def extract_final_snapshot(input_file,output_file,outtype='snap'):
	'''
		Extract final snapshot from input_file
	'''
	# call(['rm',output_file])
	prog='s2s'
	if(outtype=='ascii'):
		prog='s2a'
	csh_command=prog+" in="+input_file+" out="+output_file+" times=last"
	if(prog=='s2s'):
		csh_command+=' time=0.;'
	else:
		csh_command+=';'
	run_csh_command(csh_command)

def make_movie_from_snapshot(input_file,output_folder,timerange=[0.,200.],mass_cut=1e-5):
	'''
		Make movie from a snapshot
	'''
	call(['rm','-r',output_folder])
	call(['mkdir',output_folder])
	time=timerange[0]
	f=plt.figure(figsize=(5,5))
	label=0
	while(time<timerange[1]):
		extract_ascii_snapshot(input_file,output_folder+'/tmp',time)
		data = np.genfromtxt(output_folder+'/tmp')
		if(len(data)==0):
			break
		data = data[data.T[0]<mass_cut]
		ax=plt.gca()
		ax.set_aspect('equal')
		ax.set_xlabel(r'$x/R_d$')
		ax.set_ylabel(r'$y/R_d$')
		ax.set_xlim(-5.,5.)
		ax.set_ylim(-5.,5.)
		ax.scatter(data.T[1][::10],data.T[2][::10],color='k',edgecolor='none',alpha=0.15,s=5)
		plt.text(0.97,0.97,r'$t=$'+str(time),horizontalalignment='right',verticalalignment='top',transform=ax.transAxes)
		plt.savefig(output_folder+'/'+str(label).zfill(4)+'.jpg',bbox_inches='tight',dpi=600)
		plt.cla()
		time+=1.
		label+=1

	call(['convert','-delay','20','-loop','0',output_folder+'/*.jpg',output_folder+'/movie.gif'])

def run_under_gravity(input_file, output_file,tstop,epsilon=0.001,kmin=7,kmax=3,logfile=None,fac=0.01,fph=0.04,dT=100.):
	'''
		Run N-body simulation for time with softening epsilon and max time-step = 2e-kmax
	'''
	Nlev=kmin-kmax+1
	# if(tstop<100.):
	# 	dT=tstop
	csh_command='gyrfalcON in='+input_file+" out="+output_file+" tstop="+str(tstop)+" eps="+str(epsilon)+" Nlev="+str(Nlev)+" kmax="+str(kmax)+" fac="+str(fac)+" fph="+str(fph)+" step="+str(dT)
	if(logfile):
		csh_command+=" logfile="+logfile
	csh_command+=" ;"
	run_csh_command(csh_command)

def process_density(input_file,output_file):
	'''
		Extract density from snapshot
	'''
	output_tmp = output_file+".tmp"
	csh_command='density in='+input_file+" out="+output_tmp+" give=mxvr debug=1;"
	run_csh_command(csh_command)
	csh_command='s2a in='+output_tmp+" out="+output_file
	run_csh_command(csh_command)
	call(['rm',output_tmp])


def find_actions_spherical(input_file,output_file,potential_file):
	'''
	Calculates actions for the data in input_file using the potential
	expansion defined in potential_file
	'''
	run_csh_command(actions_folder+"./falcON_aa.exe in="+input_file+" out="+output_file+" times=last accname=NormPotExp accfile="+potential_file)

# =============================================================================
# Read in various files to pandas arrays
# =============================================================================
def read_dehnen_log_file(infile):
	'''
		Reads in M2M log file to a pandas data array
		such that the data can be addressed using data['column_name']
	'''
	 # Remove leading hash and split
	names = linecache.getline(infile,9)[1:].split()
	data = 0
	with open(infile) as ffile:
		data = ffile.read()
	data_join = ''.join(data)
	while '  ' in data_join:
		data_join = data_join.replace('  ',' ')
	data = pd.read_csv(StringIO(data_join),names=names,sep=' ',skipinitialspace=True,comment='#')
	return data

def read_dehnen_beta_file(infile,n_radial_bins):
	'''
		Reads in M2M beta profile to a pandas data array
		such that the data can be addressed using data['column_name']
	'''
	# Load final n_radial_bins+2 data points and strip leading hash
	data_str = popen('tail -n '+str(n_radial_bins+2)+' '+infile).read()[1:]
	# Remove hash line and extract headers
	data_str_split = data_str.split('\n')
	names = data_str_split[0].split()
	data_str = '\n'.join(data_str_split[2:])
	data = pd.read_csv(StringIO(data_str),names=names,sep=r'\s+',skipinitialspace=True)
	return data

def read_dehnen_density_file(infile):
	'''
		Reads in M2M density file to a pandas data array
		such that the data can be addressed using data['column_name']
	'''
	 # Remove leading hash and split
	names = ['m','x','y','z','vx','vy','vz','rho']
	data = pd.read_csv(infile,names=names,sep=r'\s+',skiprows=range(13),skipinitialspace=True)
	return data

def read_action_file(infile,Omega=False):
	'''
		Reads in M2M action file to a pandas data array
		such that the data can be addressed using data['column_name']
	'''
	 # Remove leading hash and split
	names = ['m','x','y','z','vx','vy','vz','J_r','L','H']
	if(Omega):
		names = ['m','x','y','z','vx','vy','vz','J_r','L','theta_r','theta_p', 'theta_t','Omega_r','Omega_p','Omega_t','H']
	data = pd.read_csv(infile,names=names,sep=r' ')
	return data

# =============================================================================
# Do some plotting
# =============================================================================
def trim_top_label(ax,log=False):
	'''
		Remove top label from y-axis
	'''
	# nbins = len(ax.get_yticklabels())
	# ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
	# print ax.get_yticks()
	plt.setp(ax.get_yticklabels()[-1-log],visible=False)

def make_convergence_plot(input_file,output_file,text=None,
                          cost_function_labels=[r'$C_\rho$',r'$C_\beta$']):
	'''
		Accepts a log file output from Walter's M2M code
		Plots the cost function, change in entropy and
		the rms of the velocities of change
	'''

	data = read_dehnen_log_file(input_file)

	cost_index=0
	while('C_'+str(cost_index) in data.keys()):
		cost_index+=1

	assert(cost_index<=len(cost_function_labels))

	fig,ax = plt.subplots(2+cost_index,1,figsize=[6.,5.])
	plt.subplots_adjust(hspace=0)
	plot_index=0
	for plot_index in range(cost_index):
		ax[plot_index].plot(data['time'],data['C_'+str(plot_index)],'k')
		ax[plot_index].set_yscale('log')
		ax[plot_index].set_ylabel(cost_function_labels[plot_index])
		plt.setp(ax[plot_index].get_xticklabels(), visible=False)

		if(plot_index>0):
			trim_top_label(ax[plot_index],log=True)

	ax[cost_index].plot(data['time'],data['dS'],'k')
	ax[cost_index].set_ylabel(r'$-\Delta S$')
	plt.setp(ax[cost_index].get_xticklabels(), visible=False)
	trim_top_label(ax[cost_index])

	ax[cost_index+1].plot(data['time'],data['rms{D}'],'k')
	ax[cost_index+1].set_yscale('log')
	ax[cost_index+1].set_ylabel(r'${\rm rms}\{U_i\}$')
	ax[cost_index+1].set_xlabel(r'$\tau$')
	trim_top_label(ax[cost_index+1],log=True)

	if(text):
		plt.text(1.,1.02,text,horizontalalignment='right',verticalalignment='bottom',transform=ax[0].transAxes)

	plt.savefig(output_file,bbox_inches='tight')

def diagonalize_density_matrix(arr):
	mat = np.zeros((3,3))
	m = arr['m'].values
	x = arr['x'].values
	y = arr['y'].values
	z = arr['z'].values
	r = arr['rho'].values
	rad=0
	dens=0
	for mm,xx,yy,zz,rr in izip(m,x,y,z,r):
		x2=xx**2
		y2=yy**2
		z2=zz**2
		rad+=z2+y2+x2
		xy=xx*yy
		xz=xx*zz
		yz=yy*zz
		mat+=mm*np.array([[y2+z2,-xy,-xz],[-xy,x2+z2,-yz],[-xz,-yz,x2+y2]])
		dens+=rr
	ee=np.sqrt(np.abs(eigvals(mat)))
	return dens/len(arr.index),np.sqrt(rad/len(arr.index)),ee[1]/ee[0],ee[2]/ee[0]

def dehnen_density_profile(rr,params):
	'''
		params = (rs,inner,outer,eps,r_t)
	'''
	rs,inner,outer,eps,r_t=params
	r = rr/rs
	return np.power(r,-inner)*np.power(np.power(r,eps)+1,-(outer-inner)/eps)/np.cosh(rr/r_t)

def add_to_array(data):
	data['r']=np.sqrt(data['x']**2+data['y']**2+data['z']**2)
	data['R']=np.sqrt(data['x']**2+data['y']**2)
	data['phi']=np.arctan2(data['y'],data['x'])
	data['ct']=data['z']/data['r']
	data['t']=np.arccos(data['ct'])
	data['st']=np.sqrt(1-data['ct']*data['ct'])
	data['vr']= data['vx']*np.cos(data['phi'])*data['st']+\
				data['vy']*np.sin(data['phi'])*data['st']+\
				data['ct']*data['vz']
	data['vp']=-data['vx']*np.sin(data['phi'])+data['vy']*np.cos(data['phi'])
	data['vt']= data['vx']*np.cos(data['phi'])*data['ct']+\
				data['vy']*np.sin(data['phi'])*data['ct']-\
				data['st']*data['vz']

	data['msr']=data['m']*data['vr']*data['vr']
	data['msp']=data['m']*data['vp']*data['vp']
	data['mst']=data['m']*data['vt']*data['vt']
	data['mrp']=data['m']*data['vr']*data['vp']
	data['mpt']=data['m']*data['vp']*data['vt']
	data['mrt']=data['m']*data['vr']*data['vt']

	data['mr']=data['m']*data['vr']
	data['mp']=data['m']*data['vp']
	data['mt']=data['m']*data['vt']
	return data


def bin_pandas_array_by_key(data, key, nbins=100):
	ntot=len(data)
	nbins=nbins+2
	dr = np.sort(data[key].values)
	nperbin=ntot/nbins
	bins_r = np.zeros(nbins-1)
	index=0
	ccount=1
	for count in range(ntot):
		if(ccount>nperbin):
			bins_r[index]=(dr[count]+dr[count+1])*.5
			index+=1
			ccount=0
		ccount+=1

	groups = data.groupby(np.digitize(data[key],bins_r))
	bins_c = .5*(bins_r[1:]+bins_r[:-1])
	return groups,bins_c

def process_data_and_bin_in_r(data,nbins=100):
	'''
	Equal binning
	'''
	data2 = add_to_array(data)
	return bin_pandas_array_by_key(data,'r',nbins)

def process_data_and_bin_in_x_and_y(data,zmax=0.5,nbins=10,x0=0.,x1=1.):
	data = data[np.abs(data['z'])<zmax]
	data = data[(np.abs(data['y'])>x0)*(np.abs(data['y'])<x1)]
	data = data[(np.abs(data['x'])>x0)*(np.abs(data['x'])<x1)]
	data['vx']=data['vx']*np.sign(data['x'])
	data['vy']=data['vy']*np.sign(data['y'])
	data['x']=np.abs(data['x'])
	data['y']=np.abs(data['y'])
	data2 = add_to_array(data)
	bins_x = np.linspace(x0,x1,nbins)
	bins_y = np.linspace(x0,x1,nbins)
	groups = data2.groupby(np.digitize(data2['x'],bins_x)+nbins*(np.digitize(data2['y'],bins_y)-np.ones(len(data2['y']))))
	return groups


def process_data_and_bin_in_rphitheta(data,rr,nbins=10,theta_or_phi='theta',rlog=False):
	data['vx']=data['vx']*np.sign(data['x'])
	data['vy']=data['vy']*np.sign(data['y'])
	data['vz']=data['vz']*np.sign(data['z'])
	data['x']=np.abs(data['x'])
	data['y']=np.abs(data['y'])
	data['z']=np.abs(data['z'])
	data2 = add_to_array(data)
	bins_r = np.linspace(rr[0],rr[1],nbins+1)
	if(rlog==True):
		bins_r = np.logspace(np.log10(rr[0]),np.log10(rr[1]),nbins+1)
	bins_p = np.linspace(0.,np.pi/2.,nbins+1)
	bins_t = np.linspace(0.,np.pi/2.,nbins+1)
	if(theta_or_phi=='theta'):
		data2 = data2[(data2['t']>bins_t[-2])*(data2['t']<bins_t[-1])*(data2['phi']<bins_p[-1])*(data2['phi']>bins_p[0])*(data2['r']<bins_r[-1])*(data2['r']>bins_r[0])]
		groups = data2.groupby(np.digitize(data2['r'],bins_r,right=True)+(nbins+1)*(np.digitize(data2['phi'],bins_p)-np.ones(len(data2['r']))))
		print len(groups)
		return groups
	if(theta_or_phi=='phi'):
		data2 = data2[(np.arccos(data2['y']/data2['r'])>bins_t[-2])*(np.arccos(data2['y']/data2['r'])<bins_t[-1])*(data2['t']<bins_t[-1])*(data2['t']>bins_t[0])*(data2['r']<bins_r[-1])*(data2['r']>bins_r[0])]
		groups = data2.groupby(np.digitize(data2['r'],bins_r,right=True)+(nbins+1)*(np.digitize(data2['t'],bins_t)-np.ones(len(data2['r']))))
		print len(groups)
		return groups
	if(theta_or_phi=='phimax'):
		data2 = data2[(np.arccos(data2['x']/data2['r'])>bins_t[-2])*(np.arccos(data2['x']/data2['r'])<bins_t[-1])*(data2['t']<bins_t[-1])*(data2['t']>bins_t[0])*(data2['r']<bins_r[-1])*(data2['r']>bins_r[0])]
		groups = data2.groupby(np.digitize(data2['r'],bins_r,right=True)+(nbins+1)*(np.digitize(data2['t'],bins_t)-np.ones(len(data2['r']))))
		print len(groups)
		# print groups.mean().x,groups.mean().z,groups.count().x
		# plt.clf();
		# plt.plot(groups.mean().z,groups.count().z,'.')
		# plt.show()
		return groups

from scipy.stats import moment

def make_velocity_distribution_plots(input_file,output_file,rbin=[0.5,1.,2.],width=0.1,text=None):
	'''
	Plots density and anisotropy (if a file is provided) for target, output
	and after gravity run (if a file is provided) [can also provide a list
	of times to plot from after grav file, but default is last]
	'''

	plt.clf()
	f,a=plt.subplots(1,3,figsize=(10,3))
	extract_final_snapshot(input_file,input_file+".tmp")
	process_density(input_file+".tmp",input_file+".density")
	data = read_dehnen_density_file(input_file+".density")
	call(['rm',input_file+".density",input_file+'.tmp'])
	data2=add_to_array(data)
	for k,i in enumerate(rbin):
		data3=data2[(data2.r>i-width)&(data2.r<i+width)]
		a[k].hist(data3.vr.values,histtype='step',lw=3,normed=True,bins=50,weights=data.m.values)
		a[k].hist(data3.vp.values,histtype='step',lw=3,normed=True,bins=50,weights=data.m.values)
		a[k].annotate(str(moment(data3.vr.values,moment=4))+', '+str(moment(data3.vp.values,moment=4)),xy=(-1.,2.2))
	if(text):
		plt.text(1.,1.02,text,horizontalalignment='right',verticalalignment='bottom',transform=a[0].transAxes)
	plt.savefig(output_file,bbox_inches='tight')

def make_density_anisotropy_plots(input_file,output_file,triaxial_vals=None,anisotropy_file=None,n_radial_anisotropy=100,text=None,n_per_bin=5000,analytic_params=[1.,7./9.,31./9.,4./9.,10.],after_grav_file=None,after_grav_time=None,gamma=None):
	'''
	Plots density and anisotropy (if a file is provided) for target, output
	and after gravity run (if a file is provided) [can also provide a list
	of times to plot from after grav file, but default is last]
	'''

	extract_final_snapshot(input_file,input_file+".tmp")
	process_density(input_file+".tmp",input_file+".density")
	data = read_dehnen_density_file(input_file+".density")
	call(['rm',input_file+".density",input_file+'.tmp'])

	## Density calculation
	data.sort('rho',ascending=False,inplace=True)
	number_of_bins=len(data.index)/n_per_bin
	dens = np.zeros(number_of_bins)
	ra = np.zeros(number_of_bins)
	dens_analytic = np.zeros(number_of_bins)
	ba = np.zeros(number_of_bins)
	ca = np.zeros(number_of_bins)
	for i in range(number_of_bins):
		dens[i],ra[i],ba[i],ca[i] = diagonalize_density_matrix(data[n_per_bin*i:n_per_bin*(i+1)])
		dens_analytic[i]=dehnen_density_profile(ra[i],analytic_params)
	## Anchor
	fac = dens[number_of_bins/2]/dens_analytic[number_of_bins/2]

	anisotropy=0
	if(anisotropy_file):
		anisotropy=1
	triaxial=0
	if(triaxial_vals):
		triaxial=1
	number_of_plots=1+anisotropy+triaxial
	fig,ax=plt.subplots(number_of_plots,1,figsize=[4.,1.5*number_of_plots])
	plt.subplots_adjust(hspace=0)
	xrang = (0.1,100.)

	if(number_of_plots==1):
		ax=[ax]
	ax[0].plot(ra,dens,label=r'$t=0$')
	ax[0].plot(ra,dens_analytic*fac,ls='--',label='Analytic')
	ax[0].set_xlim(xrang)
	if(number_of_plots>1):
		plt.setp(ax[0].get_xticklabels(),visible=False)
	else:
		ax[0].set_xlabel(r'$r/r_s$')
	ax[0].set_ylabel(r'$\rho$')
	ax[0].set_xscale('log')
	ax[0].set_yscale('log')
	ax[0].legend(handlelength=2, scatterpoints=1, numpoints=1,frameon=False,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.0))

	## Plot axis ratios
	if(triaxial):
		ax[1].plot(ra,ba,'k',label=r'$b/a$')
		ax[1].plot(ra,ca,'k:',label=r'$c/a$')
		ax[1].legend(frameon=False,ncol=2)
		ax[1].set_xscale('log')
		ax[1].set_ylabel('Axis ratios')
		ax[1].set_xlim(xrang)
		trim_top_label(ax[1],log=False)
		plt.setp(ax[1].get_xticklabels(),visible=False)

	## Plot anisotropy
	if(anisotropy_file):
		data = read_dehnen_beta_file(anisotropy_file,n_radial_anisotropy)
		ax[-1].set_ylabel(r'$\beta$')
		ax[-1].set_xscale('log')
		ax[-1].set_xlabel(r'$r/r_s$')
		ax[-1].plot(data['r'],data['beta'])
		ax[-1].plot(data['r'],data['beta_mod'],ls='--')
		ax[-1].set_xlim(xrang)
		trim_top_label(ax[-1],log=True)
		if(gamma):
			ax[-1].axhline(gamma/2.,c='k',ls='--')

	if(text):
		plt.text(1.,1.02,text,horizontalalignment='right',verticalalignment='bottom',transform=ax[0].transAxes)

	timelabels=[r'$t=200$']
	times=[200.]
	if(after_grav_time):
		if not(isinstance(after_grav_time,list)):
			timelabels=[r'$t=$'+str(after_grav_time)]
			times = [after_grav_time]
		else:
			timelabels=[map(lambda i: r'$t=$'+str(i),after_grav_time)]
			times = after_grav_time
	if(after_grav_file):
		for T,L in zip(times,timelabels):
			if(after_grav_time):
				extract_snapshot(after_grav_file,after_grav_file+".tmp",T)
			else:
				extract_final_snapshot(after_grav_file,after_grav_file+".tmp")
			process_density(after_grav_file+".tmp",after_grav_file+".density")
			data = read_dehnen_density_file(after_grav_file+".density")
			call(['rm',after_grav_file+".density",after_grav_file+'.tmp'])

			## Density calculation
			data.sort('rho',ascending=False,inplace=True)
			number_of_bins=len(data.index)/n_per_bin
			dens = np.zeros(number_of_bins)
			ra = np.zeros(number_of_bins)
			dens_analytic = np.zeros(number_of_bins)
			ba = np.zeros(number_of_bins)
			ca = np.zeros(number_of_bins)
			for i in range(number_of_bins):
				dens[i],ra[i],ba[i],ca[i] = diagonalize_density_matrix(data[n_per_bin*i:n_per_bin*(i+1)])
			ax[0].plot(ra,dens,ls=':',label=L)
			groups, bins_c = process_data_and_bin_in_r(data)

			ax[1].plot(bins_c,1-.5*(groups.mean().mst[1:-1]+groups.mean().msp[1:-1])/groups.mean().msr[1:-1],ls=':')

	ax[0].legend(handlelength=2, scatterpoints=1, numpoints=1,frameon=False,ncol=2+(1 if after_grav_file else 0),loc='upper center', bbox_to_anchor=(0.5, 1.0))

	plt.savefig(output_file,bbox_inches='tight')

from matplotlib.patches import Ellipse

def velocity_ellipses_polar(x,y,dens,sigrr,sigrp,sigpp,N,outputfile,width1=1.,width2=10.,vlimits=None,labels=[r'$x/{\rm kpc}$',r'$y/{\rm kpc}$'],draw_scale=None,text=None,sig_range=None,w_cb=True):

    ## Compute errors
    dsigrr2=2.*sigrr.values**2/N
    dsigpp2=2.*sigpp.values**2/N
    dsigrp2=2*sigrr.values*sigpp.values/N
    tan2ij=2.*sigrp.values/(sigrr.values-sigpp.values)
    facij=np.fabs(.5*tan2ij/(1+tan2ij**2))
    daij=facij*np.sqrt(dsigrp2/sigrp.values**2+(dsigrr2+dsigpp2)/(sigrr.values-sigpp.values)**2)
    daij=np.rad2deg(daij)
    aij =np.rad2deg(np.arctan(tan2ij)*.5)
    f=plt.figure()
    X = np.linspace(width1,width2)
    Y = np.linspace(width1,width2)
    offxygrd = griddata((x,y), aij/daij, (X[None,:], Y[:,None]), method='cubic')
    CMAP = sns.blend_palette(sns.hls_palette(7),as_cmap=True)
    v = np.linspace(-5.,5.,6)
    ax1 = plt.gca()
    ax1.set_aspect('equal')
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_xlim(width1,width2)
    ax1.set_ylim(width1,width2)
    plt.contourf(X,Y,offxygrd,v,cmap=CMAP)
    plt.colorbar(label=r'$\alpha/\sigma_\alpha$')
    plt.savefig(outputfile[:-4]+'_errors.pdf',bbox_inches='tight')
    plt.clf()

    f,ax=plt.subplots(1,2,figsize=[5.,2.4],sharey=True)
    plt.subplots_adjust(wspace=0.)

    size = 0.06*(width2-width1)
    ax1 = ax[0]
    ax2 = ax[1]

    # Draw faint polar lines
    xi = np.linspace(width1,width2)
    for i in np.linspace(0.,np.pi/2.,15):
        ax1.plot(xi,np.tan(i)*xi,color="0.3",zorder=0)
        ax2.plot(xi,np.tan(i)*xi,color="0.3",zorder=0)

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    # x-y
    diagonals_xy = [eigsorted(np.array([[i,j],[j,k]])) for i,j,k in zip(sigrr,sigrp,sigpp)]
    print np.arctan2(*diagonals_xy[0][1][:,0][::-1]),sigrr.values[0],sigrp.values[0],sigpp.values[0]

    ells_xy = [Ellipse(xy=(xx,yy),
                    width= size,
                    height=size*np.sqrt(j[0][1]/j[0][0]),
	                angle=np.degrees(np.arctan2(*j[1][:,0][::-1])+np.arctan2(yy,xx)),
	                zorder=2)
                    for xx,yy,j in zip(x,y,diagonals_xy)]

    ax1.set_aspect('equal')
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_xlim(width1,width2)
    ax1.set_ylim(width1,width2)

    # Find max and min velocity dispersions
    maxi,mini = 0.,1e5
    if(sig_range):
    	mini,maxi=sig_range[0],sig_range[1]
    else:
	    for dd,d1 in zip(dens,diagonals_xy):
	        maxic1 = np.sqrt(np.max(d1[0])/dd)
	        if(maxic1>maxi):
	            maxi = maxic1
	        elif(maxic1<mini):
	            mini = maxic1


    # Colorbar
    CMAP = sns.blend_palette(sns.hls_palette(7),as_cmap=True)
    cNorm = colors.Normalize(vmin=mini,vmax=maxi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=CMAP)
    scalarMap._A=[]
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes("top", size="5%", pad=0.05)
    if(w_cb):
	    cax = f.add_axes([0.15,0.93,0.33,0.03],transform=ax1.transAxes)
	    cbar = plt.colorbar(scalarMap,cax=cax,orientation='horizontal')
	    cbar.ax.tick_params(labelsize=6,pad=1.3)
	    cbar.ax.xaxis.set_ticks_position('top')
	    cbar.set_label(r'$\sigma_{\rm maj}/\sqrt{GM/r_0}$')
	    cbar.ax.xaxis.labelpad = -30
    for dd,e,d in zip(dens,ells_xy,diagonals_xy):
        ax1.add_artist(e)
        e.set_clip_box(ax1.bbox)
        e.set_facecolor(scalarMap.to_rgba(np.sqrt(np.max(d[0])/dd)))

	diagonals_xy = [eigsorted(np.array([[i,j],[j,k]])) for i,j,k in zip(sigrr,sigrp,sigpp)]
	offset_xy = np.zeros(len(diagonals_xy))
	for i,(xx,yy,d) in enumerate(zip(x,y,diagonals_xy)):
	    offset_xy[i]= np.degrees(np.arccos(
	            np.abs(np.dot(d[1][:,0],np.array([xx,yy])))
	            /np.sqrt(xx**2+yy**2))-np.arctan2(yy,xx))
	    if(np.abs(offset_xy[i])>45.):
	        offset_xy[i]= np.degrees(np.arccos(
	            np.abs(np.dot(d[1][:,1],np.array([xx,yy])))
	            /np.sqrt(xx**2+yy**2))-np.arctan2(yy,xx))
	    offset_xy[i]*=-1.

    # X = np.reshape(x,(sh,sh))
    # Y = np.reshape(y,(sh,sh))
    X = np.linspace(width1,width2)
    Y = np.linspace(width1,width2)
    offxygrd = griddata((x,y), offset_xy, (X[None,:], Y[:,None]), method='cubic')
    offxygrd[np.abs(offxygrd)>45.]=45.*np.sign(offxygrd[np.abs(offxygrd)>45.])
    vmin,vmax=-45.,45.
    if vlimits!=None:
    	vmin=vlimits[0]
    	vmax=vlimits[1]
    v = np.linspace(vmin,vmax,8)
    # offset_xy = np.reshape(offset_xy,(sh,sh))
    CMAP.set_under(sns.hls_palette(7)[0])
    CMAP.set_over(sns.hls_palette(7)[-1])
    CC = ax2.contourf(X,Y,offxygrd,v,cmap=CMAP)
    ax2.set_aspect('equal')
    ax2.set_xlabel(labels[0])
    # ax2.set_ylabel(labels[1])
    ax2.set_xlim(width1,width2)
    ax2.set_ylim(width1,width2)

    vmin,vmax = CC.get_clim()
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    print vmin,vmax
    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes("top", size="5%", pad=0.05)

    if(w_cb):
	    cax = f.add_axes([0.54,0.93,0.33,0.03],transform=ax2.transAxes)
	    cbar = plt.colorbar(CC,cax=cax,orientation='horizontal')
	    cbar.ax.tick_params(labelsize=6,pad=1.3)
	    cbar.ax.xaxis.set_ticks_position('top')
	    cbar.set_label(r'$\alpha/{\rm deg}$')
	    cbar.ax.xaxis.labelpad = -30

    if(draw_scale):
    	th = np.linspace(0.,np.pi/2.)
    	x = draw_scale[0]*draw_scale[1]*np.cos(th)
    	y = draw_scale[0]*draw_scale[2]*np.sin(th)
    	for i in range(2):
    		l,=ax[i].plot(x,y,'--',color=sns.color_palette()[0])
    		l.set_dashes([3,1])

    if(text):
        plt.text(0.99,0.98,text,horizontalalignment='right',verticalalignment='top',transform=ax[1].transAxes)

    plt.savefig(outputfile,bbox_inches='tight')

def load_snp_pynbody_snapshot(input_file,units_=None):

	snap = pynbody.load(input_file)

	if not units_:
		pass
		# rr=pynbody.units.IrreducibleUnit("r_s")
		# mm=pynbody.units.IrreducibleUnit("M_{tot}")
		# vv=pynbody.units.Unit("\sqrt{GM_{\rm tot}/r_s}")
		# snap.set_units_system(velocity=None,mass=mm,distance=rr)

	return snap

from scipy.interpolate import interp1d

def plot_pontzen(input_file, output_file,action_file,text=None,beta=None):
	'''
	Makes a plot of beta(r) and L(E)/L_c(E)
	These are the quantities of interest in the Pontzen paper
	'''

	fig,ax=plt.subplots(2,1,figsize=[4.,6.])
	plt.subplots_adjust(hspace=0.3)

	extract_final_snapshot(input_file,input_file+".tmp")
	process_density(input_file+".tmp",input_file+".density")
	data = read_dehnen_density_file(input_file+".density")
	call(['rm',input_file+".density",input_file+'.tmp'])
	groups, bins_c = process_data_and_bin_in_r(data)

	actiondata = read_action_file(action_file)
	groupsa, bins_a = bin_pandas_array_by_key(actiondata,'H')

	LcE = np.genfromtxt(action_file+".Lc")
	LcE_func = interp1d(LcE.T[0],LcE.T[1], kind='cubic')

	ax[0].plot(bins_c,1-.5*(groups.mean().mst[1:-1]+groups.mean().msp[1:-1])/groups.mean().msr[1:-1])
	ax[0].semilogx()
	ax[0].set_xlabel(r'$r/r_s$')
	ax[0].set_ylabel(r'$\beta(r/r_s)$')

	bins_a = bins_a-bins_a[0]

	ax[1].plot(bins_a,groupsa.mean().L[1:-1]/LcE_func(groupsa.mean().H[1:-1]))
	ax[1].semilogx()
	ax[1].set_xlabel(r'$E$')
	ax[1].set_ylabel(r'$\langle L(E)\rangle/L_c(E)$')
	ax[1].axhline(2./3.,color='k',ls='--')
	if(beta):
		ax[1].axhline((2.-2.*beta)/(3-2.*beta),color='r',ls='--')

	if(text):
		plt.text(1.,1.02,text,horizontalalignment='right',verticalalignment='bottom',transform=ax[0].transAxes)

	plt.savefig(output_file,bbox_inches='tight')

def find_cofm(data):
	R = np.array([np.sum(data['m']*data['x']),
	              np.sum(data['m']*data['y']),
	              np.sum(data['m']*data['z'])])
	R/=np.sum(data['m'])
	return R

def snapshot_to_gadget(input_file, output_gadget):
	''' Take last snapshot from input file and convert to gadget
	    format in file output_gadget '''
	csh_command='$FALCON/bin/./s2g in='+input_file+' out='+output_gadget+' times=last;'
	run_csh_command(csh_command)

def plot_rotation_curve(input_file,output_file):
	'''
		Plot a rotation curve from snapshot
	'''
	output_gadget=input_file+".gadget"
	csh_command='$FALCON/bin/./s2g in='+input_file+' out='+output_gadget+' times=last;'
	run_csh_command(csh_command)

	f=plt.figure(figsize=(5,4))
	ax=plt.gca()

	snap = load_snp_pynbody_snapshot(output_gadget)
	snap['eps']=0.001*np.ones(len(snap))
	rot_prof = pynbody.analysis.profile.Profile(snap,min=0.001,max=5.,type='log')
	ax.plot(rot_prof['rbins'],rot_prof['v_circ'])
	ax.set_xlabel(r'$R$')
	ax.set_ylabel(r'$V_c$')
	plt.savefig(output_file,bbox_inches='tight')

def plot_nbodysim(input_file,output_file,text=None,second_file=None,delete_files=True,files_not_output=True,scale_and_shape=None,use_time=None,with_vdisp=True):

	output_gadget=input_file+".gadget"
	if(files_not_output):
		csh_command='$FALCON/bin/./s2g in='+input_file+' out='+output_gadget
		if(use_time):
			csh_command+=' times='+str(use_time)+';'
		else:
			csh_command+=' times=last;'
		run_csh_command(csh_command)

	snap = load_snp_pynbody_snapshot(output_gadget)

	n_h,n_w = 3,2
	ff = 240.0/72.27
	fig,ax = plt.subplots(n_h,n_w,figsize=[ff,1.45*ff])
	# fig,ax=plt.subplots(n_w,n_h,figsize=[3.3*n_w,3.*n_h])
	# ax = [[plt.subplot2grid((3,6),(i,2*j),colspan=2,rowspan=1) for j in range(3)] for i in range(2)]
	plt.subplots_adjust(hspace=0.,wspace=0.)

	pl = pynbody.plot.image(snap.d,subplot=ax[0][0],cmap="Greys",show_cbar=False,labelx=None,labely=r'$y/r_0$',resolution=100,av_z=True)
	N = len(pl)
	W = 10.
	x = np.array([W/N*(i+.5)-.5*W for i in range(N)])
	y = x
	ax[0][0].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,locator=LogLocator(base=2.),colors=[sns.color_palette()[1]])
	ax[0][0].set_xlabel(r'$x/r_0$')
	ax[0][0].xaxis.tick_top()
	ax[0][0].xaxis.set_label_position("top")
	ax[0][0].text(0.95,0.95,r'$(x,y)$',horizontalalignment='right',verticalalignment='top',transform=ax[0][0].transAxes)
	xy_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_x(-90)
	pl = pynbody.plot.image(snap.d,subplot=ax[1][0],cmap="Greys",show_cbar=False,labelx=None,labely=r'$z/r_0$',resolution=100,av_z=True)
	ax[1][0].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,locator=LogLocator(base=2.),colors=[sns.color_palette()[1]])
	xz_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	ax[1][0].text(0.95,0.95,r'$(x,z)$',horizontalalignment='right',verticalalignment='top',transform=ax[1][0].transAxes)
	snap.rotate_x(90)
	snap.rotate_z(90)
	snap.rotate_x(90)
	snap.rotate_z(180)
	pl = pynbody.plot.image(snap.d,subplot=ax[2][0],cmap="Greys",show_cbar=False,labelx=r'$y/r_0$',labely=r'$z/r_0$',resolution=100,av_z=True)
	ax[2][0].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,locator=LogLocator(base=2.),colors=[sns.color_palette()[1]])
	ax[2][0].text(0.95,0.95,r'$(y,z)$',horizontalalignment='right',verticalalignment='top',transform=ax[2][0].transAxes)
	yz_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_z(-180)
	snap.rotate_x(-90)
	snap.rotate_z(-90)

	if(files_not_output):
		extract_final_snapshot(input_file,input_file+".tmp")
		process_density(input_file+".tmp",input_file+".density")
	data = read_dehnen_density_file(input_file+".density")
	cofm = find_cofm(data)
	ax[0][0].plot([cofm[0]],[cofm[1]],'.',color=sns.color_palette()[0])
	ax[1][0].plot([cofm[0]],[cofm[2]],'.',color=sns.color_palette()[0])
	ax[2][0].plot([cofm[1]],[cofm[2]],'.',color=sns.color_palette()[0])
	call(['rm',input_file+'.tmp'])
	if(delete_files):
		call(['rm',input_file+".density"])

	groups, bins_c = process_data_and_bin_in_r(data)

	total_mass = np.sum(data['m'])
	ax[0][1].plot(bins_c,groups.mean().rho[1:-1]/total_mass)
	ax[0][1].semilogy()
	ax[0][1].semilogx()
	# ax[0][1].xaxis.tick_top()
	# ax[0][1].xaxis.set_label_position("top")
	# ax[0][1].set_xlabel(r'$r/r_s$')
	ax[0][1].set_ylabel(r'$\rho(r/r_0)/M_{\rm tot}r_0^{-3}$')

	ax[1][1].plot(bins_c,1-.5*(groups.mean().mst[1:-1]+groups.mean().msp[1:-1])/groups.mean().msr[1:-1])
	ax[1][1].semilogx()
	# ax[1][1].set_xlabel(r'$r/r_s$')
	ax[1][1].set_ylabel(r'$\beta(r/r_0)$')

	ax[0][1].set_xticklabels([])
	ax[1][1].set_xticklabels([])
	# total_mass = np.sum(data['m'])
	# part_num = len(data['m'])
	# count,bins,patch = ax[1][2].hist(data['m']/total_mass*part_num,bins=30,range=[0.5,1.5],histtype='step',normed=True,lw=2,log=True)
	# ax[1][2].set_xlabel(r'$w_i N_{\rm tot}/M_{\rm tot}$')
	# ax[1][2].set_ylabel(r'$p(w_i N_{\rm tot}/M_{\rm tot})$')
	# ax[1][2].set_xlim(0.5,1.5)

	if(text):
		plt.text(1.,1.02,text,horizontalalignment='right',verticalalignment='bottom',transform=ax[0][1].transAxes)

	# ax = [plt.subplot2grid((3,6),(2,2*i+1),colspan=2,rowspan=1) for i in range(2)]
	c = seaborn.color_palette()
	mass = 1./groups.mean().m[1:-1]
	ax[2][1].plot(bins_c,np.sqrt(groups.mean().msr[1:-1]*mass),c=c[0],label=r'$r$')
	ax[2][1].plot(bins_c,groups.mean().mr[1:-1]*mass,ls=':',c=c[0])
	l,=ax[2][1].plot(bins_c,np.sqrt(groups.mean().msp[1:-1]*mass),'--',c=c[1],label=r'$\phi$')
	l.set_dashes([1.5,0.5])
	ax[2][1].plot(bins_c,groups.mean().mp[1:-1]*mass,ls=':',c=c[1])
	l,=ax[2][1].plot(bins_c,np.sqrt(groups.mean().mst[1:-1]*mass),'--',c=c[2],label=r'$\theta$')
	l.set_dashes([3.,1.])
	ax[2][1].plot(bins_c,groups.mean().mt[1:-1]*mass,ls=':',c=c[2])
	ax[2][1].semilogx()
	ax[2][1].set_xlabel(r'$r/r_0$')
	ax[2][1].set_ylabel(r'$\sqrt{\langle v_i v_i \rangle},\langle v_i \rangle/ \sqrt{GM/r_0}$')
	ax[2][1].legend(frameon=False,handlelength=.7)

	# ax[1].plot(bins_c,groups.mean().mrp[1:-1]*mass*1000.,c=c[0],label=r'$i=r$, $j=\phi$')
	# l,=ax[1].plot(bins_c,groups.mean().mrt[1:-1]*mass*1000.,'--',c=c[1],label=r'$i=r$, $j=\theta$')
	# l.set_dashes([3.,1.])
	# l,=ax[1].plot(bins_c,groups.mean().mpt[1:-1]*mass*1000.,'--',c=c[2],label=r'$i=\phi$, $j=\theta$')
	# l.set_dashes([6.,1.])
	# ax[1].semilogx()
	# ax[1].set_xlabel(r'$r/r_s$')
	# ax[1].set_ylabel(r'$\langle v_i v_j \rangle/ 1000 GM/r_0$')
	# ax[1].legend(frameon=False,loc=4)

	# rot_prof = pynbody.analysis.profile.Profile(snap,min=0.001,max=5.,type='log')
	# ax[2][2].plot(rot_prof['rbins'],rot_prof['v_circ'])
	# ax[2][2].set_xlabel(r'$R$')
	# ax[2][2].set_ylabel(r'$V_c$')
	# snap['eps']=np.ones(len(snap), dtype = np.float64)*0.001
	# rot_prof = pynbody.plot.profile.rotation_curve(snap,min=0.001,max=5.,type='log',axes=ax[2][2])
	for i in range(3):
		ax[i][1].yaxis.tick_right()
		ax[i][1].yaxis.set_label_position("right")
	ax[1][1].yaxis.set_major_locator(MaxNLocator(prune='upper'))
	ax[2][1].yaxis.set_major_locator(MaxNLocator(prune='upper'))
	plt.savefig(output_file,bbox_inches='tight',dpi=400)
	if(delete_files):
		call(["rm",output_gadget])

	if not with_vdisp:
		return

	width1,width2=0.05,10.
	# groups = process_data_and_bin_in_x_and_y(data,x1=width)
	groups = process_data_and_bin_in_rphitheta(data,rr=[width1,width2],rlog=False)
	scale_and_shape_x=None
	if(scale_and_shape):
		scale_and_shape_x=[scale_and_shape[0],scale_and_shape[1],scale_and_shape[2]]
	velocity_ellipses_polar(groups.mean().x,groups.mean().y,groups.mean().m,groups.mean().msr,groups.mean().mrp,groups.mean().msp,groups.count().x,output_file[:-4]+'_tilt_xy.pdf',width1=width1,width2=width2,vlimits=[-45,45],labels=[r'$x/r_0$',r'$y/r_0$'],draw_scale=scale_and_shape_x,sig_range=[0.1,0.3])
	scale_and_shape_y=None
	if(scale_and_shape):
		scale_and_shape_y=[scale_and_shape[0],scale_and_shape[1],scale_and_shape[3]]
	groups = process_data_and_bin_in_rphitheta(data,rr=[width1,width2],rlog=False,nbins=10,theta_or_phi='phi')
	velocity_ellipses_polar(groups.mean().x,groups.mean().z,groups.mean().m,groups.mean().msr,groups.mean().mrt,groups.mean().mst,groups.count().x,output_file[:-4]+'_tilt_xz.pdf',width1=width1,width2=width2,vlimits=[-45,45],labels=[r'$x/r_0$',r'$z/r_0$'],draw_scale=scale_and_shape_y,sig_range=[0.1,0.3])
	scale_and_shape_z=None
	if(scale_and_shape):
		scale_and_shape_z=[scale_and_shape[0],scale_and_shape[2],scale_and_shape[3]]
	groups = process_data_and_bin_in_rphitheta(data,rr=[width1,width2],rlog=False,nbins=10,theta_or_phi='phimax')
	velocity_ellipses_polar(groups.mean().y,groups.mean().z,groups.mean().m,groups.mean().msr,groups.mean().mrt,groups.mean().mst,groups.count().x,output_file[:-4]+'_tilt_yz.pdf',width1=width1,width2=width2,vlimits=[-45,45],labels=[r'$y/r_0$',r'$z/r_0$'],draw_scale=scale_and_shape_z,sig_range=[0.1,0.3])

	width1,width2=0.05,2.
	# groups = process_data_and_bin_in_x_and_y(data,x1=width)
	groups = process_data_and_bin_in_rphitheta(data,rr=[width1,width2],rlog=False)
	velocity_ellipses_polar(groups.mean().x,groups.mean().y,groups.mean().m,groups.mean().msr,groups.mean().mrp,groups.mean().msp,groups.count().x,output_file[:-4]+'_tilt_xy_zoom.pdf',width1=width1,width2=width2,vlimits=[-45,45],labels=[r'$x/r_0$',r'$y/r_0$'],draw_scale=scale_and_shape_x,text=text,sig_range=[0.1,0.3])

	groups = process_data_and_bin_in_rphitheta(data,rr=[width1,width2],rlog=False,nbins=10,theta_or_phi='phi')
	velocity_ellipses_polar(groups.mean().x,groups.mean().z,groups.mean().m,groups.mean().msr,groups.mean().mrt,groups.mean().mst,groups.count().x,output_file[:-4]+'_tilt_xz_zoom.pdf',width1=width1,width2=width2,vlimits=[-45,45],labels=[r'$x/r_0$',r'$z/r_0$'],draw_scale=scale_and_shape_y,text=text,sig_range=[0.1,0.3])

	groups = process_data_and_bin_in_rphitheta(data,rr=[width1,width2],rlog=False,nbins=10,theta_or_phi='phimax')
	velocity_ellipses_polar(groups.mean().y,groups.mean().z,groups.mean().m,groups.mean().msr,groups.mean().mrt,groups.mean().mst,groups.count().x,output_file[:-4]+'_tilt_yz_zoom.pdf',width1=width1,width2=width2,vlimits=[-45,45],labels=[r'$y/r_0$',r'$z/r_0$'],draw_scale=scale_and_shape_z,text=text,sig_range=[0.1,0.3])

def plot_nbodysim_rotating(input_file,output_file,text=None,second_file=None,delete_files=True,files_not_output=True,scale_and_shape=None,use_time=None,Omegap=0.):

	output_gadget=input_file+".gadget"
	if(files_not_output):
		csh_command='$FALCON/bin/./s2g in='+input_file+' out='+output_gadget
		if(use_time):
			csh_command+=' times='+str(use_time)+';'
		else:
			csh_command+=' times=last;'
		run_csh_command(csh_command)

	snap = load_snp_pynbody_snapshot(output_gadget)

	n_h,n_w = 3,2
	ff = 240.0/72.27
	fig,ax = plt.subplots(n_h,n_w,figsize=[ff,1.45*ff])
	plt.subplots_adjust(hspace=0.,wspace=0.)

	pl = pynbody.plot.image(snap.d,subplot=ax[0][0],cmap="Greys",show_cbar=False,labelx=None,labely=r'$y/r_0$',resolution=100,av_z=True)
	N = len(pl)
	W = 10.
	x = np.array([W/N*(i+.5)-.5*W for i in range(N)])
	y = x
	ax[0][0].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,locator=LogLocator(base=2.),colors=sns.color_palette()[1])
	ax[0][0].set_xlabel(r'$x/r_0$')
	ax[0][0].xaxis.tick_top()
	ax[0][0].xaxis.set_label_position("top")
	ax[0][0].text(0.95,0.95,r'$(x,y)$',horizontalalignment='right',verticalalignment='top',transform=ax[0][0].transAxes)
	xy_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_x(-90)
	pl = pynbody.plot.image(snap.d,subplot=ax[1][0],cmap="Greys",show_cbar=False,labelx=None,labely=r'$z/r_0$',resolution=100,av_z=True)
	ax[1][0].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,locator=LogLocator(base=2.),colors=sns.color_palette()[1])
	xz_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	ax[1][0].text(0.95,0.95,r'$(x,z)$',horizontalalignment='right',verticalalignment='top',transform=ax[1][0].transAxes)
	snap.rotate_x(90)
	snap.rotate_z(90)
	snap.rotate_x(90)
	snap.rotate_z(180)
	pl = pynbody.plot.image(snap.d,subplot=ax[2][0],cmap="Greys",show_cbar=False,labelx=r'$y/r_0$',labely=r'$z/r_0$',resolution=100,av_z=True)
	ax[2][0].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,locator=LogLocator(base=2.),colors=sns.color_palette()[1])
	ax[2][0].text(0.95,0.95,r'$(y,z)$',horizontalalignment='right',verticalalignment='top',transform=ax[2][0].transAxes)
	yz_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_z(-180)
	snap.rotate_x(-90)
	snap.rotate_z(-90)

	vmax = Omegap*x.max()

	pl = pynbody.plot.image(snap,qty='vz',subplot=ax[0][1],cmap="Greys",show_cbar=False,log=False,labelx=None,labely=r'$y/r_0$',resolution=100,av_z=True)
	N = len(pl)
	W = 10.
	x = np.array([W/N*(i+.5)-.5*W for i in range(N)])
	y = x
	CS = ax[0][1].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,colors=sns.color_palette()[1])
	ax[0][1].clabel(CS, inline=1, fontsize=6)
	ax[0][1].set_xlabel(r'$x/r_0$')
	ax[0][1].xaxis.tick_top()
	ax[0][1].xaxis.set_label_position("top")
	ax[0][1].text(0.95,0.95,r'$(x,y)$',horizontalalignment='right',verticalalignment='top',transform=ax[0][1].transAxes)
	xy_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_x(-90)
	pl = pynbody.plot.image(snap,qty='vz',subplot=ax[1][1],cmap="Greys",show_cbar=False,log=False,labelx=None,labely=r'$z/r_0$',resolution=100,av_z=True)
	CS = ax[1][1].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,colors=sns.color_palette()[1])
	ax[1][1].clabel(CS, inline=1, fontsize=6)
	xz_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	ax[1][1].text(0.95,0.95,r'$(x,z)$',horizontalalignment='right',verticalalignment='top',transform=ax[1][0].transAxes)
	snap.rotate_x(90)
	snap.rotate_z(90)
	snap.rotate_x(90)
	snap.rotate_z(180)
	pl = pynbody.plot.image(snap,qty='vz',subplot=ax[2][1],cmap="Greys",show_cbar=False,log=False,labelx=r'$y/r_0$',labely=r'$z/r_0$',resolution=100,av_z=True)
	CS = ax[2][1].contour(x,y,pl,extent=[x.min(),x.max(),y.min(),y.max()],linewidths=1,colors=sns.color_palette()[1])
	ax[2][1].clabel(CS, inline=1, fontsize=6)
	ax[2][1].text(0.95,0.95,r'$(y,z)$',horizontalalignment='right',verticalalignment='top',transform=ax[2][1].transAxes)
	yz_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_z(-180)
	snap.rotate_x(-90)
	snap.rotate_z(-90)
	for i in range(3):
		ax[i][1].yaxis.tick_right()
		ax[i][1].yaxis.set_label_position("right")
	plt.savefig(output_file,bbox_inches='tight',dpi=400)
	if(delete_files):
		call(["rm",output_gadget])

import time

def live_plotter(logfile,text,cost_number=2):

	time.sleep(0)
	data = read_dehnen_log_file(logfile)

	n_x,n_y=3+cost_number,2
	fig,ax = plt.subplots(n_x,n_y,figsize=[10.,12.])
	lines=[[ax[i][j].plot([],[])[0] for j in range(n_y)] for i in range(n_x)]
	plt.subplots_adjust(hspace=0.25)

	xindex, yindex = 0,0
	for i in data.keys()[13:22]:
		ax[xindex][yindex].set_ylabel(i)
		yindex+=1
		if(yindex>1):
			xindex+=1
			yindex=0
	ax[cost_number+2][1].set_ylabel('Neff')

	if(text):
		plt.text(1.,1.02,text,horizontalalignment='right',verticalalignment='bottom',transform=ax[0][1].transAxes)

	for i in ax[:3+cost_number]:
		for j in i:
			j.set_xlabel(r'$\tau$')
	for i in ax[:2+cost_number]:
		for j in i:
			j.set_yscale('log')
	plt.ion()
	plt.show()

	try:
		while True:
			data = read_dehnen_log_file(logfile)
			xindex, yindex = 0,0
			for i in data.keys()[13:22]:
				lines[xindex][yindex].set_xdata(data['time'])
				lines[xindex][yindex].set_ydata(data[i])
				xmin,xmax = np.min(data['time']),np.max(data['time'])
				dx = (xmax-xmin)*.0
				ymin,ymax = np.min(data[i]),np.max(data[i])
				dy = (ymax-ymin)*.0
				ax[xindex][yindex].set_xlim(xmin-dx,xmax+dx)
				ax[xindex][yindex].set_ylim(ymin-dy,ymax+dy)
				yindex+=1
				if(yindex>1):
					xindex+=1
					yindex=0
			lines[cost_number+2][1].set_xdata(data['time'])
			lines[cost_number+2][1].set_ydata(data['Neff'])
			xmin,xmax = np.min(data['time']),np.max(data['time'])
			dx = (xmax-xmin)*.0
			ymin,ymax = np.min(data['Neff']),np.max(data['Neff'])
			dy = (ymax-ymin)*.0
			ax[cost_number+2][1].set_xlim(xmin-dx,xmax+dx)
			ax[cost_number+2][1].set_ylim(ymin-dy,ymax+dy)
			plt.draw()
			time.sleep(20)
	except KeyboardInterrupt:
		pass

from matplotlib import cm
import seaborn as sns
from scipy.stats import linregress
from scipy.interpolate import griddata
from scipy.special import gamma
from scipy.optimize import curve_fit

class WEmodel:

	def D(self):
		if(self.eps<0.):
			return np.sqrt(2.*np.pi)*gamma(1.-1./self.eps)*np.power(-self.eps,1-1./self.eps)*np.power(-self.zeta,1./self.zeta)/gamma(-0.5-1./self.eps)
		if(self.eps>0.):
			return np.sqrt(2.*np.pi)*gamma(1.5+1./self.eps)*np.power(self.eps,-1./self.eps)*np.power(self.zeta,1./self.zeta)/gamma(1+1./self.eps)
		if(self.eps==0.):
			return np.sqrt(2.*np.pi/np.exp(1.))

	def __init__(self,slope,b):
		self.slope=slope
		self.b=b
		self.eps = 2-slope
		self.zeta=2.*self.eps/(self.eps+2)
		self.Dp = self.D()

	def f(self,L,Jr):
		return np.power(L,-2.*self.b)*np.power(L+self.Dp*Jr,-(self.eps+4.)/(self.eps+2.)+2.*self.b)

	def Jr_atfixedf(self, f, L):
		return (np.power(f*np.power(L,2.*self.b),1./(-(self.eps+4.)/(self.eps+2.)+2.*self.b))-L)/self.Dp

class WE2model:

	def D(self):
		if(self.eps<0.):
			return np.sqrt(2.*np.pi)*gamma(1.-1./self.eps)*np.power(-self.eps,1-1./self.eps)*np.power(-self.zeta,1./self.zeta)/gamma(-0.5-1./self.eps)
		if(self.eps>0.):
			return np.sqrt(2.*np.pi)*gamma(1.5+1./self.eps)*np.power(self.eps,-1./self.eps)*np.power(self.zeta,1./self.zeta)/gamma(1+1./self.eps)
		if(self.eps==0.):
			return np.sqrt(2.*np.pi/np.exp(1.))

	def __init__(self,slope,b):
		self.slope=slope
		self.b=b
		self.eps = 2-slope
		self.zeta=2.*self.eps/(self.eps+2)
		self.Dp = self.D()

	def f(self,L,Jr,alpha,D):
		return np.power(L,-2.*self.b)*np.power(L+self.Dp*Jr,.8)

	def Jr_atfixedf(self, L, alpha, D, f, L0):
		return (np.power(f*np.power(L+L0,2.*self.b),-1./alpha)-L)/D

	def fit_params(self,xdata,ydata,p0=None):
		popt,pcov = curve_fit(self.Jr_atfixedf,xdata,ydata,p0)
		return popt

def plot_action_contours(input_file,output_plot,analytic_params=None,Width=6.):
	'''
		Make contour plot for actions
	'''
	data = read_action_file(input_file)
	ap = False
	if(analytic_params):
		ap = True
	ax = [[0.,0.],[0.,0.]]
	fig,axx=plt.subplots(1+ap,2,figsize=[8.,3.*(1+ap)])
	if(ap):
		ax = axx
	else:
		ax[0][0]=axx[0]
		ax[0][1]=axx[1]
	plt.subplots_adjust(wspace=0.3,hspace=0.25)
	for i in range(2):
		ax[0][i].set_xlabel(r'$J_r/\sqrt{GM_{\rm tot}r_s}$')
		ax[0][i].set_ylabel(r'$L/\sqrt{GM_{\rm tot}r_s}$')
	counts,ybins,xbins,image = ax[0][0].hist2d(data['J_r'],data['L'],bins=100,norm=LogNorm(),range=[[0.,Width],[0.,Width]],weights=data['m']/data['L'])
	cs = ax[0][0].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=1,locator=LogLocator(base=2.),colors=sns.color_palette()[1])

	## Hamiltonian

	GridPoints = 100
	deltaX = 0. #Width/(GridPoints-1.)
	xi = np.linspace(0.-deltaX,Width+deltaX,100)
  	yi = np.linspace(0.-deltaX,Width+deltaX,100)
	zi = griddata((data['J_r'], data['L']), np.log(-data['H']), (xi[None,:], yi[:,None]), method='cubic')
  	CS = ax[0][1].contourf(xi,yi,zi,15,cmap=plt.cm.Blues)
  	CS = ax[0][1].contour(xi,yi,zi,15,linestyle='-',linewidths=0.5,colors='k')

	ax[0][0].contour(xi,yi,zi,15,linestyle='-',linewidths=0.25,colors='k')

	## Now fit some lines to the contours
	slopes = np.array([])
	locs = []
	paths = CS.collections
	for i in paths:
		line = i.get_paths()[0]
		v = line.vertices
		x = v[:,0]
		y = v[:,1]
		slope,intercept = linregress(x,y)[:2]
		loc = intercept/(1.-slope)
		locs.append((loc,loc))
		if(len(slopes)>0):
			slopes=np.append(slopes,slope)
		else:
			slopes = np.array([slope])
	fmt={}
	for l,s in zip(CS.levels,slopes):
		fmt[l]="{0:6.2f}".format(s)
	ax[0][1].clabel(CS,CS.levels,inline=True,fmt=fmt,fontsize=7,manual=locs)

	if(analytic_params):
		## Let's have a go at drawing the contours from Williams & Evans models
		innerslope = analytic_params[0]
		outerslope = analytic_params[1]
		anisotropy = analytic_params[2]
		WE_i = WEmodel(innerslope,anisotropy)
		WE_o = WEmodel(outerslope,anisotropy)
		WE2 = WE2model(innerslope,anisotropy)
		paths = cs.collections
		alphas = np.array([])
		Ds = np.array([])
		fs = np.array([])
		level=np.array([])
		for i,j in zip(cs.collections[1:-1],cs.levels[1:-1]):
			line = i.get_paths()[0]
			v = line.vertices
			if(len(v)>6):
				v = v[v.T[1]>0.]
				x = v[:,0]
				y = v[:,1]
				alpha,D,f,L0=WE2.fit_params(y,x,np.array([2.,1.,0.1,1.]))
				print alpha,D,f,L0
				if(len(alphas)>0):
					alphas=np.append(alphas,alpha)
					Ds=np.append(Ds,D)
					fs=np.append(fs,f)
					level=np.append(level,j)
				else:
					alphas = np.array([alpha])
					Ds = np.array([D])
					fs = np.array([f])
					level = np.array([j])
				ax[0][0].plot(map(lambda i,:WE2.Jr_atfixedf(i,alpha,D,f,L0),y),y,color=sns.color_palette()[2])

		fs = np.log(fs)
		level = np.log(level)-np.log(level[0])
		s,it=linregress(fs[fs<1.],level[fs<1.])[:2]
		s2,it2=linregress(fs[fs>2.],level[fs>2.])[:2]
		ax[1][0].text(1.,.95,r'Slope $1$ ='+"{0:5.2f}".format(s)+r', Slope $2$ = '+"{0:5.2f}".format(s2),horizontalalignment='right',verticalalignment='top',transform=ax[1][0].transAxes)

		ax[1][0].plot(fs,level,ls='-.',label=r'$f$')
		ax[1][0].set_xlabel(r'$\log A$')
		ax[1][0].set_ylabel(r'$\log f$')

		ax[1][1].plot(level,1./alphas,ls='-',label=r'$\alpha$')
		ax[1][1].plot(level,Ds,ls='--',label=r'$D$')
		ax[1][1].axhline(WE_i.D(),ls=':',c=sns.color_palette()[1])
		ax[1][1].axhline(WE_o.D(),ls=':',c=sns.color_palette()[1])
		ax[1][1].set_xlabel(r'$\log f$')
		ax[1][1].set_ylabel(r'$\alpha\,\mathrm{and}\,D$')
		ax[1][1].legend(frameon=False)

		# WE3 = WEmodel(outerslope,anisotropy)
		# WE4 = WE2model(outerslope,anisotropy)
		# models = [WE,WE2,WE3,WE4]
		# for n,m in enumerate(models):
		# 	for xx in xi[::10]:
		# 		yarr = np.array(map(lambda i,:m.Jr_atfixedf(m.f(xx,0.),i),xi[xi<xx]))
		# 		ax[0].plot(yarr,xi[xi<xx],color=sns.color_palette()[n+2])

	plt.savefig(output_plot,bbox_inches='tight')

def plot_radial_orbit_action_contours(input_file,output_plot,Omega=False):
	'''
		Make contour plot for fast and slow actions
	'''
	data = read_action_file(input_file)
	fig,ax=plt.subplots(1,2,figsize=[8.,3.])
	plt.subplots_adjust(wspace=0.3,hspace=0.25)

	for i in range(2):
		ax[i].set_xlabel(r'$I_s/\sqrt{GM_{\rm tot}r_s}$')
		ax[i].set_ylabel(r'$I_f/\sqrt{GM_{\rm tot}r_s}$')
	counts,ybins,xbins,image = ax[0].hist2d(data['L'],data['J_r']+.5*data['L'],bins=100,norm=LogNorm(),range=[[0.,6.],[0.,6.]],weights=data['m']/data['L'])
	cs = ax[0].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=1,locator=LogLocator(base=2.),colors=sns.color_palette()[1])

	## Hamiltonian

	Width = 6.
	GridPoints = 100
	deltaX = 0.
	xi = np.linspace(0.-deltaX,Width+deltaX,100)
  	yi = np.linspace(0.-deltaX,Width+deltaX,100)
	zi = griddata((data['L'],data['J_r']+.5*data['L']), np.log(-data['H']), (xi[None,:], yi[:,None]), method='cubic')
  	CS = ax[1].contourf(xi,yi,zi,15,cmap=plt.cm.Blues)
  	CS = ax[1].contour(xi,yi,zi,15,linestyle='-',linewidths=0.5,colors='k')

	ax[0].contour(xi,yi,zi,15,linestyle='-',linewidths=0.25,colors='k')

	if(Omega):
		zi = griddata((data['L'], data['J_r']+.5*data['L']), np.log(2.*data['Omega_t']-data['Omega_r']), (xi[None,:], yi[:,None]), method='cubic')
		CS = ax[0].contour(xi,yi,zi,15,linewidths=1,colors='r')

	# print np.shape(zi[0])
	# for i,j in range(len(ybins)-1):
	# 	ax[1][0].plot((xbins[1:]+xbins[:-1])/2.,np.log(counts.T[i]))

	# for i in cs.collections[1:-1]:
	# 	line = i.get_paths()[0]
	# 	v = line.vertices
	# 	if(len(v)>6):
	# 		v = v[v.T[1]>0.]
	# 		x = v[:,0]
	# 		y = v[:,1]
	# 		alpha,D,f,L0=WE2.fit_params(y,x,np.array([2.,1.,0.1,1.]))
	# 		print alpha,D,f,L0
	# 		if(len(alphas)>0):
	# 			alphas=np.append(alphas,alpha)
	# 			Ds=np.append(Ds,D)
	# 			fs=np.append(fs,f)
	# 			level=np.append(level,j)
	# 		else:
	# 			alphas = np.array([alpha])
	# 			Ds = np.array([D])
	# 			fs = np.array([f])
	# 			level = np.array([j])
	# 		ax[1][0].plot(map(lambda i,:WE2.Jr_atfixedf(i,alpha,D,f,L0),y),y,color=sns.color_palette()[2])

	plt.savefig(output_plot,bbox_inches='tight')

def compute_sigma_z(snap,n=100,width=10.):
	res = n
	res2=res/2
	w=width
	x = np.linspace(-w/2.,w/2.,(res+1))
	y = np.linspace(-w/2.,w/2.,(res+1))
	vzd = np.zeros((res,res))
	for kx,ky,kvz,km in zip(snap['x'],snap['y'],snap['vz'],snap['mass']):
	    xx=np.floor(kx/w*(res+1)+res2)
	    yy=np.floor(ky/w*(res+1)+res2)
	    if(yy>=0 and xx>=0 and yy<res and xx<res):
	        vzd[xx][yy]+=km*kvz**2
	img = pynbody.plot.image(snap,qty='rho',width=w,resolution=n,av_z=True)
	X,Y = np.meshgrid(.5*(x[1:]+x[:-1]),.5*(y[1:]+y[:-1]))
	return np.ravel(X),np.ravel(Y),np.ravel(np.sqrt(vzd/img))

from find_galaxy import find_galaxy
from sectors_photometry import sectors_photometry
from mge_fit_sectors import mge_fit_sectors
from mge_print_contours import mge_print_contours
# from jam_axi_rms import jam_axi_rms

def mge_m2m(input_file,output_file,files_not_output=False,minlevel = 0.,ngauss = 12,scale = 0.05,rotation_angles=[0.,0.,0.],use_hubble_psf=False,sigmaPSF=[0.001],normPSF=[1]):

	output_gadget=input_file+".gadget"
	if(files_not_output):
		csh_command='$FALCON/bin/./s2g in='+input_file+' out='+output_gadget+' times=last;'
		run_csh_command(csh_command)

	snap = load_snp_pynbody_snapshot(output_gadget)
	img = pynbody.plot.image(snap.d,resolution=1000,av_z=True)
	xy_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_x(rotation_angles[0])
	snap.rotate_y(rotation_angles[1])
	snap.rotate_z(rotation_angles[2])
	img = pynbody.plot.image(snap.d,resolution=1000,av_z=True)

	if(use_hubble_psf):
	    # Here we use an accurate four gaussians MGE PSF for
	    # the HST/WFPC2/F814W filter, taken from Table 3 of
	    # Cappellari et al. (2002, ApJ, 578, 787)

		sigmaPSF = [0.494, 1.44, 4.71, 13.4]      # In PC1 pixels
		normPSF = [0.294, 0.559, 0.0813, 0.0657]  # total(normPSF)=1

	fig,ax = plt.subplots(2,2,figsize=(10,10))

	f = find_galaxy(img,plot=True,ax=ax[0][0])
	ell = f.eps.view(type=np.ndarray)
	th = f.theta.view(type=np.ndarray)
	s = sectors_photometry(img, ell, th, f.xpeak, f.ypeak,
                           minlevel=minlevel, plot=True,ax=ax[0][1])
	m = mge_fit_sectors(s.radius, s.angle, s.counts, ell,
                        ngauss=ngauss, sigmaPSF=sigmaPSF, normPSF=normPSF,
                        scale=scale, plot=False, bulge_disk=False)

	mge_print_contours(img, th, f.xpeak, f.ypeak, m.sol, scale=scale,
                       binning=7, sigmapsf=sigmaPSF, normpsf=normPSF, magrange=9,ax=ax[1][0])

	# Extract the central part of the image to plot at high resolution.
    # The MGE is centered to fractional pixel accuracy to ease visual comparison.
	n = 50
	xmed = f.xmed.view(type=np.ndarray)
	ymed = f.ymed.view(type=np.ndarray)
	img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
	xc, yc = n - f.xpeak + xmed, n - f.ypeak + ymed
	mge_print_contours(img, th, xc, yc, m.sol,sigmapsf=sigmaPSF,
    				   normpsf=normPSF, scale=scale,ax=ax[1][1])
	plt.savefig(output_file,bbox_inches='tight')
	return m

def jam_m2m(input_file,output_file,files_not_output=False,minlevel = 0.,ngauss = 12,scale = 0.05,rotation_angles=[0.,0.,0.],use_hubble_psf=False,sigmaPSF=[0.001],normPSF=[1]):

	output_gadget=input_file+".gadget"
	if(files_not_output):
		csh_command='$FALCON/bin/./s2g in='+input_file+' out='+output_gadget+' times=last;'
		run_csh_command(csh_command)

	snap = load_snp_pynbody_snapshot(output_gadget)
	img = pynbody.plot.image(snap.d,resolution=1000,av_z=True)
	xy_profile = pynbody.analysis.profile.Profile(snap,type='log',nbins=100)
	snap.rotate_x(rotation_angles[0])
	snap.rotate_y(rotation_angles[1])
	snap.rotate_z(rotation_angles[2])
	img = pynbody.plot.image(snap.d,resolution=1000,av_z=True)

	mge = mge_m2m(input_file,output_file[:-4]+'.mge.pdf',files_not_output=files_not_output,minlevel = minlevel,ngauss = ngauss,scale = scale,rotation_angles=rotation_angles,use_hubble_psf=use_hubble_psf,sigmaPSF=sigmaPSF,normPSF=normPSF)

	surf_lum,sigma_lum,qobs_lum = mge.sol[0,:],mge.sol[1,:],mge.sol[2,:]
	surf_pot,sigma_pot,qobs_pot = mge.sol[0,:],mge.sol[1,:],mge.sol[2,:]
	inc=90
	mbh=0
	distance=0.1
	xbin,ybin,rms=compute_sigma_z(snap)
	rmsModel, ml, chi2, flux = jam_axi_rms(
	        surf_lum, sigma_lum, qobs_lum, surf_pot, sigma_pot, qobs_pot,
	        inc, mbh, distance, xbin, ybin, rms=rms, plot=True, tensor='zz',output_plot=output_file)

def get_Anlm(input_file,output_file,time,nmax,lmax,alpha,scale,symm,twonu):
	# ttype):
	# if ttype=='Lilley':
	# 	Ttype='t'
	# else:
	# 	Ttype='f'
	# csh_command='$FALCON/bin/./getAnlm in='+input_file+' out='+output_file+' times='+str(time)+' nmax='+str(nmax)+' lmax='+str(lmax)+' alpha='+str(alpha)+' scale='+str(scale)+' symm='+str(symm)+' verbose=t type=%s;'%Ttype
	csh_command='$FALCON/bin/./getAnlm in='+input_file+' out='+output_file+' times='+str(time)+' nmax='+str(nmax)+' lmax='+str(lmax)+' alpha='+str(alpha)+' scale='+str(scale)+' symm='+str(symm)+' verbose=t twonu=%i;'%twonu
	run_csh_command(csh_command)

# def print_Anlm(input_file,output_file,time,nmax,lmax,alpha,scale,symm,ttype):
# 	tmp_file = input_file+'.tmp'
# 	get_Anlm(input_file,tmp_file,time,nmax,lmax,alpha,scale,symm)
# 	csh_command='$FALCON/bin/./PrintAnlm in='+tmp_file+' out='+output_file+';'
# 	run_csh_command(csh_command)
# 	call(['rm',tmp_file])

def print_Anlm(input_file,output_file,time,nmax,lmax,alpha,scale,symm,twonu):
	tmp_file = input_file+'.tmp'
	get_Anlm(input_file,tmp_file,time,nmax,lmax,alpha,scale,symm,twonu)
	csh_command='$FALCON/bin/./PrintAnlm in='+tmp_file+' out='+output_file+';'
	run_csh_command(csh_command)
	call(['rm',tmp_file])

def display_Anlm(input_file,output_file):
	csh_command='$FALCON/bin/./PrintAnlm in='+input_file+' out='+output_file+';'
	run_csh_command(csh_command)

def generate_halo(output_file,nbody=10000000,inner=1.,outer=3.,rs=1.,eta=1.,r_t=0., b=0., M=1.,accfile=None):
	''' Sample alpha,beta,gamma model '''
	csh_command='$FALCON/bin/./mkhalo out='+output_file+' nbody='+str(nbody)+' r_s='+str(rs)+' inner='+str(inner)+' outer='+str(outer)+' eta='+str(eta)+' r_t='+str(r_t)+' b='+str(b)+' M='+str(M)
	if(accfile):
		csh_command+=' accname=NormPotExp accfile='+accfile
	csh_command+=';'
	run_csh_command(csh_command)


def stack_snapshots(infile1,infile2, outfile):
	csh_command="snapstack in1="+infile1+" in2="+infile2+" out="+outfile+";"
	run_csh_command(csh_command,use_v2=False)
	return

def convert_2_to_1(infile):
	tmpfile=infile+'.tmp'
	csh_command="s2a in="+infile+" out="+tmpfile+";"
	run_csh_command(csh_command,use_v2=False)
	num_lines = sum(1 for line in open(tmpfile) if line[0]!='#')
	call(['rm',infile])
	csh_command="a2s in="+tmpfile+" out="+infile+" read=mxv N="+str(num_lines)+";"
	run_csh_command(csh_command,use_v2=True)
	call(['rm',tmpfile])

def generate_initial_nbody_conditions(output_file,MLratio=100.,nbody=1000000,nratio=.1,rsratio=.5,b=-0.4):
	'''
		Generate initial conditions for a Plummer + NFW profile
		Units M_DM = 1, rs_DM = 1, G = 1
	'''

	generate_halo(output_file+'.dm',nbody=nbody,r_t=10.,b=b)
	generate_halo(output_file+'.stars',nbody=int(nbody*nratio),inner=0.,outer=5.,eta=2.,rs=rsratio,r_t=10.,M=1./MLratio,b=b)
	get_Anlm(output_file+'.stars',output_file+'.pots',0.,40,12,1.,1.,4)
	get_Anlm(output_file+'.dm',output_file+'.potd',0.,40,12,1.,0.5,4)
	call(['rm',output_file+'.dm',output_file+'.stars'])
	generate_halo(output_file+'.dm',nbody=nbody,r_t=10.,b=b,accfile=output_file+'.pots')
	generate_halo(output_file+'.stars',nbody=int(nbody*nratio),inner=0.,outer=5.,eta=2.,rs=rsratio,r_t=10.,M=1./MLratio,b=b,accfile=output_file+'.potd')
	#stack_snapshots(output_file+'.dm',output_file+'.stars',output_file+'.snp')
	#call(['rm',output_file+'.dm',output_file+'.stars'])
	call(['rm',output_file+'.pots',output_file+'.potd'])
	#convert_2_to_1(output_file+'.stars')
	#convert_2_to_1(output_file+'.dm')

def sumAnlm(input1,input2,output):
	csh_command="SumAnlm in1="+input1+".An in2="+input2+".An out="+output+".An;"
	run_csh_command(csh_command)

if __name__=='__main__':
	if(len(sys.argv)==3):
		make_convergence_plot(sys.argv[1],sys.argv[2])
	else:
		print "Pass .log file and name of output plotting file"
