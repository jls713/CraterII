from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import seaborn
import pynbody
from m2m_tools import run_csh_command, load_snp_pynbody_snapshot,process_data_and_bin_in_r, add_to_array
from subprocess import call
try:
    from King import King
except:
    pass
import aa_py
import linecache
from scipy.integrate import quad


##
## NFW: virial radius = c*r_s (r_s=where log slope=-2)
##      virial mass = mass within virial radius
##                  = Delta * rho_crit * Vol of virial radius sphere
##      Normally Delta=200 (but Delta\approx 90 also used)
##      Critical density = 3*H_0/(8 pi G) where H_0 is Hubble constant
##      Concentration and scale radius define virial properties

kms2kpcGyr = 1./.977775

def crit_density(G=4.300918e-6,H0=67.8e-3):
    ## in units km/s, kpc, solar masses
    return 3*H0*H0/(8.*np.pi*G)

# def rho0_NFW(c):
#   ## in solar masses/kpc^3
#   rhoc = crit_density()
#   return 200./3.*rhoc*c**3/(np.log(1.+c)-c/(1.+c))

def rs_NFW(Mvir,c,Delta=200.):
    ## pass in solar masses, return kpc
    rhoc = crit_density()
    return np.power(Mvir/(Delta*rhoc*4./3.*np.pi),1./3.)/c

def Mvir_NFW(rs,c,Delta=200.):
    ## pass in kpc, return solar masses
    rhoc = crit_density()
    return (c*rs)**3*(Delta*rhoc*4./3.*np.pi)

def M_NFW(r,rs,rho0):
    return 4.*np.pi*rho0*rs**3*(np.log((rs+r)/rs)-r/(r+rs))
def M_NFW_rs_c(r,rs,c,Delta=200.):
    M = Mvir_NFW(rs,c,Delta=Delta)
    const = M/(np.log(1.+c)-c/(1.+c))
    return const*(np.log((rs+r)/rs)-r/(r+rs))

from scipy.optimize import brentq
def NFW_solve_rs(M,r,c,delta=200.):
    rhoc = crit_density()
    f = (Delta*rhoc*4./3.*np.pi)*c**3/(np.log(1.+c)-c/(1.+c))
    def fn(rs):
        return M-f*rs**3*(np.log((rs+r)/rs)-r/(r+rs))
    return brentq(fn,1e-3*r,1e3*r)

def c_NFW(Mvir,rs,Delta=200.):
    rhoc = crit_density()
    return np.power(Mvir/(Delta*rhoc*4./3.*np.pi),1./3.)/rs


def Vc_NFW(r, rs, c, Delta=200.,G=4.300918e-6):
    return np.sqrt(G*M_NFW_rs_c(r,rs,c,Delta=Delta)/r)

def rmax_NFW(rs):
    return 2.16258*rs

def Vmax_NFW(rs,c,Delta=200.,G=4.300918e-6):
    M = Mvir_NFW(rs,c,Delta=Delta)
    const = M/(np.log(1.+c)-c/(1.+c))
    return np.sqrt(G*const/rs)*0.46499096

def rs_Vmax(Vmax, c, Delta=200.,G=4.300918e-6):
    return Vmax/np.sqrt(1./(np.log(1.+c)-c/(1.+c))*c**3*4./3.*np.pi*Delta*G*crit_density())/0.46499096

def rho_NFW(r,rs,c,Delta=200.):
    Mvir = Mvir_NFW(rs,c=c,Delta=Delta)
    const = Mvir/rs**3/(np.log(1.+c)-c/(1.+c))/(4.*np.pi)
    return const*(rs/r)/(1.+r/rs)**2

def truncated_NFW_mass(r,rs,c,rt,Delta=200.):
    Mvir = Mvir_NFW(rs,c=c,Delta=Delta)
    const = Mvir/rs**3/(np.log(1.+c)-c/(1.+c))
    def rho_integrand(rr):
        return rr**2*const*(rs/rr)/(1.+rr/rs)**2/np.cosh(rr/rt)
    return quad(rho_integrand,0.,r)[0]

def truncated_double_power_mass(r,rs,rho0,abg,rt=0.):
    def rho_integrand(rr):
        if(rt>0.):
            trunc=1./np.cosh(rr/rt)
        else:
            trunc=1.
        return rr**2*((rs/rr)**abg[2])*(1.+(rr/rs)**abg[0])**(-(abg[1]-abg[2])/abg[0])*trunc
    return 4.*np.pi*rho0*quad(rho_integrand,0.,r)[0]

def load_pynbody_snapshot(input_file,use_v2=True):
    output_gadget=input_file+".gadget"
    csh_command='$FALCON/bin.gcc/./s2g in='+input_file+' out='+output_gadget+' times=last;'
    run_csh_command(csh_command,use_v2=use_v2)
    F = load_snp_pynbody_snapshot(output_gadget+'000')
    call(['rm',output_gadget+'000'])
    return F

def find_centre(input_file):
    F = load_pynbody_snapshot(input_file)
    return pynbody.analysis.halo.shrink_sphere_center(F)

def shrinking_centre(input_data,with_vel=False,mass_weight=True):


    N = len(input_data)
    centre = np.zeros(3)
    rmax=0.
    if with_vel:
        upper=7
    else:
        upper=4
    niter=0

    if mass_weight:
        centre, rmax = shrinking_centre(input_data, mass_weight=False)
        r =np.sqrt( (input_data.T[1]-centre[0])**2+(input_data.T[2]-centre[1])**2+(input_data.T[3]-centre[2])**2)
        input_data = input_data[r<5*rmax]

    while(len(input_data)>N/100. and niter<5000):
        centre = [np.sum(input_data.T[0]*input_data.T[i])/np.sum(input_data.T[0]) for i in range(1,upper)]
        if not mass_weight:
            centre = np.mean(input_data.T[1:upper],axis=1)
        r =np.sqrt( (input_data.T[1]-centre[0])**2+(input_data.T[2]-centre[1])**2+(input_data.T[3]-centre[2])**2)
        if(rmax==0.):
            rmax=np.max(r)
        rmax*=0.95
        input_data = input_data[r<rmax]
        niter+=1
    if(niter==5000):
        print 'Failed to find centre -- likely entirely unbound.'
    print centre
    return centre, rmax

def read_xv_cen_file(xvcen):
    return np.genfromtxt(xvcen)[1:7]

def dens_centre(input_file, nsphkernel=100, times='all'):
    tmp_file=input_file+'xvcen.tmp'
    csh_command="manipulate in=%s manipname=dens_centre manippars=%i manipfile=%s"%(input_file,nsphkernel,tmp_file)+" times="+str(times)+";"
    run_csh_command(csh_command)
    xvcen = read_xv_cen_file(tmp_file)
    call(['rm',tmp_file])
    return xvcen

def bound_centre(input_file, softening, Kparticles=256, alpha=3., kernel=1, external_potential=1, times='all'):
    tmp_file=input_file+'xvcen.tmp'
    csh_command="manipulate in=%s manipname=bound_centre manippars=%i,%0.3f,%i,%0.3f,%i manipfile=%s"%(input_file,Kparticles,alpha,kernel,softening,external_potential,tmp_file)+" times="+str(times)+";"
    run_csh_command(csh_command)
    xvcen = read_xv_cen_file(tmp_file)
    call(['rm',tmp_file])
    return xvcen

# def trim_bound(input_data,Ethresh=0.00,Grav=1.):
#   ''' Iteratively find energy of particles relative to total point mass, remove 1% least bound: input data = (m,x,v,potential)'''
#   N = len(input_data)
#   centre = shrinking_centre(input_data,with_vel=True)
#   rmax=0.
#   deltaN=20
#   while(deltaN>10):
#       mass = np.sum(input_data.T[0])
#       E =.5*np.sum((input_data[:,4:7]-centre[3:6])**2,axis=1)+input_data[:,7]
#       minE = np.min(E)
#       input_data = input_data[E<Ethresh*minE]
#       Np = len(input_data)
#       deltaN = N-Np
#       N = len(input_data)
#   return input_data

def trim_bound(input_data,centre=None,Ethresh=0.00,Grav=1.,mass_weight=True):
    ''' Iteratively find energy of particles relative to total point mass, remove 1% least bound: input data = (m,x,v,key,potential,...)'''
    if centre is None:
        centre, rmax = shrinking_centre(input_data, with_vel=True,
                                    mass_weight=mass_weight)
    E =.5*np.sum((input_data[:,4:7]-centre[3:6])**2,axis=1)+input_data[:,8]
    return input_data[E<Ethresh*np.min(E)]

def integrate(X,pot,N=1000,t=None):
    O = aa_py.Orbit(pot,1e-8)
    if(t==None):
        t = 10.*pot.torb(X)
    results = np.zeros((N,6))
    results[0]=X
    for i in range(1,N):
        results[i]=O.integrate(results[i-1],t/(N-1),t/(N-1),False)
    return results

def run_under_gravity(input_file, output_file,tstop,pot,potpars=None,potfile=None,epsilon=0.001,kmin=7,kmax=3,logfile=None,fac=0.01,fph=0.01,fpa=0.01,fea=0.01,Grav=1,debug=0,tau_step=0.,threads=16,step=None):
    '''
        Run N-body simulation for time with softening epsilon and max time-step = 2e-kmax
        ADVICE:
            Scaling the simulations by a radial scale (Rscale) and mass scale
            (Mscale) along with a new choice of G sets the velocity scale as
            vscale = sqrt(G Mscale/Rscale) and the time unit tunit =
            sqrt(Rscale^3/G M_scale).
            This can be done with shift_scale_snapshot routine and then passing
            Grav = G.
            Running with tau_step = tunit/2**6 appears to work pretty well.

    '''
    if(pot and (potpars==None and potfile==None)):
        print 'Must pass either potpars or potfile'
    Nlev=kmin-kmax+1
    if(tau_step==0.):
        tau_step=1./np.power(2.,kmax)/np.sqrt(Grav)
    #csh_command='gyrfalcON in='+input_file+" out="+output_file+" tstop="+str(tstop)+" eps="+str(epsilon)+" Nlev="+str(Nlev)+" kmax="+str(kmax)+" fac="+str(fac)+" fph="+str(fph)+" fea="+str(fea)+" fpa="+str(fpa)
    csh_command='griffin in='+input_file+" out="+output_file+" tstop="+str(tstop)+" eps="+str(epsilon)+" fea="+str(fea)+" fpa="+str(fpa)+" tau="+str(tau_step)+" threads="+str(threads)
    if(step):
        csh_command+=" step="+str(step)
    else:
        csh_command+=" step=0"
    # if(pot):
    #   csh_command+=" accname="+pot
    if(pot):
        csh_command+=" acc="+pot
    if(potpars):
        csh_command+=" accpars="+potpars
    if(potfile):
        csh_command+=" accstrg="+potfile
    # if(potfile):
    #   csh_command+=" accfile="+potfile

    #csh_command+=" Grav="+str(Grav)+" debug="+str(debug)
    csh_command+=" Gstd="+str(Grav)+" Gsnk="+str(Grav)+" debug="+str(debug)
    if(logfile):
        csh_command+=" logfile="+logfile
    csh_command+=" ;"
    run_csh_command(csh_command,use_v2=True)


def run_under_gravity_gyrfalcON(input_file, output_file,tstop,pot,potpars=None,potfile=None,epsilon=0.001,kmin=7,kmax=3,logfile=None,fac=0.01,fph=0.01,fpa=0.01,fea=0.01,Grav=1,debug=0,tau_step=0.,threads=16,step=None):
    '''
        Run N-body simulation for time with softening epsilon and max time-step = 2e-kmax
        ADVICE:
            Scaling the simulations by a radial scale (Rscale) and mass scale
            (Mscale) along with a new choice of G sets the velocity scale as
            vscale = sqrt(G Mscale/Rscale) and the time unit tunit =
            sqrt(Rscale^3/G M_scale).
            This can be done with shift_scale_snapshot routine and then passing
            Grav = G.
            Running with tau_step = tunit/2**6 appears to work pretty well.

    '''
    if(pot and (potpars==None and potfile==None)):
        print 'Must pass either potpars or potfile'
    Nlev=kmin-kmax+1
    if(tau_step==0.):
        tau_step=1./np.power(2.,kmax)/np.sqrt(Grav)
    csh_command='gyrfalcON in='+input_file+" out="+output_file+" tstop="+str(tstop)+" eps="+str(epsilon)+" Nlev="+str(Nlev)+" kmax="+str(kmax)+" fac="+str(fac)+" fph="+str(fph)+" fea="+str(fea)+" fpa="+str(fpa)
    if(step):
        csh_command+=" step="+str(step)
    else:
        csh_command+=" step=0"
    if(pot):
      csh_command+=" accname="+pot
    if(potpars):
        csh_command+=" accpars="+potpars
    if(potfile):
      csh_command+=" accfile="+potfile

    csh_command+=" Grav="+str(Grav)+" debug="+str(debug)
    #csh_command+=" Gstd="+str(Grav)+" Gsnk="+str(Grav)+" debug="+str(debug)
    if(logfile):
        csh_command+=" logfile="+logfile
    csh_command+=" ;"
    run_csh_command(csh_command,use_v2=False)

def make_single_particle(x,name):
    """ Generate snapshot of single particle """
    np.savetxt('tmp',[np.insert(x,0,1)])
    csh_command = "a2s in=tmp out="+name+" N=1 read=mxv;"
    run_csh_command(csh_command)


def run_orbit(x,tstop,pot,name,potfile=None,potpars=None,kmax=7,debug=0):
    """ Run an isolated particle in gyrfalcON """
    make_single_particle(x,name+'_in.snp')
    run_under_gravity_gyrfalcON(name+'_in.snp',name+'_out.snp',tstop,pot,potpars=potpars,potfile=potfile,logfile=name+'.log',kmax=kmax,kmin=kmax+5,debug=debug)
    run_csh_command("s2a in="+name+"_out.snp out="+name+".dat header=f")
    run_csh_command("rm "+name+"_out.snp")
    run_csh_command("rm "+name+"_in.snp")

def make_king_cluster(outfile,x,nbody,mass,W0,Vc=0.,rperi=0.,pot=None,rtidal=None):
    '''
        Seed a King cluster on orbit with IC x (kpc, km/s)
        Concentration W0, number of particles = nbody, mass in units M_sun
        Pass either potential pot or Vc/rperi combo
        Outputs kpc, kpc/Gyr G=1 mass units for use with gyrfalcON
    '''
    if(pot):
        R = integrate(x,pot)
        modR=np.sqrt(np.sum(R.T[:3].T**2,axis=1))
        rperi=np.min(modR)
        rmin = R[np.where(modR==rperi)][0]
        print rmin
        Vc=np.sqrt(-np.dot(pot.Forces(rmin[:3]),rmin[:3]))
        plt.plot(R.T[0],R.T[1])
        plt.savefig('tmp.pdf')
    else:
        if(Vc==0. or rperi==0.):
            print "You must pass pot or Vc,peri"
    rho0,rt,r0,sigma,conc,tdyn,eps,tr=King(nbody,W0,mass,rperi,Vc)
    if(rtidal):
        rt=rtidal
    print "Making King cluster of mass "+str(mass)+", tidal radius "+str(rt)+", veldisp "+str(sigma)+" on orbit with pericentre at "+str(rperi)
    csh_command="mkking - W0="+str(W0)+" nbody="+str(nbody)+" mass="+str(mass)+" r_t="+str(rt)+" WD_units=t | snapshift - "+outfile+" rshift="+str(x[0])+","+str(x[1])+","+str(x[2])+" vshift="+str(x[3]*kms2kpcGyr)+","+str(x[4]*kms2kpcGyr)+","+str(x[5]*kms2kpcGyr)+";"
    run_csh_command(csh_command)
    run_csh_command("s2a in="+outfile+" out="+outfile+".dat header=f")
    return eps,tr

def extract_ascii_snapshot(input_file,output_file,time):
    '''
        Extract snapshot at time as ascii from input_file
    '''
    if os.path.isfile(output_file):
        call(['rm',output_file])
    csh_command="s2a in="+input_file+" out="+output_file+" times="
    if(time=='all'):
        csh_command+='all'
    else:
        csh_command+=str(time-0.001)+":"+str(time+0.001)+";"
    run_csh_command(csh_command,use_v2=True)

def extract_snapshot(input_file,output_file,time,use_v2=True):
    '''
        Extract snapshot at time as ascii from input_file
    '''
    call(['rm',output_file])
    csh_command="s2s in="+input_file+" out="+output_file+" format='nemo' times="
    if(time=='all'):
        csh_command+='all'
    else:
        csh_command+=str(time-0.001)+":"+str(time+0.001)+";"
    run_csh_command(csh_command,use_v2=use_v2)

def convert_ascii_to_snapshot(input_file,output_file,read,N):
    '''
        Convert ascii file into snapshot
    '''
    call(['rm',output_file])
    csh_command="a2s in="+input_file+" out="+output_file+" read="+read+" N="+str(N)+" Nsink=0 Nsph=0;"
    run_csh_command(csh_command,use_v2=True)

def get_times(input_file):
    output_file=input_file+'.times.tmp'
    csh_command="tsf in="+input_file+" > "+output_file+";"
    run_csh_command(csh_command,use_v2=True)
    Times = np.array([])
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if 'double Time' in l:
                Times=np.append(Times,np.float(l.split(' ')[-2]))
    call(['rm',output_file])
    return Times

def convert_numpy_to_snapshot(input_array,output_file,read):
    np.savetxt(output_file+'.tmp',input_array)
    convert_ascii_to_snapshot(output_file+'.tmp',output_file,read,N=len(input_array))
    call(['rm',output_file+'.tmp'])

def read_snapshot_to_pandas(infile,time):
    extract_ascii_snapshot(infile,infile+'.tmp',time=time)
    hdr = linecache.getline(infile+'.tmp', 13)
    names = hdr[1:].split()
    for i in range(len(names)):
        if names[i]=='mass':
            names[i]='m'
        if names[i][:3]=='pos':
            if names[i][4]=='0':
                names[i]='x'
            elif names[i][4]=='1':
                names[i]='y'
            elif names[i][4]=='2':
                names[i]='z'
        if names[i][:3]=='vel':
            if names[i][4]=='0':
                names[i]='vx'
            elif names[i][4]=='1':
                names[i]='vy'
            elif names[i][4]=='2':
                names[i]='vz'
    data = pd.read_csv(infile+'.tmp',sep='\s+',names=names,skiprows=14,skipinitialspace=True)
    call(['rm',infile+'.tmp'])
    return data

def tidal_radius(GM,peri_radius,pot):
    vc = np.sqrt(-pot.Forces(np.array([peri_radius,0.,0.]))[0]*peri_radius)
    rt = np.power(GM*peri_radius**2/(vc*vc),1.0/3.0) # Tidal Radius
    return rt

def scale_snapshot(infile, outfile, rscale, vscale, Mscale):
    csh_command="s2s in="+infile+" out="+outfile+" scale1='"+str(rscale)+","+str(vscale)+","+str(Mscale)+"';"
    run_csh_command(csh_command,use_v2=True)
    return

def shift_snapshot(infile, outfile, x):
    csh_command="s2s in="+infile+" out="+outfile+" dx1='"+str(x[0])+","+str(x[1])+","+str(x[2])+"' dv1='"+str(x[3]*kms2kpcGyr)+","+str(x[4]*kms2kpcGyr)+","+str(x[5]*kms2kpcGyr)+"';"
    run_csh_command(csh_command,use_v2=True)
    return

def stack_snapshots(infile1,infile2, outfile):
    csh_command="s2s in="+infile1+" in2="+infile2+" out="+outfile+";"
    run_csh_command(csh_command,use_v2=True)
    return

def optimal_softening(N, kernel='P1',r_m2=.5):
    ## First row Table 2 Dehnen (2001) for Hernquist model
    ## multiplied by table in gyrfalcON manual
    ## r_m2 is radius where d ln rho/ d ln r = -2 = 1. for NFW,
    ## = .5 for Hernquist
    softt = {'P0':1., 'P1':1.43892, 'P2':2.07244, 'P3':2.56197}
    return softt[kernel]*np.power(N/1.e5,-0.23)*0.017*r_m2/.5

def stack_shift_scale_snapshot(infile, infile2, outfile, rscale, vscale, Mscale, x, rot=None, rotaxis=None):
    csh_command="s2s in="+infile+" in2="+infile2+" out="+outfile+" scale1='"+str(rscale)+","+str(vscale)+","+str(Mscale)+"' scale2='"+str(rscale)+","+str(vscale)+","+str(Mscale)+"' dx1='"+str(x[0])+","+str(x[1])+","+str(x[2])+"' dv1='"+str(x[3]*kms2kpcGyr)+","+str(x[4]*kms2kpcGyr)+","+str(x[5]*kms2kpcGyr)+"' dx2='"+str(x[0])+","+str(x[1])+","+str(x[2])+"' dv2='"+str(x[3]*kms2kpcGyr)+","+str(x[4]*kms2kpcGyr)+","+str(x[5]*kms2kpcGyr)+"'"
    if(rot is not None and rotaxis is not None):
        csh_command+=" rotangle=%0.5f rotaxis='%0.3f,%0.3f,%0.3f'"%(rot,rotaxis[0],rotaxis[1],rotaxis[2])
    csh_command += ";"
    run_csh_command(csh_command,use_v2=True)
    return

def shift_snapshot_snapshift(infile, outfile, x):
    ''' Shift by x: in units of kpc, km/s '''
    csh_command="snapshift in="+infile+" out="+outfile+" rshift="+str(x[0])+","+str(x[1])+","+str(x[2])+" vshift="+str(x[3])+","+str(x[4])+","+str(x[5])+";"
    run_csh_command(csh_command,use_v2=True)
    return

def scale_snapshot(infile, outfile, rscale, vscale, Mscale):
    ''' Scale snapshot by rscale in position, vscale velocity and Mscale mass '''
    csh_command="snapscale in="+infile+" out="+outfile+" rscale="+str(rscale)+" vscale="+str(vscale)+" mscale="+str(Mscale)+";"
    run_csh_command(csh_command,use_v2=True)
    return

def s2s(infile, outfile):
    csh_command="s2s in="+infile+" out="+outfile+" times='first';"
    run_csh_command(csh_command,use_v2=True)
    return

def s2a2s(infile, outfile,N):
    csh_command="snapprint in="+infile+" options=m,x,y,z,vx,vy,vz | a2s in=- out="+outfile+" read=mxv N="+str(N)+" Nsink=0 Nsph=0;"
    run_csh_command(csh_command,use_v2=True)
    return

def shift_scale_snapshot_old(infile, outfile, rscale, vscale, Mscale, x, N):
    ''' Shift by x'''
    scale_snapshot(infile,outfile+'.tmp',rscale,vscale,Mscale)
    shift_snapshot_snapshift(outfile+'.tmp',outfile+'.tmp2',x)
    call(['rm',outfile+'.tmp'])
    s2a2s(outfile+'.tmp2',outfile,N)
    # call(['rm',outfile+'.tmp2'])
    return

def shift_scale_snapshot(infile, outfile, rscale, vscale, Mscale, x):
    csh_command="s2s in="+infile+" out="+outfile+" scale1='"+str(rscale)+","+str(vscale)+","+str(Mscale)+"' dx1='"+str(x[0])+","+str(x[1])+","+str(x[2])+"' dv1='"+str(x[3])+","+str(x[4])+","+str(x[5])+"' format='nemo';"
    run_csh_command(csh_command,use_v2=True)
    return

from scipy.spatial import KDTree

def find_density(xdata,mdata,N=32,KernelOrder=1):
    tree = KDTree(xdata)
    nn_d,nn_i = tree.query(xdata,N)
    ## Use Epanechnikov kernel
    max_d2 = np.max(nn_d,axis=1)**-2.
    ## compute norm
    F = 0.75/np.pi
    for n in range(1,KernelOrder):
        F *= (n+n+3)/(n+n)
    return F*np.sum(mdata[nn_i].T*np.power(1.-max_d2*nn_d.T**2,KernelOrder),axis=0)*np.power(max_d2,3./2.)





    # // void SetDensity(const bodies*B, const OctTree::Leaf*L,
        # //   const Neighbour*NB, int K)
    # // {
    # //   real iHq = 1./NB[K-1].Q;
    # //   real rho = 0.;
    # //   for(int k=0; k!=K-1; ++k)
    # //     rho += (double)(NB[k].L) * std::pow(1.-iHq*NB[k].Q,N);
    # //   rho *= F * std::pow(sqrt(iHq),3);
    # //   B->aux(mybody(L)) = rho;
    # // }
    # double SetDensity(const snapshot*B, std::vector<int> I, std::vector<double> Q)
    # {
    #   real iHq = 1./Q[K-1];
    #   real rho = 0.;
    #   for(unsigned k=0; k!=K-1; ++k)
    #     rho += mass(B(I[k]))* std::pow(1.-iHq*Q[k],N);
    #   rho *= F * std::pow(sqrt(iHq),3);
    #   return rho;


def measure_moment_of_inertia(data,max_radius=np.inf,max_height=100000.):
    data_r = data[(data.R<max_radius)&(np.abs(data.z)<max_height)]
    I = np.zeros((3,3))
    I[0][0]=np.sum(data_r.m*data_r.y*data_r.y+data_r.m*data_r.z*data_r.z)
    I[0][1]=-np.sum(data_r.m*data_r.x*data_r.y)
    I[1][0]=I[0][1]
    I[1][1]=np.sum(data_r.m*data_r.x*data_r.x+data_r.m*data_r.z*data_r.z)
    I[0][2]=-np.sum(data_r.m*data_r.x*data_r.z)
    I[2][0]=I[0][2]
    I[1][2]=-np.sum(data_r.m*data_r.y*data_r.z)
    I[2][1]=I[1][2]
    I[2][2]=np.sum(data_r.m*data_r.x*data_r.x+data_r.m*data_r.y*data_r.y)
    return I

def moment_of_inertia_principal_comp(data,max_radius=np.inf):
    ''' For a disc this will be ~ z '''
    E,V = np.linalg.eig(measure_moment_of_inertia(data,max_radius))
    arg = np.argmax(np.abs(E))
    return V[arg]*np.sign(E[arg])

def moment_of_inertia_second_comp(data,max_radius=np.inf):
    ''' For a barred disc this should align with the bar's major axis '''
    E,V = np.linalg.eig(measure_moment_of_inertia(data,max_radius))
    arg = np.argsort(np.abs(E))[1]
    return V[arg]*np.sign(E[arg])

def moment_of_inertia_third_comp(data,max_radius=np.inf):
    ''' For a barred disc this should align with the bar's minor axis '''
    E,V = np.linalg.eig(measure_moment_of_inertia(data,max_radius))
    arg = np.argsort(np.abs(E))[0]
    return V[arg]*np.sign(E[arg])

def measure_pattern_speed(infile,max_radius=np.inf,timerange=[0.,200.],key_cut=240000):
    time=np.arange(timerange[0],timerange[1]+1.,1.)
    phi=np.zeros(len(time))
    f = open("mom_inertia.dat","w")
    for i,t in enumerate(time):
        data = read_snapshot_to_pandas(infile,t)
        data = add_to_array(data)
        data = data[data.key<key_cut].reset_index(drop=True)
        x = moment_of_inertia_second_comp(data,max_radius)
        phi[i] = np.arctan2(x[1],x[0])
    for i,p in enumerate(phi[1:-1]):
        dp = (phi[i+1]-phi[i-1])
        if(dp>np.pi/2.):
            dp=dp-np.pi
        if(dp<-np.pi/2.):
            dp=np.pi+dp
        omegap = dp/(time[i+1]-time[i-1])
        f.write("%0.9f "*3%(time[i],p,omegap)+"\n")
    f.close()

def convert_snapshot_to_hdf5(infile,outfile):
    csh_command='s2s in='+infile+' out='+outfile+' format=hdf5'
    run_csh_command(csh_command,use_v2=True)

def find_nbody_potential_accel(infile, outfile, Grav, eps):
    csh_command = 'griffin in=%s out=%s tstop=0. tau=0.1 eps='%(infile,outfile)+str(eps)+' Gstd='+str(Grav)+' Gsnk='+str(Grav)+';'
    run_csh_command(csh_command,use_v2=True)


import aa_py
from scipy.optimize import minimize

def fit_surface_density(input_data,centre=None,
                                solar=np.array([8.2,0.,0.]),
                                prune_radius=None, profile='Plummer'):
    if centre is None:
        centre, rmax = shrinking_centre(input_data)
    eq = np.array([aa_py.CartesianToGalactic(ii[1:4],solar)
                    for ii in input_data])
    centre_eq = aa_py.CartesianToGalactic(np.array(centre),solar)
    distance = centre_eq[2]

    if not prune_radius:
        prune_radius=np.percentile(np.sqrt(np.sum((input_data.T[1:4].T-centre[:3])**2,axis=1)),99.)
    guess_radius=np.percentile(np.sqrt(np.sum((input_data.T[1:4].T-centre[:3])**2,axis=1)),50.)
    eq = eq[np.sqrt(np.sum((input_data.T[1:4].T-centre[:3])**2,axis=1))<prune_radius]
    Nbg = np.pi*(prune_radius/distance)**2

    def transformed_xy(e,theta,ra0,dec0):
        x = np.sin(eq.T[0]-ra0)*np.cos(eq.T[1])
        y = np.sin(eq.T[1])*np.cos(dec0)-np.cos(eq.T[1])*np.sin(dec0)*np.cos(eq.T[0]-ra0)
        xtilde=(np.cos(theta)*x+np.sin(theta)*y)/(1.-e)
        ytilde = -np.sin(theta)*x+np.cos(theta)*y
        return xtilde,ytilde

    def minus_lnprob_plummer(params):
        rs,e,theta,ra0,dec0,f=params[0],params[1],params[2],params[3],params[4],params[5]
        # print params
        if(rs<0. or e<0. or e>1. or theta<0. or theta>2.*np.pi or ra0<-2.*np.pi or ra0>2.*np.pi or dec0<-np.pi or dec0>np.pi or f<0. or f>1.):
            return np.inf
        x,y=transformed_xy(e,theta,ra0,dec0)
        return -np.sum(np.logaddexp(np.log(1.-f)-2.*np.log(1.+(x**2+y**2)/rs**2)-np.log(np.pi*rs**2*(1-e)),np.log(f/Nbg)))

    def king_surface(R,rc,ck):
        return np.power(1./np.sqrt(1.+(R/rc)**2)-1./np.sqrt(1.+ck**2),2.)

    def mass_contained(r,rc,ck):
        return quad(lambda x: 2.*np.pi*x*king_surface(x,rc,ck),0.,r)[0]

    def minus_lnprob_king(params):
        rs,e,theta,ra0,dec0,f,ck=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
        # print params
        if(rs<0. or e<0. or e>1. or theta<0. or theta>2.*np.pi or ra0<-2.*np.pi or ra0>2.*np.pi or dec0<-np.pi or dec0>np.pi or f<0. or f>1. or ck<0.):
            return np.inf
        x,y=transformed_xy(e,theta,ra0,dec0)
        M = mass_contained(ck,1.,ck)
        model = np.log(1.-f)+2.*np.log(1./np.sqrt(1.+(x**2+y**2)/rs**2)-1./np.sqrt(1.+ck**2))-np.log(M*rs**2*(1-e))
        # return np.sum(model)
        return -np.sum(np.logaddexp(model,np.log(f/Nbg))[model==model])\
               -np.log(f/Nbg)*np.count_nonzero(model!=model)

    if profile=="Plummer":
        init0=[guess_radius/distance,0.05,0.6,
               centre_eq[0],centre_eq[1],1e-4]
        results = minimize(minus_lnprob_plummer, init0, method='Nelder-Mead')
    elif profile=="King":
        x,y=transformed_xy(0.05,0.6,centre_eq[0],centre_eq[1])
        if len(x)==0:
            return np.zeros(6),[],[]
        r0=0.7/distance
        init0=[r0,0.05,0.6,
               centre_eq[0],centre_eq[1],1e-4,
               1.1*np.max(np.sqrt(x**2+y**2))/r0]
        results = minimize(minus_lnprob_king, init0, method='Nelder-Mead')
    # print results.x
    results = results.x
    results[0]*=distance
    # print results
    x,y=transformed_xy(results[1],results[2],results[3],results[4])
    # print results[0]*distance, results
    return results, x*distance, y*distance
