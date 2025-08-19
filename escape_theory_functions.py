from __future__ import division
from math import *
import math
from astropy import units as u
from sympy import *
from sympy.matrices import *
import numpy as np
#import pylab as p
import scipy
import scipy.special as ss
import astropy.constants as astroc
#import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import os
import pickle
from scipy import interpolate


######## constants ########
Msun = 1.9891e+30 #kg
c = 299792.458 # km/s


"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		    ESCAPE VELOCITY PROFILES	   ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""

#total mass of a halo from Retana-Montenegro et al. 2012; f-la (8) in terms of masses of the Sun. [Msun]
def M_total(rho_0, h, n):
    return 4.*np.pi*rho_0*(h**3.)*n*ss.gamma(3.*n)

def M_einasto(r,rho_0, h, n):
    part1 = np.array((r/h)**(1./n))
    return np.array(M_total(rho_0, h, n))*(1. - (ss.gammaincc(3.*n, part1)/ss.gamma(3.*n)))
              
#define Einasto potential profile [(km/s)^2] need to multiply by 3.24077929e-29 to keep units correct
def phi_einasto(r,rho_0,h,n):
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value #Mpc km2/s^2 Msol
    part1 = np.array((r/h)**(1./n))
    part2 = np.array(r/h)
    part3 = np.array(M_total(rho_0, h, n)/r)
    return -G_newton*part3*(1. - ss.gammaincc(3.*n, part1) +  part2*ss.gamma(2.*n)*ss.gammaincc(2.*n, part1)/ss.gamma(3.*n) )

def phi_nfw(r,rho_s,r_s):
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value #Mpc km2/s^2 Msol
    return -4*np.pi*G_newton*rho_s*(r_s**2.0)*np.log(1+r/r_s)/(r/r_s)

def phi_dehnen(r,mass_0, r_s, gamma):
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value #Mpc km2/s^2 Msol
    return -G_newton*mass_0/(r_s)*(1.0/(2.0-gamma))*(1.0-(r/(r+r_s))**(2.0-gamma))

def v_esc_dehnen(theta,z,mass_0, r_s, gamma,cosmo_params, case):
    if case == 'Flatw0waCDM':
        omega_M, w0, wa, little_h =cosmo_params 
    elif case == 'FlatwCDM':
        omega_M, w, little_h =cosmo_params 
    elif case == 'wCDM':
        omega_M, omega_DE, w, little_h =cosmo_params 
    elif case == 'LambdaCDM':
        omega_M, omega_DE, little_h =cosmo_params 
    elif case == 'FlatLambdaCDM':
        omega_M,  little_h = cosmo_params
    elif case == 'natural':
        q_z,H_z,little_h = cosmo_params

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value 
    num_phases = 1
    for i in range(num_phases):   
        H_z = H_z_function(z[i],cosmo_params,case).value
        q_z= q_z_function(z[i],cosmo_params,case)
        r = theta[i,:] * D_A(z[i],cosmo_params,case).value
        if q_z < 0.:
            req = r_eq(z[i],mass_0[i],cosmo_params,case).value
            v_esc = (-2.*phi_dehnen(r,mass_0[i],r_s[i],gamma[i])
                     +2.*phi_dehnen(req,mass_0[i],r_s[i],gamma[i]) 
                     - q_z*(H_z**2.)*(r**2. - req**2.)  )**0.5
        else:
            v_esc =  ( -2.*phi_dehnen(r,mass_0[i],r_s[i],gamma[i]))**0.5
        
        if i == 0:
            r_return = r
            v_esc_return = np.array(v_esc)
        else:
            r_return = np.concatenate((r_return,r),axis=None)
            v_esc_return = np.concatenate((v_esc_return, v_esc),axis=None)

    v_esc_return = np.split(np.array(v_esc_return),num_phases)
    v_esc_return = np.reshape(v_esc_return, (num_phases,len(r)))
    r_return = np.split(np.array(r_return),num_phases)
    r_return = np.reshape(r_return,(num_phases,len(r)))
    return r_return, v_esc_return


def v_esc_einasto(theta,z,rho_0,h,n,cosmo_params, case):
    if case == 'Flatw0waCDM':
        omega_M, w0, wa, little_h =cosmo_params 
    elif case == 'FlatwCDM':
        omega_M, w, little_h =cosmo_params 
    elif case == 'wCDM':
        omega_M, omega_DE, w, little_h =cosmo_params 
    elif case == 'LambdaCDM':
        omega_M, omega_DE, little_h =cosmo_params 
    elif case == 'FlatLambdaCDM':
        omega_M,  little_h = cosmo_params
    elif case == 'natural':
        q_z,H_z,little_h = cosmo_params

    num_phases = 1
    for i in range(num_phases):   
        H_z = H_z_function(z[i],cosmo_params,case).value
        q_z= q_z_function(z[i],cosmo_params,case)
        r = theta[i,:] * D_A(z[i],cosmo_params,case).value
        if q_z < 0.:
            req = r_eq(z[i],M_total(rho_0[i], h[i], n[i]),cosmo_params,case).value
            v_esc =  (-2.*phi_einasto(r,rho_0[i],h[i],n[i])
                      +2.*phi_einasto(req,rho_0[i],h[i],n[i]) 
                      - q_z*(H_z**2.)*(r**2. - req**2.)  )**0.5
        else:
            v_esc =  ( -2.*phi_einasto(r,rho_0[i],h[i],n[i]))**0.5
    
        if i == 0:
            r_return = r
            v_esc_return = np.array(v_esc)
        else:
            r_return = np.concatenate((r_return,r),axis=None)
            v_esc_return = np.concatenate((v_esc_return, v_esc),axis=None)

    v_esc_return = np.split(np.array(v_esc_return),num_phases)
    v_esc_return = np.reshape(v_esc_return, (num_phases,len(r)))
    r_return = np.split(np.array(r_return),num_phases)
    r_return = np.reshape(r_return,(num_phases,len(r)))
    return r_return, v_esc_return


def v_esc_NFW_M200(theta,z,M200,cosmo_params,case):
    num_phases = 1
    for i in range(num_phases):
        H_z = H_z_function(z[i],cosmo_params,case).value
        q_z= q_z_function(z[i],cosmo_params,case)
        r = theta[i,:] * D_A(z[i],cosmo_params,case).value
        rho_crit = rho_crit_z(z[i],cosmo_params,case).value
        r200 =   (3*M200[i]/(4*np.pi*200*rho_crit))**(1/3.0)
        Mtot = M200[i]
        c200 =  concentration_meta(M200[i],z[i],cosmo_params,case)
        g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
        rho_s = (M200[i]/(4.*np.pi*r200**3.)) * c200**3. * g
        r_s = r200/c200

        if q_z < 0.:
            req = r_eq(z[i],Mtot,cosmo_params,case).value
            v_esc = (-2.*phi_nfw(r,rho_s,r_s) +2*phi_nfw(req,rho_s,r_s)
                     -q_z*(H_z**2.)*(r**2 - req**2) )**0.5
        else:
            v_esc = np.sqrt(-2.*phi_nfw(r,rho_s,r_s))

        if i == 0:
            r_return = r
            v_esc_return = np.array(v_esc)
        else:
            r_return = np.concatenate((r_return,r),axis=None)
            v_esc_return = np.concatenate((v_esc_return, v_esc),axis=None)

    v_esc_return = np.split(np.array(v_esc_return),num_phases)
    v_esc_return = np.reshape(v_esc_return, (num_phases,len(r)))
    r_return = np.split(np.array(r_return),num_phases)
    r_return = np.reshape(r_return,(num_phases,len(r)))
    return r_return, v_esc_return


def v_esc_NFW(theta,z,M200,C200,cosmo_params,case):
    num_phases = 1
    for i in range(num_phases):
        H_z = H_z_function(z[i],cosmo_params,case).value
        q_z= q_z_function(z[i],cosmo_params,case)
        r = theta[i,:] * D_A(z[i],cosmo_params,case).value
        rho_crit = rho_crit_z(z[i],cosmo_params,case).value
        r200 =   (3*M200[i]/(4*np.pi*200*rho_crit))**(1/3.0)
        Mtot = M200[i]
        c200 =  C200[i]
        g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
        rho_s = (M200[i]/(4.*np.pi*r200**3.)) * c200**3. * g
        r_s = r200/c200

        if q_z < 0.:
            req = r_eq(z[i],Mtot,cosmo_params,case).value
            v_esc = (-2.*phi_nfw(r,rho_s,r_s) +2*phi_nfw(req,rho_s,r_s)
                     -q_z*(H_z**2.)*(r**2 - req**2) )**0.5
        else:
            v_esc = np.sqrt(-2.*phi_nfw(r,rho_s,r_s))

        if i == 0:
            r_return = r
            v_esc_return = np.array(v_esc)
        else:
            r_return = np.concatenate((r_return,r),axis=None)
            v_esc_return = np.concatenate((v_esc_return, v_esc),axis=None)

    v_esc_return = np.split(np.array(v_esc_return),num_phases)
    v_esc_return = np.reshape(v_esc_return, (num_phases,len(r)))
    r_return = np.split(np.array(r_return),num_phases)
    r_return = np.reshape(r_return,(num_phases,len(r)))
    return r_return, v_esc_return


def v_esc_NFW_M200_qH2(theta,z,M200,qH2,cosmo_params,case):
    num_phases = 1
    for i in range(num_phases):
        r = theta[i] * D_A(z[i],cosmo_params,case).value
        rho_crit = rho_crit_z(z[i],cosmo_params,case).value
        r200 =   (3*M200[i]/(4*np.pi*200*rho_crit))**(1/3.0)
        Mtot = M200[i]
        c200 =  concentration_meta(M200[i],z[i],cosmo_params,case)
        g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
        rho_s = (M200[i]/(4.*np.pi*r200**3.)) * c200**3. * g
        r_s = r200/c200 

        if qH2 < 0.:
            req = r_eq_qH2(z[i],Mtot,qH2).value
            v_esc = (-2.*phi_nfw(r,rho_s,r_s) +2*phi_nfw(req,rho_s,r_s)
                     - qH2*(r**2 - req**2) )**0.5
        else:
            v_esc = np.sqrt(-2.*phi_nfw(r,rho_s,r_s))

        if i == 0:
            r_return = r
            v_esc_return = np.array(v_esc)
        else:
            r_return = np.concatenate((r_return,r),axis=None)
            v_esc_return = np.concatenate((v_esc_return, v_esc),axis=None)

    v_esc_return = np.split(np.array(v_esc_return),num_phases)
    r_return = np.split(np.array(r_return),num_phases)
    return r_return, v_esc_return


def v_esc_NFWs(theta,z, rho_s, r_s,cosmo_params,case):
    num_phases = 1
    r = np.linspace(0.01,10.0,10).round(3) 
    r_interp = np.linspace(0.01,10.0,100).round(3)    

    for i in range(num_phases):
        rho_crit = rho_crit_z(z[i],cosmo_params,case)
        rho_crit = rho_crit.to(u.solMass / u.Mpc**3)      
        rho_r = []
        for j in range(10):
            rho_r = np.append(rho_r,4*np.pi*integrate.quad(rhos_nfw_int,0,r[j],
                                                           args=(rho_s[i],r_s[i]))[0]/(4/3.*np.pi*(r[j]**3.0)))
        rho_r_func = interp1d(r, rho_r, kind='quadratic')
        rho_r_interp = rho_r_func(r_interp)
        w = np.where(rho_r_interp < 200*rho_crit.value)[0]
        M200_fit = 4*np.pi*integrate.quad(rhos_nfw_int,0,r_interp[w[0]],
                                          args=(rho_s[i],r_s[i]))[0]
        Mtot = M200_fit
        H_z = H_z_function(z[i],cosmo_params,case).value
        q_z= q_z_function(z[i],cosmo_params,case)
        r = theta[i] * D_A(z[i],cosmo_params,case).value       

        if q_z < 0.:
            req = r_eq(z[i],Mtot,cosmo_params,case).value
            v_esc = (-2.*phi_nfw(r,rho_s,r_s) +2*phi_nfw(req,rho_s,r_s)
                     -q_z*(H_z**2.)*(r**2 - req**2) )**0.5
        else:
            v_esc = np.sqrt(-2.*phi_nfw(r,rho_s,r_s))

        if i == 0:
            r_return = r
            v_esc_return = np.array(v_esc)
        else:
            r_return = np.concatenate((r_return,r),axis=None)
            v_esc_return = np.concatenate((v_esc_return, v_esc),axis=None)

    v_esc_return = np.split(np.array(v_esc_return),num_phases)
    r_return = np.split(np.array(r_return),num_phases)
    return r_return, v_esc_return


def v_esc_NFWs_qH2(theta,z,M200, rho_s, r_s,qH2,cosmo_params,case):
    r = theta * D_A(z,cosmo_params,case).value
    Mtot = M200

    req_cubed = -(astroc.G.to(u.Mpc * u.km**2 / u.s**2 / u.solMass)*Mtot) / (qH2) 
    req = (req_cubed)**(1.0/3.0)
    if req > 0.:
        v_esc = (-2.*phi_nfw(r,rho_s,r_s) +2*phi_nfw(req,rho_s,r_s)
                 - qH2*(r**2 - req**2) )**0.5
    else:
        v_esc = np.sqrt(-2.*phi_nfw(r,rho_s,r_s))

    return r, v_esc


"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		  HOME GROWN COSMOLOGY 			   ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
def D_A(z_c,cosmo_params,case):
    """ 
    Returns angular diameter distance in Mpc for all three cosmological cases
    as a function of redshift (z_c) for five cases corresponding to different 
    sets of cosmological parameters:
    
    I)     'FlatLambdaCDM' takes in cosmo_params = w, omega_M
    II)    'wCDM' takes in cosmo_params = w0, wa, omega_M
    III)   'LambdaCDM'
    IV)    'FlatwCDM'
    V)     'Flatw0waCDM'
    VI_    'natural'
    NOTE: 'wCDM' assumes the CPL parametrization of dark energy
    """

    if case == 'FlatLambdaCDM':
        omega_M,little_h = cosmo_params 
        H0 = little_h * 100.
        r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0])

    elif case == 'LambdaCDM':
        omega_M, omega_DE, little_h = cosmo_params 
        omega_K = 1- omega_M - omega_DE
        H0 = little_h * 100.
        if omega_K == 0.:
#            print 'WARNING: you picked a flat cosmology! omegaK = 0!'
            r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0])
        else:
            r_z = (c / (H0*np.sqrt(np.abs(omega_K)))) * np.sin( np.sqrt(np.abs(omega_K))*(integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0]))

    elif case == 'Flatw0waCDM':
        omega_M, w0, wa, little_h = cosmo_params
        H0 = little_h * 100.
        r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0])

    elif case == 'FlatwCDM':
        omega_M, w, little_h = cosmo_params
        H0 = little_h * 100.
        r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0])
                      
    elif case == 'wCDM':
        omega_M, omega_DE, w,little_h = cosmo_params
        omega_K = 1- omega_M - omega_DE
        H0 = little_h * 100.
        if omega_K == 0.:
#            print 'WARNING: you picked a flat cosmology! omegaK = 0!'
            r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0])
        else:
            r_z = (c / (H0*np.sqrt(np.abs(omega_K)))) * np.sin( np.sqrt(np.abs(omega_K))*(integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0]))
        
    elif case == 'FlatwCDM':
        omega_M, w, little_h = cosmo_params
        H0 = little_h * 100.
        r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z_function(x,cosmo_params,case).value, 0 , z_c)[0])

    elif case == 'natural':
        q_z, H_z, little_h = cosmo_params
        H0 = little_h * 100.
        r_z = (c/H0) * ( integrate.quad(lambda x: H0/H_z, 0 , z_c)[0])

     
    return r_z/(1.+z_c) *u.Mpc



def concentration_meta(mass,redshift,cosmo_params,case):
    """
    input m200 & cosmology --> c200
    (concentration relation used in Dutton 2014)
    https://academic.oup.com/mnras/article/441/4/3359/1209689

    NOTE: input masses must be in same cosmology as little_h listed
    and in units of Msun
    I)     'FlatLambdaCDM' takes in cosmo_params = w, omega_M
    II)    'wCDM' takes in cosmo_params = w0, wa, omega_M
    III)   'LambdaCDM'
    IV)    'FlatwCDM'
    V)     'Flatw0waCDM'
    VI)    'natural'
    NOTE: 'wCDM' assumes the CPL parametrization of dark energy

    """
    if case == 'Flatw0waCDM':
        omega_M, w0, wa, little_h =cosmo_params 
    elif case == 'FlatwCDM':
        omega_M, w, little_h =cosmo_params 
    elif case == 'wCDM':
        omega_M, omega_DE, w, little_h =cosmo_params 
    elif case == 'LambdaCDM':
        omega_M, omega_DE, little_h =cosmo_params 
    elif case == 'FlatLambdaCDM':
        omega_M,  little_h = cosmo_params 
    elif case == 'natural':
        q_z, H_z,  little_h = cosmo_params 
     
    #Dutton
    #b = -.101 + (.026*redshift)
    #a = .520 + (.905-.520)*(np.exp(-.617*(redshift**1.21)))
    #Duffy
    A = 5.71 
    B = -0.084 
    C= -0.47 
    Mpivot = 2e12/little_h

    c200 = A * (mass/Mpivot)**B * (1+redshift)**C
    
    return c200
    
    #return 10**(a + (b*np.log10(mass/1e12)))
    
 
    
def q_z_function(z,cosmo_params,case):

    """ 
    Returns the deceleration parameter as a function of redshift (z_c) for three cases
    corresponding to different sets of cosmological parameters:
    
    I)     'FlatLambdaCDM' takes in cosmo_params = w, omega_M
    II)    'wCDM' takes in cosmo_params = w0, wa, omega_M
    III)   'LambdaCDM'
    IV)    'FlatwCDM'
    V)     'Flatw0waCDM'
    VI)    'natural'
    NOTE: 'wCDM' assumes the CPL parametrization of dark energy

    """
    if case == 'Flatw0waCDM':
        omega_M, w0, wa, little_h =cosmo_params 
        #assume flatness:
        omega_DE = 1. - omega_M 
        E_z= np.sqrt(omega_M*(1 + z)**3. + (omega_DE*(1 + z)**(3*(1 + w0 + wa))) * np.exp(-(3*wa*z)/(1 + z)) )
        omega_M_z = (omega_M * (1+z)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z)**(3*(1+w0+wa)) *np.exp(-3*wa*z/(1+z)) ) / E_z**2.
        q =  ((omega_M_z + omega_DE_z*(1 + 3*w0 + (3*wa*z/(1+z)) ) )/2.)
        return q

    elif case == 'FlatwCDM':
        omega_M, w, little_h =cosmo_params 
        #assume flatness:
        omega_DE = 1. - omega_M
        E_z= np.sqrt( omega_DE * (1+z)**(3.+ 3.*w)  + omega_M * (1+z)**3. )
        omega_M_z = (omega_M * (1+z)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z)**(3.+3.*w)) / E_z**2.
        q = (( omega_M_z + omega_DE_z*(1. + 3.*w) )/2.)
        return q
  
    elif case == 'wCDM':
        omega_M, omega_DE, w, little_h =cosmo_params 
        E_z= np.sqrt( omega_DE * (1+z)**(3.+ 3.*w)  + omega_M * (1+z)**3. )
        omega_M_z = (omega_M * (1+z)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z)**(3.+3.*w)) / E_z**2.
        q = (( omega_M_z + omega_DE_z*(1. + 3.*w) )/2.) 
        return q

    elif case == 'LambdaCDM':
        omega_M, omega_DE, little_h =cosmo_params 
        #omega_K != 0
        E_z= np.sqrt( omega_DE  + omega_M * (1+z)**3. + (1. - omega_DE - omega_M ) * (1+z)**2. )
        omega_M_z = ( omega_M * (1+z)**3. ) / E_z**2.
        omega_DE_z = omega_DE / E_z**2.
        q = ((omega_M_z/2.) - omega_DE_z)
        return q
    
    elif case == 'FlatLambdaCDM':
        omega_M,  little_h = cosmo_params 
        #flat,
        omega_DE = 1.- omega_M  
        E_z= np.sqrt( omega_DE  + omega_M * (1+z)**3.)
        omega_M_z = ( omega_M * (1+z)**3. ) / E_z**2.
        omega_DE_z = omega_DE / E_z**2.
        q = ((omega_M_z/2.) - omega_DE_z)
        return q
    
    elif case == 'natural':
        q_z, H_z, little_h = comso_params
        return q_z
    
def z_trans(cosmo_params, name):
    """
    Returns redshift at which q_z drops to zero.
    
    cosmology -> An astropy cosmology object.
    """
        
    redshift_array= np.arange(0,1.4,1e-7)         
    q_z_array = q_z_function(redshift_array,cosmo_params,name)

    return redshift_array[np.abs(np.subtract.outer(q_z_array, 0.)).argmin(0)]



def r_eq(z_val,M200_val,cosmo_params,case):

    M200,R200, conc, mass_0, r_s,gamma, sigma_mass_0, sigma_r_s, sigma_gamma=dehnen_nfwM200_errors(M200_val,z_val,cosmo_params,case)


    all_masses=[]
    rho_crit = rho_crit_z(z_val,cosmo_params,case)
    #structure of above are the masses enclosed within each r200_guess, for all r_s and gamma pairs
    r_range=np.linspace(1e-12,30,300)
    y = 4*np.pi*rho_Dehnen_int(r_range,mass_0, r_s, gamma)
    all_masses = integrate.cumulative_trapezoid(y, r_range,initial=0)
    
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass) #Mpc km2/s^2 kg
    a_g = ((-G_newton*all_masses)/(r_range**2)).value
    a_cosmo = ((q_z_function(z_val, cosmo_params,case) * H_z_function(z_val,cosmo_params,case)**2.)*r_range).value
    w=np.where(r_range>1)[0]
    f = interpolate.interp1d(a_cosmo[w]-a_g[w],r_range[w],fill_value='extrapolate')
    
    return float(f(0))*u.Mpc

def r_eq_qH2(z,M,qH2):
    """
    Returns the  equivalence radius in Mpc for all cosmology cases

    """
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass) #Mpc km2/s^2 kg
    r_eq_cubed = -((G_newton*M) / (qH2)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc
    
    return r_eq

def H_z_function(z,cosmo_params,case):
    """
    Returns the Hubble parameter as a function of redshift (z_c) for three cases
    corresponding to different sets of cosmological parameters:
    
    I)     'FlatLambdaCDM' takes in cosmo_params = w, omega_M
    II)    'wCDM' takes in cosmo_params = w0, wa, omega_M
    III)   'LambdaCDM'
    IV)    'FlatwCDM'
    V)     'Flatw0waCDM'
    VI)    'natural'
    NOTE: 'wCDM' assumes the CPL parametrization of dark energy
    """

    if case == 'Flatw0waCDM':
        omega_M, w0, wa, little_h = cosmo_params
        #assume flatness:
        omega_DE = 1. - omega_M
        #Using w(z) from Linder, 2003a; Chevallier and Polarski, 2001.
        H0 = little_h * 100
        return H0 * np.sqrt(omega_M*(1 + z)**3. + (omega_DE*(1 + z)**(3*(1 + w0 + wa))) * np.exp(-(3*wa*z)/(1 + z)) )*u.km / u.s / u.Mpc


    elif case == 'FlatwCDM':
        omega_M,w, little_h = cosmo_params 
        #assume flatness:
        omega_DE = 1. - omega_M
        E_z= np.sqrt( omega_DE * (1+z)**(3.+ 3.*w)  + omega_M * (1+z)**3. )
        H0 = little_h * 100
        return H0 * E_z*u.km / u.s / u.Mpc

    elif case == 'wCDM':
        omega_M, omega_DE, w,little_h = cosmo_params 
        E_z= np.sqrt( omega_DE * (1+z)**(3.+ 3.*w)  + omega_M * (1+z)**3. )
        H0 = little_h * 100      
        return H0 * E_z*u.km / u.s / u.Mpc

        
    elif case == 'LambdaCDM':
        omega_M, omega_DE, little_h = cosmo_params 
        E_z= np.sqrt( omega_DE  + omega_M * (1+z)**3. + (1.- omega_DE - omega_M ) * (1+z)**2. )      
        H0 = little_h * 100
        return H0 * E_z*u.km / u.s / u.Mpc


    elif case == 'FlatLambdaCDM':
        omega_M,  little_h = cosmo_params 
        #flat
        omega_DE = 1.- omega_M  
        E_z= np.sqrt( omega_DE  + omega_M * (1+z)**3. )     
        H0 = little_h * 100
        return H0 * E_z*u.km / u.s / u.Mpc

    elif case == 'natural':
        q_z, H_z,  little_h = cosmo_params 
        #flat      
        return H_z *u.km / u.s / u.Mpc

def rho_crit_z(z,cosmo_params,case):
    if case == 'Flatw0waCDM':
        omega_M, w0, wa, little_h = cosmo_params
    elif case == 'FlatwCDM':
        omega_M,w, little_h = cosmo_params 
    elif case == 'wCDM':
        omega_M, omega_DE, w,little_h = cosmo_params 
    elif case == 'LambdaCDM':
        omega_M, omega_DE, little_h = cosmo_params 
    elif case == 'FlatLambdaCDM':
        omega_M,  little_h = cosmo_params 
    elif case == 'natural':
        q_z, H_z,  little_h = cosmo_params     

    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass) #Mpc km2/s^2 kg
    H0= 100. * little_h *u.km/u.s/u.Mpc
    rho_crit = 3*(H_z_function(z,cosmo_params,case)**2.0)/(8*np.pi*G_newton)
    
    return rho_crit
           
           
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		   DENSITY PROFILES  	           ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""


"""define mass perc error"""
#statistical mass error on M200 increases if  cosmology is not fixed, as such, the following uncertainties in M200
# (for M200 = 4e14) are doubled (5% becomes 10% and so on)

###Nominal NFW Density Profile###
def rhos_nfw(r,rho_s,r_s):
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
           rho_s is the scale density Msol/Mpc^3
           r_s is the scale radius (Mpc)
    OUTPUT:  Msun/Mpc^3
    """
    return rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

def rhos_nfw_int(r,rho_s,r_s):
    """
    Radial r^2*rho NFW integrand for Mass measurement
    INPUT: r is radial in [Mpc]
           rho_s is the scale density Msol/Mpc^3
           r_s is the scale radius (Mpc)
    OUTPUT:  Msun/Mpc^3
    """
    return r**2 * rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

###NFW Density Profile given just M200 and an M-c relation (e.g., Sereno)###
def rho_nfw_m200(r,m200,z,cosmo_params,case):
    
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
           m200 is the mass inside r200 (Msol)
           z is the redshift
           cosmology is an astropy object
    OUTPUT:  Msun/Mpc^3
    """
    rho_crit = rho_crit_z(z,cosmo_params,case)
#    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    r200 =   (3*m200/(4*np.pi*200*rho_crit.value))**(1/3.0)
    c200 =  concentration_meta(m200,z,cosmo_params,case)
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius

    return rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

def rho_nfw_M200_int(r,m200,z,cosmo_params,case):
    
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
           m200 is the mass inside r200 (Msol)
           z is the redshift
           cosmology is an astropy object
    OUTPUT:  Msun/Mpc^3
    """
    rho_crit = rho_crit_z(z,cosmo_params,case)
#    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    r200 =   (3*m200/(4*np.pi*200*rho_crit.value))**(1/3.0)
    c200 =  concentration_meta(m200,z,cosmo_params,case)
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius

    return r**2.0*rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

###NFW Density Profile given M200, R200, and concentration###
def rho_nfw(r,m200,r200,c200):
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
           m200 is the mass inside r200 (Msol)
           r200 is physical radius containing 200 times critical density
           c200 is NFW concentration parameter
    OUTPUT:  Msun/Mpc^3
    """
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius

    return rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

def rho_nfw_int(r,m200,r200,c200):
    """
    Radial r^2*rho NFW integrand for Mass measurement
    INPUT: r is radial in [Mpc]
           m200 is the mass inside r200 (Msol)
           r200 is physical radius containing 200 times critical density
           c200 is NFW concentration parameter
    OUTPUT:  Msun/Mpc^3
    """
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius
    return r**2 * rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )


###Nominal Einasto Density Profile###
def rho_einasto(r,rho_0, h, n):    
    """
    Radial rho Einasto 
    INPUT: r is radial in [Mpc]
           rho_0 is the half density Msol/Mpc^3
           h is the half radius (Mpc)
           n is the outer slope (?)
    OUTPUT:  Msun/Mpc^3
    """
    return rho_0*np.exp(-(r/h)**(1./n))


def rho_einasto_int(r,rho_0, h, n):
    
    """
    Radial r^2*rho Einasto integrand for Mass measurement.
    INPUT: r is radial in [Mpc]
           rho_0 is the half density Msol/Mpc^3
           h is the half radius (Mpc)
           n is the outer slope (?)
    OUTPUT:  Msun/Mpc^3
    """
    return r**2*rho_0*np.exp(-(r/h)**(1./n))

def rho_dehnen(r,mass_0, r_s, gamma):
    return (3.0-gamma)*mass_0*r_s/(r**gamma)/((r+r_s)**(4-gamma))/(4.0*np.pi)

def mass_einasto(r,rho_0,h,n):
    rho_r = []
    M_ein = []
    for j in range(len(r)):
        rho_r = np.append(rho_r,4*np.pi*integrate.quad(rho_einasto_int,0,r[j],args=(rho_0,h, n))[0]/(4/3.*np.pi*(r[j]**3.0)))
        M_ein = np.append(M_ein, 4*np.pi*integrate.quad(rho_einasto_int,0,r[j],args=(rho_0,h,n))[0])
    return rho_r, M_ein

def mass_nfw_M200(r,M200_fit_50,z,cosmo_params,cosmo_name):
    rho_r = []
    M_nfw = []
    for j in range(len(r)):
        rho_r = np.append(rho_r,4*np.pi*integrate.quad(rho_nfw_M200_int,0,r[j],args=(M200_fit_50,z,cosmo_params, cosmo_name))[0]/(4/3.*np.pi*(r[j]**3.0)))
        M_nfw = np.append(M_nfw, 4*np.pi*integrate.quad(rho_nfw_M200_int,0,r[j],args=(M200_fit_50,z,cosmo_params, cosmo_name))[0])
    return rho_r, M_nfw

def mass_nfws(r,rho_s,r_s):
    rho_r = []
    M_nfws = []
    for j in range(len(r)):
        rho_r = np.append(rho_r,4*np.pi*integrate.quad(rho_nfws_int,0,r[j],args=(rho_s,r_s))[0]/(4/3.*np.pi*(r[j]**3.0)))
        M_nfws = np.append(M_nfws, 4*np.pi*integrate.quad(rho_nfws_int,0,r[j],args=(rho_s,r_s))[0])
    return rho_r, M_nfws

"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
" 		  TOOLS  	                        "
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""


def einasto_nfwM200_errors(M200, mass_perc_error, z,cosmo_params,case):
    rho_crit = rho_crit_z(z,cosmo_params,case)
#    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    R200 =  (3*M200/(4*np.pi*200.* rho_crit))**(1./3.)
    conc =  concentration_meta(M200.value,z,cosmo_params,case)
    r_array_fit = np.arange(0.1,3,0.01)
    ein_params_guess= [1e15,0.5,0.5]
    R200_uncertainty = (R200.value / 3.) * (mass_perc_error)
    M200_deltaM200_plus =  M200.value + (M200.value * mass_perc_error)
    c200_deltac200_plus = concentration_meta(M200_deltaM200_plus,z,cosmo_params,case)
    R200_deltaR200_plus = R200.value + R200_uncertainty
    end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200.value),2)+0.001) & (r_array_fit > round(find_nearest(r_array_fit,R200.value),2)-0.001) )[0]
    rho_ein_params_array = scipy.optimize.curve_fit(rho_einasto,r_array_fit[0:end_r200+1], rho_nfw(r_array_fit[0:end_r200+1],M200.value,R200.value,conc),p0=ein_params_guess,maxfev=2000)
    n =rho_ein_params_array[0][2]
    h= rho_ein_params_array[0][1]
    rho_0 = rho_ein_params_array[0][0]
    end_r200_plus, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200_deltaR200_plus),2)+0.001) & (r_array_fit > round(find_nearest(r_array_fit,R200_deltaR200_plus),2)-0.001) )[0]
    delta_plus_rho_ein_params_array = scipy.optimize.curve_fit(rho_einasto,r_array_fit[0:end_r200_plus+1], rho_nfw(r_array_fit[0:end_r200_plus+1],M200_deltaM200_plus,R200_deltaR200_plus,c200_deltac200_plus),p0=ein_params_guess,maxfev=2000)
    delta_plus_n =delta_plus_rho_ein_params_array[0][2]
    delta_plus_h= delta_plus_rho_ein_params_array[0][1]
    delta_plus_rho_0 = delta_plus_rho_ein_params_array[0][0]
    sigma_n = np.abs(n-delta_plus_n)
    sigma_h = np.abs(h - delta_plus_h)
    sigma_rho_0 = np.abs(rho_0-delta_plus_rho_0)

    return M200,R200, conc, rho_0, h,n, sigma_rho_0, sigma_h, sigma_n


def nfws_errors(M200, mass_perc_error, z,cosmo_params,case):
    rho_crit = rho_crit_z(z,cosmo_params,case)
#    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    R200 =  (3*M200/(4*np.pi*200.* rho_crit))**(1./3.)
    conc =  concentration_meta(M200,z,cosmo_params,case)
    r_array_fit = np.arange(0.1,3,0.01)
    nfw_params_guess= [1e15,0.3]
    R200_uncertainty = (R200 / 3.) * (mass_perc_error)
    M200_deltaM200_plus =  M200 + (M200 * mass_perc_error)
    c200_deltac200_plus = concentration_meta(M200_deltaM200_plus,z,cosmo_params,case)
    R200_deltaR200_plus = R200 + R200_uncertainty
    end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200.value),2)+0.005) & (r_array_fit > round(find_nearest(r_array_fit,R200.value),2)-0.005) )[0]
#   fit2 = curve_fit(lambda x, a, c: parabola(x, a, b_fixed, c), x, y) 

    rho_nfw_params_array = scipy.optimize.curve_fit(lambda r,rho_s,r_s: rhos_nfw(r,rho_s,r_s),
       r_array_fit[0:end_r200+1], rho_nfw(r_array_fit[0:end_r200+1],M200,R200.value,conc),p0=nfw_params_guess,maxfev = 100000)
    rho_s_fit =rho_nfw_params_array[0][0]
    r_s_fit = rho_nfw_params_array[0][1]
    end_r200_plus, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200_deltaR200_plus.value),2)+0.005) & (r_array_fit > round(find_nearest(r_array_fit,R200_deltaR200_plus.value),2)-0.005) )[0]
    delta_plus_rho_nfw_params_array = scipy.optimize.curve_fit(lambda r,rho_s,r_s: rhos_nfw(r,rho_s,r_s), 
        r_array_fit[0:end_r200_plus+1],
        rho_nfw(r_array_fit[0:end_r200_plus+1],M200_deltaM200_plus,R200_deltaR200_plus.value,c200_deltac200_plus),
        p0=nfw_params_guess,maxfev = 100000)
    delta_plus_rho_s =delta_plus_rho_nfw_params_array[0][0]
    delta_plus_r_s = delta_plus_rho_nfw_params_array[0][1]
    sigma_rho_s = np.abs(rho_s_fit - delta_plus_rho_s)
    sigma_r_s = np.abs(r_s_fit - delta_plus_r_s)
    return M200,R200, conc, rho_s_fit, sigma_rho_s, r_s_fit, sigma_r_s

def dehnen_nfwM200_errors(M200, z,cosmo_params,case):
    mass_perc_error=1e-30
    rho_crit = rho_crit_z(z,cosmo_params,case)
#    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    R200 =  (3*M200/(4*np.pi*200.* rho_crit.value))**(1./3.)
    conc =  concentration_meta(M200,z,cosmo_params,case)
    r_array_fit = np.arange(0.1,3,0.01)
    den_params_guess= [1e15,0.5,0.5]
    R200_uncertainty = (R200 / 3.) * (mass_perc_error)
    M200_deltaM200_plus =  M200 + (M200 * mass_perc_error)
    c200_deltac200_plus = concentration_meta(M200_deltaM200_plus,z,cosmo_params,case)
    R200_deltaR200_plus = R200 + R200_uncertainty
    end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200),2)+0.001) & (r_array_fit > round(find_nearest(r_array_fit,R200),2)-0.001) )[0]
    rho_den_params_array = scipy.optimize.curve_fit(rho_dehnen,r_array_fit[0:end_r200+1], rho_nfw(r_array_fit[0:end_r200+1],M200,R200,conc),p0=den_params_guess,maxfev=20000)

    gamma =rho_den_params_array[0][2]
    r_s= rho_den_params_array[0][1]
    mass_0 = rho_den_params_array[0][0]
    end_r200_plus, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200_deltaR200_plus),2)+0.001) & (r_array_fit > round(find_nearest(r_array_fit,R200_deltaR200_plus),2)-0.001) )[0]
    delta_plus_rho_den_params_array = scipy.optimize.curve_fit(rho_dehnen,r_array_fit[0:end_r200_plus+1], rho_nfw(r_array_fit[0:end_r200_plus+1],M200_deltaM200_plus,R200_deltaR200_plus,c200_deltac200_plus),p0=den_params_guess,maxfev=20000)
    delta_plus_gamma =delta_plus_rho_den_params_array[0][2]
    delta_plus_r_s= delta_plus_rho_den_params_array[0][1]
    delta_plus_mass_0 = delta_plus_rho_den_params_array[0][0]
    sigma_gamma = np.abs(gamma-delta_plus_gamma)
    sigma_r_s = np.abs(r_s - delta_plus_r_s)
    sigma_mass_0 = np.abs(mass_0-delta_plus_mass_0)

    return M200,R200, conc, mass_0, r_s,gamma, sigma_mass_0, sigma_r_s, sigma_gamma

def rho_Dehnen_int(r,M, r_s, gamma):
    
    """
    Radial r^2*rho Einasto integrand for Mass measurement.
    INPUT: r is radial in [Mpc]
           rho_0 is the half density Msol/Mpc^3
           h is the half radius (Mpc)
           n is the outer slope (?)
    OUTPUT:  Msun/Mpc^3
    """
    return r**2*rho_dehnen(r,M, r_s, gamma)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

