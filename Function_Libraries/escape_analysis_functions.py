import numpy as np
import pandas as pd
import random
from scipy.stats import skewnorm
from escape_theory_functions import (
    dehnen_nfwM200_errors,  # Used in v_esc_den_M200
    v_esc_dehnen,           # Used in v_esc_den_M200
    D_A,                    # Used in calculate_projected_quantities and get_edge
    rho_crit_z              # Used in dehnen_nfwM200_errors
)
from astropy.coordinates import SkyCoord, angular_separation
from scipy import interpolate
import pickle
import os
from astropy import units as u
from multiprocessing import Pool
from astropy import constants as const
import emcee
import matplotlib.pyplot as plt


class EscapeVelocityModeling:
    """
    Class for handling escape velocity modeling functions including 
    theoretical calculations and statistical sampling.
    """
    
    def __init__(self, path_to_calibration=None):
        """
        Initialize the modeling class.
        
        Parameters:
        -----------
        path_to_calibration : str, optional
            Path to Zv calibration data directory
        """
        self.path_to_calibration = path_to_calibration
    
    def v_esc_den_M200(self, theta, z, M200, cosmo_params, case):
        """
        Generates best-fit parameters for a Dehnen density profile 
        (fitting to an NFW with an m-c relation) to generate Dehnen escape profile.
        
        Parameters:
        -----------
        theta : array-like
            Angular positions
        z : array-like
            Redshifts
        M200 : array-like
            M200 masses
        cosmo_params : list
            Cosmological parameters
        case : str
            Cosmology case name
            
        Returns:
        --------
        tuple
            Escape velocity profile results
        """
        all_mass_0 = []
        all_r_s = []
        all_gamma = []
        
        for i in range(1):
            M200_0, R200, conc, mass_0, r_s, gamma, sigma_mass_0, sigma_r_s, sigma_gamma = \
                dehnen_nfwM200_errors(M200[i], z[i], cosmo_params, case)
            all_mass_0.append(mass_0)
            all_r_s.append(r_s)
            all_gamma.append(gamma)
            
        all_mass_0 = np.array(all_mass_0)
        all_r_s = np.array(all_r_s)
        all_gamma = np.array(all_gamma)

        return v_esc_dehnen(theta, z, all_mass_0, all_r_s, all_gamma, cosmo_params, case)

    @staticmethod
    def z_round(x, base=5):
        """
        Round redshift to nearest 0.05 for Zv calibration.
        
        Parameters:
        -----------
        x : float
            Redshift value
        base : int, default=5
            Rounding base
            
        Returns:
        --------
        float
            Rounded redshift
        """
        x = x * 100
        z_rounded = (base * round(x/base)) / 100
        if z_rounded == 0:
            z_rounded = 0.01
        return z_rounded

    @staticmethod
    def random_draw(x, f_x, N):
        """
        Randomly sample from a pdf (Zv in our case) using inverse transform sampling.
        
        Parameters:
        -----------
        x : array-like
            Points in domain to be sampled
        f_x : array-like
            PDF values
        N : int
            Number of points to sample
            
        Returns:
        --------
        array
            Random samples
        """
        # Get cdf normalized to maximum
        cdf = np.cumsum(f_x) / np.cumsum(f_x).max()
        # Inverse transform using interpolation
        inverse_cdf = interpolate.interp1d(cdf, x)
        samples = inverse_cdf(np.random.uniform(np.min(cdf), np.max(cdf), int(N)))
        return samples

    @staticmethod
    def obtain_fits(m, b, num_mem_meds):
        """
        Fit linear function to Zv (location, scale, and skewness).
        
        Parameters:
        -----------
        m : float
            Slope parameter
        b : float
            Intercept parameter
        num_mem_meds : array-like
            Number of members
            
        Returns:
        --------
        tuple
            Fitted x and y values
        """
        fit = (m * np.log10(num_mem_meds)) + b
        fit_x, fit_y = np.log10(num_mem_meds), fit
        fit_x, fit_y = 10**fit_x, 10**fit_y
        return fit_x, fit_y

    def sample_Zv(self, N, bins, M_use, z_use):
        """
        Randomly sample Zv for a given sampling, mass, and redshift.
        
        Parameters:
        -----------
        N : int
            Number of samples
        bins : int
            Number of bins
        M_use : float
            Mass to use
        z_use : float
            Redshift to use
            
        Returns:
        --------
        array
            Sampled Zv values
        """
        if self.path_to_calibration is None:
            raise ValueError("Path to calibration data must be set")
            
        M_use = np.round(M_use, 1)
        z_use = self.z_round(z_use)
        
        calib_file = os.path.join(self.path_to_calibration, f"Zv_fits_z_{z_use:.2f}_M200_{M_use:.1f}.pkl")
        self._calib_cache = {}
        key = (z_use, M_use)
        if key not in self._calib_cache:
            with open(calib_file, "rb") as f:
                self._calib_cache[key] = pickle.load(f)
        data_use, num_mem_meds = self._calib_cache[key]
        xmin, xmax = 0.1, 10
        all_rand_Zv = []
        
        for i in range(bins):
            bin_index = i

            # Skewness
            m, b = data_use[0][bin_index]
            fit_x, fit_y = self.obtain_fits(m, b, num_mem_meds)
            f_N_skewness = interpolate.interp1d(fit_x, fit_y)

            # Location
            m, b = data_use[1][bin_index]
            fit_x, fit_y = self.obtain_fits(m, b, num_mem_meds)
            f_N_loc = interpolate.interp1d(fit_x, fit_y)

            # Scale
            m, b = data_use[2][bin_index]
            fit_x, fit_y = self.obtain_fits(m, b, num_mem_meds)
            f_N_scale = interpolate.interp1d(fit_x, fit_y)

            x = np.linspace(xmin, xmax, 10000)
            a, loc, scale = f_N_skewness(N), f_N_loc(N), f_N_scale(N)

            p = skewnorm.pdf(x, a, loc, scale)
            fx = interpolate.interp1d(x, p)
            rand_Zv = self.random_draw(x, fx(x), N=1)
            all_rand_Zv.append(float(rand_Zv))
            
        return np.array(all_rand_Zv)


class ClusterDataHandler:
    """
    Class for handling cluster data processing including interloper removal,
    coordinate transformations, and phase-space analysis.
    """
    
    def __init__(self):
        """Initialize the data handler."""
        pass
    
    @staticmethod
    def shiftgapper(data, gap_prev, nbin_val, gap_val, coremin):
        """
        Standard shifting-gapper function used to identify interlopers.
        See e.g. Gifford 2015.
        
        Parameters:
        -----------
        data : array
            Stacked array of radii and velocities, shape (N, 2)
        gap_prev : float
            Initialize the gap size for initial comparison
        nbin_val : int
            Galaxies per bin
        gap_val : float
            Velocity threshold
        coremin : float
            Minimum core radius
            
        Returns:
        --------
        array
            Filtered data after interloper removal
        """
        npbin = nbin_val
        nbins = np.int32(np.ceil(data[:,0].size/(npbin*1.0)))
        origsize = data[:,0].shape[0]
        data = data[np.argsort(data[:,0])]  # sort by r to ready for binning
        
        for i in range(nbins):
            databin = data[npbin*i:npbin*(i+1)]
            datanew = None
            nsize = databin[:,0].size
            datasize = nsize - 1
            
            if nsize > 5:
                while nsize - datasize > 0 and datasize >= 5:
                    nsize = databin[:,0].size
                    databinsort = databin[np.argsort(databin[:,1])]  # sort by v
                    f = (databinsort[:,1])[databinsort[:,1].size-np.int32(np.ceil(databinsort[:,1].size/4.0))] - \
                        (databinsort[:,1])[np.int32(np.ceil(databinsort[:,1].size/4.0))]
                    gap = f / (1.349)
                    
                    if gap < gap_val:
                        break
                    if gap >= 2.0 * gap_prev:
                        gap = gap_prev
                        
                    databelow = databinsort[databinsort[:,1] <= 0]
                    gapbelow = databelow[:,1][1:] - databelow[:,1][:-1]
                    dataabove = databinsort[databinsort[:,1] > 0]
                    gapabove = dataabove[:,1][1:] - dataabove[:,1][:-1]
                    
                    try:
                        if np.max(gapbelow) >= gap:
                            vgapbelow = np.where(gapbelow >= gap)[0][-1]
                        else:
                            vgapbelow = -1
                        try:
                            datanew = np.append(datanew, databelow[vgapbelow+1:], axis=0)
                        except:
                            datanew = databelow[vgapbelow+1:]
                    except ValueError:
                        pass
                        
                    try:
                        if np.max(gapabove) >= gap:
                            vgapabove = np.where(gapabove >= gap)[0][0]
                        else:
                            vgapabove = 99999999
                        try:
                            datanew = np.append(datanew, dataabove[:vgapabove+1], axis=0)
                        except:
                            datanew = dataabove[:vgapabove+1]
                    except ValueError:
                        pass
                        
                    databin = datanew
                    datasize = datanew[:,0].size
                    datanew = None
                    
                if gap >= 2000.0:
                    gap_prev = gap
                else:
                    gap_prev = 2000.0

            try:
                datafinal = np.append(datafinal, databin, axis=0)
            except:
                datafinal = databin
                
        w1 = np.where(data[:,0] < coremin)[0]
        w2 = np.where(datafinal[:,0] > coremin)[0]
        datafinal = np.array(data[w1].tolist() + datafinal[w2].tolist())
        
        return datafinal

    @staticmethod
    def calculate_projected_quantities(gal_ras, gal_decs, gal_z, cl_ra, cl_dec, cl_z, cosmo_params, case):
        """
        Calculate projected distance and line-of-sight velocity.
        
        Parameters:
        -----------
        gal_ras, gal_decs, gal_z : array-like
            Galaxy coordinates and redshifts
        cl_ra, cl_dec, cl_z : float
            Cluster coordinates and redshift
        cosmo_params : list
            Cosmological parameters
        case : str
            Cosmology case name
            
        Returns:
        --------
        tuple
            Projected distance and line-of-sight velocity
        """
        d_A = D_A(cl_z, cosmo_params, case).value
        sep = angular_separation(np.radians(cl_ra), np.radians(cl_dec),
                               np.radians(gal_ras), np.radians(gal_decs))

        R_proj = sep * d_A  # Projected distance in Mpc
        from astropy import constants as const
        c_value = const.c.value / 1000  # speed of light in km/s
        v_los = c_value * (gal_z - cl_z) / (1 + cl_z)

        return R_proj, v_los

    def iterate_center(self, gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,
                      R200, min_r, max_r, cut, cosmo_params, case):
        """
        Iterate cluster center determination.
        
        Parameters:
        -----------
        gal_ras, gal_decs, gal_zs : array-like
            Galaxy coordinates and redshifts
        cl_ra, cl_dec, cl_z : float
            Initial cluster center coordinates and redshift
        R200 : float
            R200 radius
        min_r, max_r : float
            Minimum and maximum radial cuts in units of R200
        cut : float
            Velocity cut
        cosmo_params : list
            Cosmological parameters
        case : str
            Cosmology case name
            
        Returns:
        --------
        tuple
            Updated galaxy coordinates, projected distances, velocities, and count
        """
        r_proj, v_los = self.calculate_projected_quantities(gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,
                                         cosmo_params, case)

        mask_vlos = np.abs(v_los) < cut
        mask_r = (r_proj > min_r * R200) & (r_proj < max_r * R200)
        mask_z = np.abs(gal_zs - cl_z) < 0.05  # cluster center is measured within 0.05 in redshift

        w = np.where(mask_vlos & mask_r & mask_z)[0]

        r_proj = np.array(r_proj)[w]
        v_los = np.array(v_los)[w]
        N = len(w)

        return gal_ras[w], gal_decs[w], gal_zs[w], r_proj, v_los, N

    def iterate_center_N_times(self, gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,
                              R200, min_r, max_r, cut, cosmo_params, case, N_iterations=10):
        """
        Centering algorithm for phase-space data (iterate N=10 times). Ideally the user should check for convergence.
        
        Parameters:
        -----------
        gal_ras, gal_decs, gal_zs : array-like
            Galaxy coordinates and redshifts
        cl_ra, cl_dec, cl_z : float
            Initial cluster center coordinates and redshift
        R200 : float
            R200 radius
        min_r, max_r : float
            Minimum and maximum radial cuts in units of R200
        cut : float
            Velocity cut
        cosmo_params : list
            Cosmological parameters
        case : str
            Cosmology case name
        N_iterations :  int, optional
            Number of centering iterations (default: 10)
        --------
        tuple
            Lists of updated center coordinates and final projected data
        """
        gal_ras_news = []
        gal_decs_news = []
        gal_zs_news = []
        
        for i in range(N_iterations):
            if i == 0:
                gal_ras_new, gal_decs_new, gal_zs_new, r_proj, v_los, N = self.iterate_center(gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z,R200, min_r, max_r, cut, cosmo_params, case)
                gal_ras_news.append(np.mean(gal_ras_new))
                gal_decs_news.append(np.mean(gal_decs_new))
                gal_zs_news.append(np.mean(gal_zs_new))
            else:
                gal_ras_new, gal_decs_new, gal_zs_new, r_proj, v_los, N = self.iterate_center(gal_ras, gal_decs,gal_zs, gal_ras_news[-1],gal_decs_news[-1], gal_zs_news[-1], R200, min_r, max_r,cut, cosmo_params, case)
                gal_ras_news.append(np.mean(gal_ras_new))
                gal_decs_news.append(np.mean(gal_decs_new))
                gal_zs_news.append(np.mean(gal_zs_new))

        return gal_ras_news, gal_decs_news, gal_zs_news, r_proj, v_los, N

    @staticmethod
    def get_edge(bins, galaxy_r, galaxy_v, cl_z, R200, min_r, max_r, cut, cosmo_params, case):
        """
        Estimate the phase-space boundary, where monotonicity is enforced in the outer 3 bins.
        
        Parameters:
        -----------
        bins : int
            Number of radial bins
        galaxy_r, galaxy_v : array-like
            Galaxy projected radii and velocities
        cl_z : float
            Cluster redshift
        R200 : float
            R200 radius
        min_r, max_r : float
            Minimum and maximum radial cuts in units of R200
        cut : float
            Velocity cut
        cosmo_params : list
            Cosmological parameters
        case : str
            Cosmology case name
            
        Returns:
        --------
        tuple
            Radial positions, angular positions, and escape velocities
        """
        ww = np.linspace(min_r * R200, max_r * R200, bins + 1)
        r_fit = []
        v_fit_1 = []
        
        for i in range(len(ww) - 1):
            w = np.where(((galaxy_r) > ww[i]) & (galaxy_r < ww[i+1]) & (np.abs(galaxy_v) < cut))[0]
            if len(w) > 0:
                if i == 0 or i == 1:
                    v_fit_1.append(np.max((np.max(galaxy_v[w]), -np.min(galaxy_v[w]))))
                    r_fit.append((ww[i] + ww[i+1]) / 2)

                if i >= 2:
                    max_i = v_fit_1[-1]
                    w_use = np.where(np.abs(galaxy_v[w]) < max_i)[0]
                    galaxy_v_use = galaxy_v[w][w_use]
                    if len(galaxy_v_use) > 0:
                        v_fit_1.append(np.max((np.max(galaxy_v_use), -np.min(galaxy_v_use))))
                        r_fit.append((ww[i] + ww[i+1]) / 2)
            else:
                # Bin does not have galaxies
                r_fit.append(np.nan)
                v_fit_1.append(np.nan)

        v_fit_1 = np.array(v_fit_1)
        r_fit = np.array(r_fit)

        vesc_data_r = np.reshape(r_fit, (1, len(r_fit)))
        theta = (((r_fit * u.Mpc) / D_A(cl_z, cosmo_params, case)).value) * u.rad
        vesc_data_theta = theta.to(u.arcmin)
        vesc_data = np.reshape(v_fit_1, (1, len(v_fit_1)))

        return vesc_data_r, vesc_data_theta, vesc_data
    
        
class MCMCMassEstimator:
    """Class to handle MCMC mass estimation with multiprocessing support."""
    
    def __init__(self, M200, cl_z, N, bins, escape_modeler, cosmo_params, cosmo_name, mass_range_factor=1.5):
        self.M200 = M200
        self.cl_z = cl_z
        self.N = N
        self.bins = bins
        self.escape_modeler = escape_modeler
        self.cosmo_params = cosmo_params
        self.cosmo_name = cosmo_name
        self.mass_range_factor = mass_range_factor
        
    def lnprior(self, omega):
        """Log prior probability function."""
        p_log10M200 = omega[0]
        log10M200_min = np.log10(self.M200) - self.mass_range_factor
        log10M200_max = np.log10(self.M200) + self.mass_range_factor
        
        if not(log10M200_min < p_log10M200 < log10M200_max):
            return -np.inf
        return 0.0

    def lnlike(self, omega, x, y, yerr):
        """Log likelihood function."""
        p_theta_array = x
        p_z = np.repeat(self.cl_z, 1)
        p_M200 = np.repeat(10**omega[0], len(p_z))

        try:
            # Use the escape_modeler method
            ymodel_fixed = lambda p_theta_array, p_M200: self.escape_modeler.v_esc_den_M200(
                p_theta_array, p_z, p_M200, self.cosmo_params, self.cosmo_name)
            r_cosmo, ymodel = ymodel_fixed(p_theta_array, p_M200)

            # Use the escape_modeler method for sampling Zv
            Zv_vec = self.escape_modeler.sample_Zv(self.N, self.bins, np.log10(self.M200), self.cl_z)
            ymodel = ymodel / Zv_vec

            inv_sigma2 = 1.0 / (yerr**2)
            return np.nan_to_num(-0.5 * (np.sum((y - ymodel)**2 * inv_sigma2)))

        except (TypeError, ValueError, RuntimeError) as e:
            #print(f"Rejected M200={10**omega[0]:.2e} due to: {str(e)}")
            return -np.inf

    def lnprob(self, omega, x, y, yerr):
        """Log posterior probability function."""
        lp = self.lnprior(omega)
        if not np.isfinite(lp):
            return -np.inf
            
        ll = self.lnlike(omega, x, y, yerr)
        if not np.isfinite(ll):
            return -np.inf
            
        return lp + ll

def mass_estimation_preprocessing(cluster_positional_data, galaxy_positional_data,
                                  M200, vesc_error, coremin_cut, cut, bins,
                                  cosmo_params, cosmo_name):
    """
    Prepares the Escape Velocity Mass Estimation by estimating the edge from phase-space data.
    Raises ValueError if the run is outside the calibrated domain or if the edge is unusable.
    """
    # Model Restrictions
    N_min, N_max   = 50, 1200
    logM_min, logM_max = 14.0, 15.6
    z_min, z_max   = 0.0, 0.7
    
    # Model Hyper-parameters
    min_r, max_r   = 0.2, 1.0
    Nbin, gap      = 20, 600

    data_handler = ClusterDataHandler()
    cl_ra, cl_dec, cl_z = cluster_positional_data

    # Bounds on mass and z
    log10M = float(np.log10(M200))
    if not (logM_min <= log10M <= logM_max):
        raise ValueError(f"log10(M200)={log10M:.3f} outside [{logM_min}, {logM_max}]")
    if not (z_min <= cl_z <= z_max):
        raise ValueError(f"z={cl_z:.3f} outside [{z_min}, {z_max}]")

    # R200 from M200 and rho_crit(z)
    rho_crit = rho_crit_z(cl_z, cosmo_params, cosmo_name).to(u.Msun/u.Mpc**3).value
    R200 = (3.0 * M200 / (200.0 * 4.0 * np.pi * rho_crit)) ** (1.0/3.0)  # [Mpc]

    # Galaxy inputs (already in decimal degrees and redshift)
    gal_ras  = galaxy_positional_data[:, 0].astype(float)
    gal_decs = galaxy_positional_data[:, 1].astype(float)
    gal_zs   = galaxy_positional_data[:, 2].astype(float)

    # centering
    gal_ras_new, gal_decs_new, gal_zs_new, r_proj, v_los, N = data_handler.iterate_center(
        gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z, R200, min_r, max_r,
        cut, cosmo_params, cosmo_name
    )
    if len(gal_ras_new) == 0:
        raise ValueError("Insufficient galaxy data after first centering pass.")

    gal_ras_news, gal_decs_news, gal_zs_news, r_proj, v_los, N = data_handler.iterate_center_N_times( gal_ras, gal_decs, gal_zs, cl_ra, cl_dec, cl_z, R200, min_r, max_r, cut, cosmo_params, cosmo_name)
    
    # Ensure the re-centering has converged to within (default) 1% variations of theta_200
    def recenter_convergence_test(gal_ras_news, gal_decs_news, cl_z, R200,
                                  cosmo_params, cosmo_name, frac_tol=0.01):
        """
        Return (converged_bool, frac_shift, dtheta_rad, theta200_rad).
        Checks only the *last two* centers.
        """
        if len(gal_ras_news) < 2:
            return False, np.nan, np.nan, np.nan

        c_prev = SkyCoord(gal_ras_news[-2]*u.deg, gal_decs_news[-2]*u.deg)
        c_curr = SkyCoord(gal_ras_news[-1]*u.deg, gal_decs_news[-1]*u.deg)
        dtheta = (c_prev.separation(c_curr).to(u.rad)).value    # last-step shift

        theta200 = R200/D_A(cl_z, cosmo_params, cosmo_name).value
        frac = (dtheta / theta200)

        return (frac <= frac_tol), frac, dtheta, theta200
    
    converged, frac, dtheta, theta200 = recenter_convergence_test(gal_ras_news, gal_decs_news, cl_z, R200, cosmo_params, cosmo_name)
    if not converged:
        raise ValueError(f"Cluster re-centering not converged: last shift = {frac:.3%} of θ_200 "
                         f"({dtheta:.3e} vs θ_200={theta200:.3e}). Either proceed with caution, increase N_iterations in iterate_center_N_times,  or increase frac_tol threshold criterion in recenter_convergence_test.")
        
    # interloper removal
    data = np.vstack((r_proj, v_los)).T
    coremin = R200 * coremin_cut
    datafinal = data_handler.shiftgapper(data, 1000, Nbin, gap, coremin)
    galaxy_r, galaxy_v = datafinal[:, 0], datafinal[:, 1]

    # members within [min_r, max_r] R200 after cleaning
    N = int(np.sum((galaxy_r/R200 > min_r) & (galaxy_r/R200 < max_r)))
    if not (N_min <= N <= N_max):
        raise ValueError(f"N={N} outside calibrated range [{N_min}, {N_max}]")

    # --- edge finding ---
    vesc_data_r, vesc_data_theta, vesc_data = data_handler.get_edge(
        bins, galaxy_r, galaxy_v, cl_z, R200, min_r, max_r, cut, cosmo_params, cosmo_name
    )

    # normalize shapes (expect 1D arrays of length `bins`)
    vesc_data_r = np.asarray(vesc_data_r[0], dtype=float)     # (bins,)
    vesc_data    = np.asarray(vesc_data[0], dtype=float)       # (bins,)

    # no NaN/inf allowed in edge, AKA there must be galaxies in each bin
    if (not np.all(np.isfinite(vesc_data_r))) or (not np.all(np.isfinite(vesc_data))):
        raise ValueError("Data does not have sufficient sampling for galaxies to be contained in each radial bin.")

    # theta in radians
    vesc_data_theta = vesc_data_theta.to(u.radian).value.reshape(1, bins)
    vesc_data_err   = np.full(bins, float(vesc_error), dtype=float)
    
    #print('N:',N)
    #print(f"DEC before: {cl_dec}")
    #print(f"DEC after: {gal_decs_news[-1]}")

    #print(f"RA before: {cl_ra}")
    #print(f"RA after: {gal_ras_news[-1]}")
    #print('z before:',cl_z)
    #print('z after:',gal_zs_news[-1])

    return galaxy_r, galaxy_v, N, vesc_data_r, vesc_data_theta, vesc_data, vesc_data_err, cl_z



def run_mcmc_mass_estimation(M200, cl_z, N, vesc_data_theta, vesc_data, vesc_data_err, bins,
                             escape_modeler, cosmo_params, cosmo_name,
                             nwalkers, nsteps, n_processes,
                             mass_range_factor=1.5, progress=True):
    """
    Run MCMC estimation for cluster mass using escape velocity data.
    
    Parameters:
    -----------
    M200 : float
        Initial mass estimate in solar masses
    cl_z : float
        Cluster redshift
    N : int
        Number of galaxies in the sample
    vesc_data_theta : array
        Angular positions for escape velocity data
    vesc_data : array
        Observed escape velocity data
    vesc_data_err : array
        Uncertainties in escape velocity data
    bins : int
        Number of bins in the sample (default: 5)
    escape_modeler : object
        Escape velocity modeling object with v_esc_den_M200 and sample_Zv methods
    cosmo_params : list
        Cosmological parameters
    cosmo_name : str
        Name of cosmology model
    nwalkers : int, optional
        Number of MCMC walkers
    nsteps : int, optional
        Number of MCMC steps
    n_processes : int, optional
        Number of processes for parallelization
    mass_range_factor : float, optional
        Factor determining prior range around M200 (default: 1.5)
    progress : bool, optional
        Show progress bar (default: True)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'samples': MCMC samples
        - 'median': median of posterior
        - 'one_sig_up': Upper 1-sigma bound
        - 'one_sig_down': Lower 1-sigma bound
        - 'sampler': The emcee sampler object
    """
    
    # Create the estimator instance
    estimator = MCMCMassEstimator(M200, cl_z, N, bins, escape_modeler, cosmo_params, cosmo_name, mass_range_factor)
    
    # Set up MCMC parameters
    ndim = 1
    log10M200_min = np.log10(M200) - mass_range_factor
    log10M200_max = np.log10(M200) + mass_range_factor
    
    # Initialize walker positions
    p0 = np.transpose([np.random.uniform(log10M200_min, log10M200_max, size=nwalkers)])
    
    # Set up multiprocessing pool
    pool = Pool(processes=n_processes)
    
    try:
        # Initialize and run sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, estimator.lnprob,
                                      args=(vesc_data_theta, vesc_data, vesc_data_err),
                                      pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=progress)
        
        # Extract results
        #samples=sampler.chain.reshape((-1, 1)).flatten()
        
        # Burn-in: discard first half of *steps*
        burn = int(0.5 * nsteps)  # or 3*max(tau)
        flat = sampler.get_chain(discard=burn, flat=True)[:, 0]
        logp = sampler.get_log_prob(discard=burn, flat=True)
        samples = flat[np.isfinite(logp)]
        #try:
            #tau = sampler.get_autocorr_time(quiet=True)
            # consider thin = int(0.5 * max(tau))
        #except Exception:
            #pass
        plt.hist(samples)
        plt.xlabel(r'Posterior $M_{200}$')
        plt.show()

        # Central 68% interval
        one_sig_down, median, one_sig_up = np.quantile(samples, [0.158655, 0.5, 0.841345])

        
        print('Escape Velocity Mass Estimate:',
              np.round(median, 2), '+', np.round(one_sig_up-median, 2),
              '-', np.round(median-one_sig_down, 2))
        
        return {
            'samples': samples,
            'median': median,
            'one_sig_up': one_sig_up,
            'one_sig_down': one_sig_down,
            'sampler': sampler
        }
        
    finally:
        # Always close the pool
        pool.close()
        pool.join()

def mass_estimation_post_processing(escape_modeler,results,M200,M200_err_up,M200_err_down,N,cl_z,vesc_data_r,vesc_data_theta,
                                   vesc_data,vesc_data_err,bins,galaxy_r, galaxy_v,cluster_name,cosmo_params,cosmo_name):
        '''
        Prepares a Radial Velocity Phase-Space Given the Results of the Mass Constraint
        '''
        rho_crit = rho_crit_z(cl_z,cosmo_params,cosmo_name).to(u.Msun/u.Mpc**3)
        R200 = (3 * M200 / (200 * 4 * np.pi * rho_crit.value))**(1/3)
        # Create mass arrays for both measurements including uncertainties
        escape_masses_min = 10**(results['median'] - (results['median']-results['one_sig_down']))
        escape_masses_max = 10**(results['median'] + (results['one_sig_up']-results['median']))
        escape_masses_med = 10**results['median']
        wl_masses_min = 10**(np.log10(M200) - M200_err_down)
        wl_masses_max = 10**(np.log10(M200) + M200_err_up)
        wl_masses_med = M200
        Zv_vec0 = np.array([escape_modeler.sample_Zv(N, bins, np.log10(M200), cl_z) for l in range(1000)])
        Zv_average = np.median(Zv_vec0, axis=0)
        fig, ax = plt.subplots(figsize=(10,10))
        r_av = vesc_data_r/R200
        v_av = vesc_data
        v_std = vesc_data_err
        plt.plot(r_av, v_av, c='black',label='Edge')
        plt.plot(r_av, -v_av, c='black')
        plt.fill_between(r_av, v_av-v_std, v_av+v_std, alpha=0.3, color='r')
        plt.fill_between(r_av, -(v_av-v_std), -(v_av+v_std), alpha=0.3, color='r')
        z_use=np.repeat(cl_z,1)

        r_fit_med, v_fit_med = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(escape_masses_med,1),
                                             cosmo_params, cosmo_name)
        r_fit_min, v_fit_min = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(escape_masses_min,1),
                                             cosmo_params, cosmo_name)
        r_fit_max, v_fit_max = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(escape_masses_max,1),
                                             cosmo_params, cosmo_name)
        plt.plot(r_fit_med[0]/R200, v_fit_med[0]/Zv_average, c='b', label='Dynamical Fit')
        plt.plot(r_fit_med[0]/R200, -v_fit_med[0]/Zv_average, c='b')
        alpha_use=0.1
        plt.fill_between(r_fit_med[0]/R200,
                         v_fit_min[0]/Zv_average,
                         v_fit_max[0]/Zv_average,
                         alpha=alpha_use, color='b')
        plt.fill_between(r_fit_med[0]/R200,
                         -v_fit_min[0]/Zv_average,
                         -v_fit_max[0]/Zv_average,
                         alpha=alpha_use, color='b')


        r_theory_med, v_theory_med = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(wl_masses_med,1),
                                                  cosmo_params, cosmo_name)
        r_theory_min, v_theory_min = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(wl_masses_min,1),
                                                   cosmo_params, cosmo_name)
        r_theory_max, v_theory_max = escape_modeler.v_esc_den_M200(vesc_data_theta, z_use, np.repeat(wl_masses_max,1),
                                                   cosmo_params, cosmo_name)

        plt.plot(r_theory_med[0]/R200, v_theory_med[0]/Zv_average, c='g', label='Starting Estimate')
        plt.plot(r_theory_med[0]/R200, -v_theory_med[0]/Zv_average, c='g')
        alpha_use=0.1
        plt.fill_between(r_theory_med[0]/R200,
                         v_theory_min[0]/Zv_average,
                         v_theory_max[0]/Zv_average,
                         alpha=alpha_use, color='g')
        plt.fill_between(r_theory_med[0]/R200,
                         -v_theory_min[0]/Zv_average,
                         -v_theory_max[0]/Zv_average,
                         alpha=alpha_use, color='g')
        
        plt.title(cluster_name,fontsize=40)
        plt.scatter(galaxy_r/R200, galaxy_v, c='black')
        plt.xlabel(r'$r_{\perp}/r_{200}$', size=40)
        plt.ylabel(r'$v_{\text{los}}\,[\text{km/s}]$', size=40)
        plt.ylim(-4500, 4500)
        plt.xlim(0.19,1.01)
        plt.legend()

        fig.tight_layout()
        plt.show()
        
# main function
def main(path_to_Zv_calibration, galaxy_positional_data, cluster_positional_data,
         M200, M200_err_up, M200_err_down, cluster_name, cosmo_params, cosmo_name,
         nwalkers=250, nsteps=2000, n_processes=os.cpu_count(),
         vesc_error=30, coremin_cut=0.44, cut=4500, bins=5):
    """
    Perform escape velocity-based cluster mass estimation using MCMC analysis.
    
    This function implements a complete pipeline for estimating cluster masses using
    the escape velocity method. It processes galaxy phase-space data, removes 
    interlopers, identifies the escape velocity boundary, and performs MCMC fitting
    to constrain the cluster mass.
    
    Parameters
    ----------
    path_to_Zv_calibration : str
        Path to the directory containing Zv calibration data files. These files
        contain pre-computed calibration data for velocity anisotropy corrections
        in the format 'Zv_fits_z_{z:.2f}_M200_{M:.1f}.pkl'.
        
    galaxy_positional_data : array-like, shape (N, 3)
        Galaxy positions and redshift with columns:
        [RA_deg, DEC_deg, redshift]
        where:
        - RA_deg  : Right ascension in decimal degrees (0 <= RA_deg < 360)
        - DEC_deg : Declination in decimal degrees (-90 <= DEC_deg <= +90)
        - redshift: Dimensionless z (NOT velocity/c)

    cluster_positional_data : tuple of (float, float, float)
        Cluster coordinates and redshift as (cl_ra_deg, cl_dec_deg, cl_z) where:
        - cl_ra_deg  : Cluster right ascension in decimal degrees
        - cl_dec_deg : Cluster declination in decimal degrees
        - cl_z       : Cluster redshift (dimensionless)
        
    M200 : float
        Initial estimate of cluster mass M200 in solar masses.
        This serves as the starting point and prior center for MCMC estimation.
        
    M200_err_up : float
        Upper 1-sigma uncertainty on M200 in log10 space (dex).
        Used for plotting comparison with the dynamical mass estimate.
        
    M200_err_down : float  
        Lower 1-sigma uncertainty on M200 in log10 space (dex).
        Used for plotting comparison with the dynamical mass estimate.
        
    cluster_name : str  
       Name of the cluster you are observing, e.g. 'A7'.
       Used for plotting the phase-space diagram.
        
    cosmo_params : list
        Cosmological parameters in format dependent on `cosmo_name`:
        - 'FlatLambdaCDM': [Omega_m, h]
        - 'LambdaCDM': [Omega_m, Omega_de, h]
        - 'FlatwCDM': [Omega_m, w0, h]
        - 'wCDM': [Omega_m, Omega_de, w0, h]
        - 'Flatw0waCDM': [Omega_m, w0, wa, h]
        
    cosmo_name : str
        Name of cosmological model. Must match one of the supported types
        listed in `cosmo_params`.
        
    nwalkers : int, optional (default=250)
        Number of MCMC walkers for the ensemble sampler.
        Must be at least 2 × the number of parameters (2 for this analysis).
        More walkers increase sampling robustness but also runtime.
        
    nsteps : int, optional (default=2000)
        Number of MCMC steps per walker. Consider discarding a burn-in fraction
        (e.g., ~3× the integrated autocorrelation time) when summarizing posteriors.

    n_processes : int or None, optional (default=os.cpu_count())
        Number of parallel worker processes to use. If None, a single process is used.
        Larger values speed up likelihood evaluations at the cost of CPU usage.

    vesc_error : float, optional (default=30)
        Per-bin uncertainty (in km/s) assigned to the measured escape-velocity edge
        in the Gaussian likelihood. This sets the vertical error bars of the
        binned edge (typically ≈ 20–40 km/s depending on data quality).

    coremin_cut : float, optional (default=0.44)
        Dimensionless core exclusion radius for interloper removal via the
        shifting-gapper. Galaxies with r < coremin_cut × R200 are exempt from
        being flagged as interlopers (prevents over-cleaning the core). For 5
        radial bins over 0.2–1.0 R200, 0.44 leaves the first bin un-gapped.

    cut : float, optional (default=4500)
        Line-of-sight velocity clip in km/s used when constructing the phase-space
        and measuring the edge. Only galaxies with |v_los| < cut enter the edge
        finding step (guards against extreme outliers).
        
    bins : float, optional (default=5)
        The number of bins used the divide the phase-space edge. The default will be
        5 bins as this is what the Zv calibration is measured for. DO NOT MODIFY unless
        you have also re-measured the Zv calibration for your desired bin count.
        
    Returns
    -------
    results : Dictionary containing:
        - 'samples': MCMC samples
        - 'median': median of posterior
        - 'one_sig_up': Upper 1-sigma bound
        - 'one_sig_down': Lower 1-sigma bound
        - 'sampler': The emcee sampler object
    
    The additional displayed outputs are:
        - Printed MCMC mass estimate with uncertainties
        - Posterior plot of M200
        - Phase-space diagram showing galaxy data, escape velocity boundary,
          dynamical fit, and comparison with starting mass estimate
          
    Notes
    -----
    The analysis pipeline includes:
    2. Data preprocessing: coordinate conversion, centering, velocity cuts. 
    The centering algorithim undergoes 10 iterations. Convergence is defined such that the re-centered position of the last   iteration has moved by less than 1% of theta_200 from the prior iteration. 
    3. Interloper removal using shifting-gapper algorithm
    4. Escape velocity boundary identification from phase-space data
    5. MCMC mass estimation using escape velocity suppression
    6. Visualization of results in phase-space diagram
    
    The method assumes a Dehnen density profile fitted to match NFW with
    a mass-concentration relation. Suppression corrections are applied
    using the Zv calibration data generated via AGAMA, using a skewed-Normal Distribution to represent Zv, fitted via linear   
    relations in location, scale, and skewness, vs. N.
    
    Examples
    --------
    >>> cosmo_params = [0.3, 0.7]  # Omega_m, h for FlatLambdaCDM
    >>> cosmo_name = 'FlatLambdaCDM'
    >>> galaxy_data = np.loadtxt('galaxy_positions.txt')  # Shape (N, 3)
    >>> cluster_coords = (150.25, -10.75, 0.23)  # RA, Dec, z
    >>> M200_initial = 1e15  # Solar masses, guess for M200 measured via Weak lensing, SZ, X-ray, etc.
    >>> 
    >>> main('/path/to/Zv/calibration', galaxy_data, cluster_coords,
    ...      M200_initial, 0.1, 0.1, cosmo_params, cosmo_name)
    """
    
    cl_ra, cl_dec, cl_z = cluster_positional_data
    galaxy_r, galaxy_v, N, vesc_data_r, vesc_data_theta, vesc_data, vesc_data_err, cl_z=mass_estimation_preprocessing(cluster_positional_data,galaxy_positional_data,
                                                       M200, vesc_error,coremin_cut, cut, bins, cosmo_params,cosmo_name)

    escape_modeler = EscapeVelocityModeling(path_to_calibration=path_to_Zv_calibration)
    
    results = run_mcmc_mass_estimation(
        M200, cl_z, N, vesc_data_theta, vesc_data, vesc_data_err, bins,
        escape_modeler, cosmo_params, cosmo_name,
        nwalkers=nwalkers, nsteps=nsteps, n_processes=n_processes,
        mass_range_factor=1.5, progress=True
    )
    
    mass_estimation_post_processing(escape_modeler,results,M200,M200_err_up,
                                       M200_err_down,N,cl_z,vesc_data_r,vesc_data_theta,vesc_data,
                                            vesc_data_err,bins,galaxy_r, galaxy_v,cluster_name,cosmo_params,cosmo_name)
        
    return results

if __name__ == "__main__":
    main()
