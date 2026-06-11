#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
'''! Core definitions for Jamfit - filament reconstruction
@authors Jamie Xia
@date June 2026
'''

## IMPORTING EXTERNAL LIBRARIES ##
import numpy
import matplotlib.pyplot as plt
import pyvista as pv #depreciated 
import math
from matplotlib.path import Path

pv.set_jupyter_backend('static')  # Comment to enable interactive plots #depreciated

from ._core import ThinCurr, ThinCurr_reduced
from .._core import OFT_env
from .sensor import save_sensors
from ..io import histfile
from ..TokaMaker._core import TokaMaker
from ..TokaMaker.meshing import load_gs_mesh

# ===============================
# Jamfit Base Class
# ===============================

class Jamfit():
    '''! Main class for Jamfit filament reconstruction.
    Manages setup, synthetic data generation, reduced model creation,
    and eventual reconstruction of plasma filament currents using the
    ThinCurr framework.
    '''

    # =================================
    # Jamfit Creation Relevant Classes
    # =================================

    def __init__(self, xml_file, thincurr_meshfile, nthreads=None, oft_env=None):
        '''! Initialize the Jamfit object.

        @param xml_file str, path to the XML configuration file
        @param thincurr_meshfile str, path to the ThinCurr mesh file
        @param nthreads int, number of threads to use (required if oft_env is None)
        @param oft_env OFT_env, an existing OFT environment instance (optional)
        '''
        if oft_env is not None:
            self.myOFT = oft_env
        else:
            if nthreads is None:
                raise ValueError("Either oft_env or nthreads must be provided")
            self.myOFT = OFT_env(nthreads=nthreads)

        self.xml_file = xml_file
        self.thincurr_meshfile = thincurr_meshfile
        self.reduced_created_flag = False

    def set_xml(self, xml_file):
        '''! Update the XML configuration file path.

        @param xml_file str, path to the new XML configuration file
        '''
        self.xml_file = xml_file

    def set_sensors(self, sensor_array, floops_path='floops.loc'):
        '''! Save a sensor array to file and return the file path.

        @param sensor_array list, array of sensor objects (e.g. Mirnov, flux loops)
        @param floops_path str, output file path for sensor locations (default: 'floops.loc')
        @result str, path to the saved sensor file
        '''
        save_sensors(sensor_array, filename=floops_path)
        return floops_path
    
    def setup_jamfit(self, floops_path, plot_files = None, use_legacy_io=False, hodlr_path = 'full_HOLDR_M.save'):
        '''! Set up the ThinCurr model, I/O, and sensor/coil mutual inductance matrices.
        
        @param floops_path str, path to the sensor locations file
        '''
        self.torus = ThinCurr(self.myOFT)
        self.torus.setup_model(mesh_file=self.thincurr_meshfile, xml_filename=self.xml_file)
        if plot_files is not None:
            self.torus.setup_io(basepath=plot_files, legacy_hdf5=use_legacy_io)
        else:
            self.torus.setup_io() 
        self.Msensor, self.Msc, self.sensor_obj = self.torus.compute_Msensor(floops_path)
        self.torus.compute_Mcoil(cache_file=hodlr_path)
        self.torus.compute_Lmat(use_hodlr=True, cache_file=hodlr_path)
        self.torus.compute_Rmat(copy_out=True)
        print('Jamfit setup complete.')

    def setup_fil_timeseries(self, time_array, totalip, coil_currs, r_list, z_list, sigma_r, sigma_z, rgrid, zgrid):
        '''! Build a combined coil + plasma current array for a time-dependent run.
        Generates a Gaussian plasma current distribution at each time step and
        appends it to the coil currents for use with run_td.

        @param time_array numpy.ndarray, array of time points
        @param totalip numpy.ndarray, total plasma current at each time point
        @param coil_currs numpy.ndarray, coil currents with shape (ntimes, ncoils)
        @param r_list list, R position of plasma centroid at each time step
        @param z_list list, Z position of plasma centroid at each time step
        @param sigma_r float, Gaussian spread in the R direction
        @param sigma_z float, Gaussian spread in the Z direction
        @param rgrid numpy.ndarray, R coordinates of the filament grid
        @param zgrid numpy.ndarray, Z coordinates of the filament grid
        @result numpy.ndarray, combined time + coil + plasma current array for run_td
        '''
        coil_curr_wtime = numpy.hstack([time_array, coil_currs])
        plasma_curr_wtime = setup_synthetic_current(time_array, totalip, sigma_r, sigma_z, r_list, z_list, rgrid, zgrid)
        plasma_curr = plasma_curr_wtime[:, 1:]
        final_coil_currs = numpy.hstack((coil_curr_wtime, plasma_curr))
        return final_coil_currs

    def gen_synthetic_data(self, coil_currs, dt, nsteps, verbose = False, hodlr_path = 'full_HOLDR_L.save',s_freq = 10, p_freq=10):
        '''! Run a synthetic time-dependent simulation and compute sensor signals.
        Computes the inductance and resistance matrices, runs the time-dependent
        simulation, plots results, and saves the B matrix and sensor history.

        @param coil_currs numpy.ndarray, combined coil + plasma current array (with time column)
        @param dt float, time step size in seconds
        @param nsteps int, number of time steps
        '''
        self.torus.run_td(dt, nsteps, coil_currs=coil_currs, sensor_obj=self.sensor_obj, status_freq= s_freq, plot_freq=p_freq)
        self.torus.plot_td(nsteps, sensor_obj=self.sensor_obj)
        hist_file = histfile('floops.hist')
        if verbose: 
            plot_data = self.torus.build_XDMF()
            return hist_file, plot_data
        else: 
            return hist_file 

    def create_from_runTD_top_modes(self, num_modes, reduced_filename, coil_currs, dt, nsteps, initial_num_eigs=50, verbose=False, s_freq = 10, p_freq = 10 ):
        '''! Build a reduced model using the dominant eigenmodes from a full run.
        Computes eigenvalues, runs a preliminary reduced model to identify the
        most active modes by current amplitude, then builds a final reduced model
        using only those dominant modes.

        @param num_modes int, number of dominant modes to retain in the reduced model
        @param reduced_filename str, output filename for the reduced model (HDF5)
        @param coil_currs numpy.ndarray, combined coil + plasma current array (with time column)
        @param nsteps int, number of time steps for the preliminary run
        @param dt float, time step size in seconds
        @param num_eigs int, number of eigenmodes to compute initially (default: 50)
        @param verbose bool, if True, plots mode amplitudes and prints mode info (default: False)
        @result ThinCurr_reduced, the constructed reduced model object
        '''
        self.eig_vals, self.eig_vecs = self.torus.get_eigs(initial_num_eigs, False)
        torus_first_reduced = self.torus.build_reduced_model(
            self.eig_vecs, filename='first_reduced_model_temp.h5', sensor_obj=self.sensor_obj
        )
        sensors_measurement, currents = torus_first_reduced.run_td(dt, nsteps, coil_currs, status_freq=s_freq, plot_freq=p_freq)

        temp_curr = currents['curr']
        temp_curr = temp_curr[:, 0:initial_num_eigs]  # Only consider the modes we computed
        max_weights = [abs(temp_curr[:, i]).sum() for i in range(temp_curr.shape[1])] # total sum over time 

        top_modenum_indices = sorted(range(len(max_weights)), key=lambda i: max_weights[i], reverse=True)[:num_modes]

        eig_inds = []
        weight_amplitude = []

        if verbose:
            fig, ax = plt.subplots(1, 1)
            self.torus.build_XDMF()

        for i in range(temp_curr.shape[1]):
            if i in top_modenum_indices:
                eig_inds.append(i)
                weight_amplitude.append(max_weights[i])
                if verbose:
                    ax.semilogy(currents['time'], abs(currents['curr'][:, i]), label=f'Mode {i}')
                    print(f'Saved mode {i} has max weight {max_weights[i]}')
            else:
                if verbose:
                    ax.semilogy(currents['time'], abs(currents['curr'][:, i]), color='gray', alpha=0.3)
        print("eig_inds:", eig_inds)
        print("eig_vecs shape:", self.eig_vecs.shape)
        print("eig_vecs[eig_inds] shape:", self.eig_vecs[eig_inds, :].shape)
        print("first row of selected eig_vecs:", self.eig_vecs[eig_inds[0], :5])  # first 5 elements
        self.reduced_torus = self.create_reduced_model(self.eig_vecs[eig_inds, :], reduced_filename, compute_B=False)
        self.reduced_created_flag = True
        if verbose:
             return self.reduced_torus, sensors_measurement, currents, self.eig_vecs[eig_inds, :], eig_inds, weight_amplitude
        else:
            return self.reduced_torus, sensors_measurement, currents
    
    
    def create_reduced_model(self, eig_vecs, reduced_filename, compute_B=False):
        '''! Build a reduced model using specified eigenvectors.
        
        @param eig_vecs numpy.ndarray, array of eigenvectors to use for the reduced model
        @param reduced_filename str, output filename for the reduced model (HDF5)
        @param compute_B bool, if True, computes the B matrix for the reduced model (default: False)
        @result ThinCurr_reduced, the constructed reduced model object
        '''
        self.reduced_torus = self.torus.build_reduced_model(
            eig_vecs, filename=reduced_filename, compute_B=compute_B, sensor_obj=self.sensor_obj
        )
        print(f"Reduced model created with {eig_vecs.shape[0]} modes")
        self.reduced_created_flag = True
        return self.reduced_torus
    

    def add_freq_eigenvalues(self, specific_fil_array): 
        '''! Augment the eigenvector basis with frequency-response vectors for specific filaments.
        For each filament index provided, computes the steady-state frequency
        response driven by that filament's mutual inductance and appends the
        result to the stored eigenvector matrix.

        @param specific_fil_array list, indices of filaments to compute frequency responses for
        @result str, confirmation message on completion
        '''
        from IPython.display import clear_output
        if self.eig_vecs is None:
            raise ValueError("Eigenvalues and eigenvectors have not been computed yet.")

        eig_vecs_wfreq = numpy.copy(self.eig_vecs)
        for target_fil in specific_fil_array:
            Mcoil = self.torus.compute_Mcoil()
            driver = numpy.zeros((2, self.torus.nelems))
            driver[0, :] = Mcoil[target_fil, :]
            result = self.torus.compute_freq_response(driver, freq=1.E3)
            eig_vecs_wfreq = numpy.concatenate((eig_vecs_wfreq, result[:1, :]), axis=0)
            clear_output(wait=True)

        self.eig_vecs = eig_vecs_wfreq
        return eig_vecs_wfreq

    # =================================
    # Reconstruction Relevant Classes
    # =================================

    def initialize_reduced_model(self, reduced_filename):
        '''! Load a previously saved reduced model from file.
        
        @param reduced_filename str, path to the HDF5 reduced model file
        @result str, confirmation message on successful load
        '''
        self.torus_reduced = ThinCurr_reduced(reduced_filename)
        self.reduced_created_flag = True
        return "Reduced model initialized from file."

    def run_reconstruction_lstsq(self, Psi_at_time, ip_at_time, num_non_fil_coils, coil_curr_at_time, ip_weight, sigma, reg_factor_fil, reg_factor_wall, num_sensors = None):
        '''! Run the filament current reconstruction with the lstsq method.
        
        @param Psi_at_time numpy.ndarray, sensor flux measurements at time
        @param ip_at_time float, total plasma current measurement at time
        @param num_non_fil_coils int, number of non-filament coils in the system
        @param ip_weight float, weight for the total plasma current in the reconstruction
        @param sigma numpy.ndarray, array of standard deviations for each sensor measurement (for weighting)
        @param reg_factor_fil float, regularization factor for filament currents
        @param reg_factor_wall float, regularization factor for wall currents
        @result tuple, containing the solution, residual, Ax, and B
        '''
        if not self.reduced_created_flag:
            raise ValueError("Reduced model has not been created yet. Please create or initialize a reduced model before running reconstruction.")
        if num_sensors is None: 
            num_sensors = self.torus_reduced.Ms.shape[1] 
        # intializing the Ms and Msc matrices with the appropriate weighting and scaling based off sigma
        Ms_weighted = self.torus_reduced.Ms[:,:num_sensors]/sigma[:]
        Msc_weighted = self.torus_reduced.Msc[:, :num_sensors]/sigma[:]
        Msc_weighted_fil = Msc_weighted[num_non_fil_coils:, :]

        # This section of code scales the total ip constraint row of the matrix to ensure it has a comparable influence on the 
        # least squares solution as the magnetic measurements, based on the provided ip_weight and the magnitude of ip_at_time
        ip_row_scale = 1.0 / abs(ip_at_time)
        ip_col_ms = numpy.zeros((Ms_weighted.shape[0], 1))
        ip_col_msc = numpy.ones((Msc_weighted_fil.shape[0], 1)) * (ip_weight * ip_row_scale)

        # appending the ip constraint as an additional row to the Ms and Msc matrices, with appropriate scaling
        Ms_final = numpy.append(Ms_weighted, ip_col_ms, axis=1)
        Msc_final = numpy.append(Msc_weighted_fil, ip_col_msc, axis=1)
        combined_matrix_A = numpy.vstack((Ms_final, Msc_final)).T

        # here we prepare the the B vector by subtacting the non plasma filament contribution from the sensor measurements
        # we also scale by sigma here as well (to ensure magnetic sensor signals are normalized to each other - one sensor doesnt dominate)
        # finally we append the ip cosntraint row as well 
        B_weighted = (Psi_at_time - coil_curr_at_time @ self.torus_reduced.Msc[:num_non_fil_coils, :num_sensors]) / sigma[:] 
        B_weighted = numpy.append(B_weighted, [ip_at_time * ip_weight * ip_row_scale])

        # here we apply tikonov regularization to both the filament and wall component of the lstq 
        # note that we must use the unmodified shapes of Ms to construct the identity matrices for regularization
        # the first half of the identity matrix corresponds to the wall currents and the second half corresponds to the filament currents, so we slice accordingly when stacking them below
        # we then add the reg rows to B as well to match the extra rows that we stacked to A 
        num_to_solve = Ms_final.shape[0] + Msc_final.shape[0]
        reg_identity = numpy.eye(num_to_solve)
        A = numpy.vstack([combined_matrix_A, reg_factor_wall * reg_identity[0:Ms_weighted.shape[0], :], reg_factor_fil * reg_identity[Ms_weighted.shape[0]:, :]])
        B= numpy.concatenate([B_weighted, numpy.zeros(A.shape[1])])

        # solving the least squares problem
        AtA = A.T @ A
        AtB = A.T @ B
        solution = numpy.linalg.solve(AtA, AtB)
        Ax = numpy.dot(A, solution)
        residual = numpy.sqrt(numpy.sum((B - Ax)**2))

        return solution, residual, Ax, B
    
    
    def prepare_tsvd_laplace(self, sigma, num_non_fil_coils, rgrid, zgrid, nModes, verbose = False):
        '''! Prepare matrices and projections for the svd + laplacian reconstruction method.
        
        @param sigma numpy.ndarray, array of standard deviations for each sensor measurement (for weighting)
        @param num_non_fil_coils int, number of non-filament coils in the system
        @param rgrid numpy.ndarray, R coordinates of the filament grid
        @param zgrid numpy.ndarray, Z coordinates of the filament grid
        @param nModes int, number of SVD modes to truncate to
        @param verbose bool, if True, plots singular values (default: False)
        ''' 

        # getting laplacina matrix for smoothing purposes
        lap_mat, N = get_laplace_matrix(rgrid, zgrid, verbose)
        num_Ms = self.torus_reduced.Ms.shape[0]

        # intializing the Ms and Msc matrices with the appropriate weighting and scaling based off sigma
        Msc_fil_weighted =  self.torus_reduced.Msc[num_non_fil_coils:, :]/sigma[:]
        Msc_coils_weighted = self.torus_reduced.Msc[:num_non_fil_coils, :]/sigma[:]
        Ms_weighted = self.torus_reduced.Ms/sigma[:]

        # break the problem into just the filaments and find svd modes (we do not solve for the shaping coil currents during the reconstruction)
        U, S, Vh = numpy.linalg.svd(Msc_fil_weighted, full_matrices=False)

        # intializing the least squares matrix for the truncated SVD solution 
        ls_mat_fil = Msc_fil_weighted.T @ U[:, :nModes]
        ls_mat_wall = Ms_weighted.T 
        ls_mat = numpy.hstack([ls_mat_wall, ls_mat_fil])

        # projecting the laplacian onto the TSVD space to get a reg term that smooths in the physical filament space
        lap_proj = lap_mat @ U[:, :nModes]

        # getting the ip constraint row in the TSVD space, since U is already normalized to the sensor signals, we only need to normalize the ip row to itself
        ip_row_fil = U[:, :nModes].sum(axis=0, keepdims=True)
        ip_row_fil_norm = numpy.linalg.norm(ip_row_fil)
        ip_row = numpy.hstack([numpy.zeros((1, num_Ms)), ip_row_fil/ip_row_fil_norm])


        if verbose: 
            plt.figure(figsize=(8, 5))
            plt.semilogy(S/S[0], marker='o')
            plt.title('Normalized Singular Values of plasma filament contribution to sensor signals')
            plt.xlabel('Mode Index')
            plt.ylabel('Singular Value (log scale)')
            plt.grid(True)
            plt.show()
        return Msc_coils_weighted, Ms_weighted, U[:, :nModes], ls_mat, ls_mat_fil, lap_proj, ip_row, N
    

    
    def run_reconstruction_tsvd_laplace(self, Psi_at_time, ip_at_time, coil_curr_at_time, Msc_coils, Ms, U_trun, ls_mat, ls_mat_fil, lap_proj, ip_row, N, lam=None, lap_lam=1e-8, reg_wall=1e-5):
        '''! Run the filament current reconstruction with the svd + laplacian method.
        
        @param Psi_at_time numpy.ndarray, sensor flux measurements at time
        @param ip_at_time numpy.ndarray, plasma current measurements at time
        @param coil_curr_at_time numpy.ndarray, coil currents at time
        @param Msc_coils numpy.ndarray, coil matrix
        @param Ms numpy.ndarray, wall matrix
        @param U_trun numpy.ndarray, truncated singular vectors
        @param ls_mat numpy.ndarray, least squares matrix
        @param ls_mat_fil numpy.ndarray, filament least squares matrix
        @param lap_proj numpy.ndarray, laplacian projection matrix
        @param ip_row numpy.ndarray, ip constraint row, normalized to both sigma and itself
        @param N int, number of filaments
        @param lam float, ip weight (usually calculated automatically but can be set manually)
        @param lap_lam float, laplacian regularization parameter
        @param reg_wall float, wall regularization parameter
        '''
        num_Ms = Ms.shape[0]

        # taking out coil contributions from the magnetic sensor signals 
        B = Psi_at_time - coil_curr_at_time @ Msc_coils

        if lam is None: 
            # calculating the weight of the ip constraint row based on the magnitudal difference between magnetic sensor signals and the total plasma current
            # ensures that they are on the same order of magnitude for the least squares solution 
            # Compare typical sensor signal magnitude to IP magnitude
            magnitude_diff = math.floor(math.log10(numpy.mean(numpy.abs(B)) / (abs(ip_at_time) + 1e-30)))
            lam = 100 * 10**magnitude_diff # note that I multiply by 100 to give the ip constraint slighly more weight as the magnetic sensor signals have more rows over the singular total plasma current row
        
        ip_scale = numpy.linalg.norm(U_trun.sum(axis=0)) # this is for scaling the ip constraint row to the svd space for the totalip on the B side of Ax=B
        
        # constructing the A matrix by stacking the svd matrix with the regularization rows  
        reg_mat = numpy.vstack([
            ls_mat,                                    # measurements
            lam * ip_row,                                                        # Ip constraint 
            numpy.hstack([reg_wall * numpy.eye(num_Ms), numpy.zeros((num_Ms, ls_mat_fil.shape[1]))]),      # wall Tikhonov
            numpy.hstack([numpy.zeros((N, num_Ms)), lap_lam * lap_proj])                   # fil Laplacian
        ])

        # constructing the B vector by stacking the measurement vector with the ip constraint and zeros for the regularization rows
        psi_reg = numpy.concatenate([
            B,
            numpy.array([lam * ip_at_time / ip_scale]),
            numpy.zeros(num_Ms),
            numpy.zeros(N)
        ])

        # solving using least squares explicitly of Ax=B
        AtA = reg_mat.T @ reg_mat
        AtB = reg_mat.T @ psi_reg
        curr_weights = numpy.linalg.solve(AtA, AtB)

        # projecting the solution back to the physical space 
        curr_expand = U_trun @ curr_weights[num_Ms:]  # only the filament part contributes to the expansion
        wall_expand = curr_weights[:num_Ms]  # wall coefficients


        # calculating diagnostics
        Ax          = ls_mat @ curr_weights  
        ip_reconstructed = curr_expand.sum()
        ip_actual        = ip_at_time
        ip_error_pct     = 100 * numpy.abs(ip_reconstructed - ip_actual) / (abs(ip_actual) + 1e-30)
        fit_residual     = numpy.linalg.norm(Ax - B)
        fit_residual_nonorm = Ax - B

        diagnostics = {
            'lam':              lam,
            'lap_lam':          lap_lam,
            'ip_reconstructed': ip_reconstructed,
            'ip_actual':        ip_actual,
            'ip_error_pct':     ip_error_pct,
            'fit_residual':     fit_residual,
            'curr_weights':     curr_weights,
            'fit_residual_nonorm': fit_residual_nonorm,
            'ip_row':           ip_row,
        }

        return curr_expand, wall_expand, Ax, diagnostics

    # ======================================
    # Post Processesing and Visualization
    # ======================================

    def get_wall_psi_tidx(self, num_sensors, solution_wall_tidx): 
        wall_psi_probes = self.torus_reduced.Ms[:, num_sensors](2*numpy.pi)
        wall_psi_tidx = solution_wall_tidx @ wall_psi_probes
        return wall_psi_tidx 
    
    def post_process_tidx(self, filaments_at_time, coil_curr_dict, rgrid, zgrid, meshfile_tokamaker, wall_psi, B0, R0, myOFT, verbose = False):
        '''! Post-process the results at a given time index to compute plasma parameters and visualize.
        
        @param filaments_at_time numpy.ndarray, filament currents at the given time index
        @param coil_curr_dict dict, dictionary of coil currents at the given time index
        @param rgrid numpy.ndarray, R coordinates of the filament mesh
        @param zgrid numpy.ndarray, Z coordinates of the filament mesh
        @param meshfile_tokamaker str, path to the Tokamaker mesh file for equilibrium reconstruction
        @param wall_psi numpy.ndarray, precomputed wall contribution to the flux
        @param B0 float, reference magnetic field strength for equilibrium reconstruction
        @param R0 float, reference major radius for equilibrium reconstruction
        @param myOFT OFT_env, the Open FUSION Toolkit environment instance
        @param verbose bool, if True, plots the equilibrium and LCFS (default: False)'''

        #intialize tokamaker 
        mygs = TokaMaker(myOFT)
        mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(meshfile_tokamaker)
        mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
        mygs.setup(order=2, F0= B0 * R0)# F0 = B0 * R0 
        limiter = mygs.lim_contour

        # calculate psi from plasmas
        fil_points = list(zip(rgrid, zgrid))
        psi_fil = []
        for filcount, (r, z) in enumerate(fil_points):
            mygs.set_coil_currents()
            if filaments_at_time[filcount] > 0:
                mygs.set_targets(Ip=filaments_at_time[filcount])
                mygs.init_psi(r, z, 0.3, 1.0, 0.0) 
                psi_fil.append(mygs.get_psi(False))
        psi_fil = numpy.array(psi_fil)
        psi_total_fil = numpy.sum(psi_fil, axis=0)
        ip = numpy.sum(filaments_at_time)
        
        # Calculate psi from coils 
        mygs.set_coil_currents(coil_curr_dict)
        psi_vf = mygs.vac_solve()

        # Summate Psis 
        total_psi = psi_total_fil + psi_vf + wall_psi
        mygs.set_psi(total_psi, update_bounds = True)

        # getting relevant values 
        lcfs_points = mygs.trace_surf(1) # Trace LCFS
        if lcfs_points is not None:
            psiatlcfs = mygs.psinorm_to_absolute(1)

        if lcfs_points is None: 
            lcfs_points = mygs.trace_surf(0.99)
            psiatlcfs = mygs.psinorm_to_absolute(0.99)

        limiting_pts = [] 
        q_vals = None
        q95 = None 
        internal_inductance = None
        current_cent = calc_current_centroid(rgrid, zgrid, ip)
        area_cent = None
        area = None
     
        if lcfs_points is not None:
            lim_R, lim_Z, _ = find_limiting_point(lcfs_points, limiter)
            limiting_pts.append((lim_R, lim_Z))
            _, q_vals, _, _, _, _= mygs.get_q() 
            internal_inductance = mygs.get_stats(beta_Ip = ip)['l_i']
            _, q95, _ , _, _, _ = mygs.get_q(psi_norm=0.95)
            area, area_cent = calc_lcfs_geo(lcfs_points) 
        else: 
            psiatlcfs = None 
            limiting_pts.append(None)

        if verbose: 
            fig, ax = plt.subplots(1,1) 
            mygs.plot_machine(fig,ax, cond_color='blue') 
            mygs.plot_psi(fig,ax ,plasma_nlevels = 75, normalized=False)
            plt.gca().set_aspect('equal', adjustable='box')
            if lcfs_points is not None:
                ax.plot(lcfs_points[:,0], lcfs_points[:,1], 'r--', label='LCFS')
    
        return lcfs_points, limiting_pts, psiatlcfs, total_psi, q_vals, q95, internal_inductance, current_cent, area_cent, area
    # ========================================================
    # Jamfit Depreciated Functions to be worked on or removed 
    # ========================================================

    def plot_sensors(self, sensor_points_mirnov_array, sensor_points_flux, orientations):
        '''! Visualize sensor positions on the ThinCurr mesh using PyVista.
        Renders Mirnov probes as oriented arrows and flux loops as spheres
        overlaid on the wall mesh.

        @param sensor_points_mirnov_array numpy.ndarray, shape (N, 3) array of Mirnov probe positions
        @param sensor_points_flux dict, dict of flux loop data with keys 'x', 'y', 'z' per loop name
        @param orientations numpy.ndarray, shape (N, 3) array of normal vectors for each Mirnov probe
        '''
        plot_data = self.torus.build_XDMF()
        grid = plot_data['ThinCurr']['smesh'].get_pyvista_grid()

        p = pv.Plotter()
        p.camera.up = [0, 0, -100]
        p.camera_position = [(9, 6, 2), (0, 0, 0), (0, 0, 1)]
        p.add_mesh(grid, color="white", opacity=0.75, show_edges=False)

        # Mirnov sensor arrows
        sensor_points_mirnov = pv.PolyData(sensor_points_mirnov_array)
        sensor_points_mirnov.point_data['vectors'] = orientations
        arrow = pv.Arrow(tip_length=0.5, tip_radius=0.2, shaft_radius=0.05)
        arrows = sensor_points_mirnov.glyph(orient='vectors', scale='vectors', factor=0.5, geom=arrow)
        p.add_mesh(arrows, color='blue', show_scalar_bar=False)

        # Flux loop spheres
        for name, data in sensor_points_flux.items():
            positions = numpy.column_stack([data['x'], data['y'], data['z']])
            flux_polydata = pv.PolyData(positions)
            spheres_flux = flux_polydata.glyph(geom=pv.Sphere(radius=0.05))
            p.add_mesh(spheres_flux, color='red')

        p.add_axes(interactive=False)
        p.show()

    def plot_wall_currents(self, time_index):
        '''! Visualize wall current vectors on the mesh at a given time index.
        Renders the current density field J as scaled arrows on the wall mesh,
        colored by magnitude.

        @param time_index int, index into the simulation time array to plot
        '''
        plot_data = self.torus.build_XDMF()
        plot_times = plot_data['ThinCurr']['smesh'].times
        grid = plot_data['ThinCurr']['smesh'].get_pyvista_grid()
        Jfull = plot_data['ThinCurr']['smesh'].get_field('J_v', plot_times[time_index])

        grid["vectors"] = Jfull
        grid.set_active_vectors("vectors")

        p = pv.Plotter()
        scale = 1 / (numpy.linalg.norm(Jfull, axis=1)).max()
        arrows = grid.glyph(scale="vectors", orient="vectors", factor=scale)
        p.add_mesh(grid, color="white", opacity=0.75, show_edges=True)
        p.add_mesh(arrows, cmap="turbo", scalar_bar_args={
            'title': "|J|", "vertical": True, "position_y": 0.25, "position_x": 0.0
        })
        p.show()

# ===============================
# Jamfit Helper Functions
# ===============================

def setup_synthetic_current(timepoints, ip_list, sigma_r, sigma_z, r0, z0, rgrid, zgrid):
    '''! Generate synthetic filament currents using a Gaussian plasma distribution.
    At each time step, spreads the total plasma current across the filament grid
    using a 2D Gaussian centered at (r0, z0) with widths (sigma_r, sigma_z).

    @param timepoints list or numpy.ndarray, array of time values
    @param ip_list list or numpy.ndarray, total plasma current at each time step
    @param sigma_r float, Gaussian width in the R direction
    @param sigma_z float, Gaussian width in the Z direction
    @param r0 list or numpy.ndarray, R position of the plasma centroid at each time step
    @param z0 list or numpy.ndarray, Z position of the plasma centroid at each time step
    @param rgrid numpy.ndarray, R coordinates of the filament mesh
    @param zgrid numpy.ndarray, Z coordinates of the filament mesh
    @result numpy.ndarray, shape (ntimes, 1 + nfilaments), time column followed by filament currents
    '''
    coil_curr = []
    for i in range(len(timepoints)):
        gaussian_raw = numpy.exp(
            -((rgrid - r0[i])**2 / (2 * sigma_r**2) + (zgrid - z0[i])**2 / (2 * sigma_z**2))
        )
        gaussian_values = ip_list[i] * (gaussian_raw / numpy.sum(gaussian_raw))
        coil_curr.append(gaussian_values)

    coil_curr = numpy.array(coil_curr)
    time_column = numpy.array(timepoints).reshape(-1, 1)
    coil_curr = numpy.hstack((time_column, coil_curr))
    return coil_curr


def interpolate_total_current(coil_currs, nsteps, verbose=False):
    '''! Interpolate total plasma current to a higher-resolution time grid.
    Sums the sensor currents at each time step and interpolates the total
    onto a finer time grid using linear interpolation.

    @param coil_currs numpy.ndarray, shape (ntimes, nsensors+1), first column is time
    @param nsteps int, number of high-resolution steps between first and last time
    @param verbose bool, if True, plots the original and interpolated total current (default: False)
    @result tuple of (high_res_time, total_current_high_res) as numpy.ndarrays
    '''
    times = coil_currs[:, 0]
    sensor_currents = coil_currs[:, 1:]

    high_res_time = numpy.linspace(times[0], times[-1], nsteps + 1)

    interpolated_currents = numpy.array([
        numpy.interp(high_res_time, times, sensor)
        for sensor in sensor_currents.T
    ]).T  # shape: (nsteps+1, nsensors)

    total_current_high_res = numpy.sum(interpolated_currents, axis=1)

    if verbose:
        plt.figure(figsize=(8, 5))
        plt.scatter(times, numpy.sum(sensor_currents, axis=1), color='red', label='Original Data')
        plt.plot(high_res_time, total_current_high_res, color='blue', label='Interpolated Total Current')
        plt.xlabel('Time [s]')
        plt.ylabel('Total Current [A]')
        plt.title('Total Current vs Time with Higher Resolution')
        plt.legend()
        plt.grid(True)
        plt.show()

    return high_res_time, total_current_high_res

def get_laplace_matrix(rgrid, zgrid, verbose =False):
    '''! Construct a Laplacian matrix for the 2D filament grid,
    
    @param rgrid numpy.ndarray, R coordinates of the filament grid
    @param zgrid numpy.ndarray, Z coordinates of the filament grid
    @return lap_mat numpy.ndarray, the constructed Laplacian matrix for the filament grid
    @ return N int, the number of points in the filament grid
    '''
    points = numpy.array(list(zip(rgrid, zgrid)))
    R_vals = rgrid 
    Z_vals = zgrid 
    r_unique = numpy.unique(R_vals)                      
    z_unique = numpy.unique(Z_vals)
    dr = numpy.min(numpy.diff(numpy.sort(r_unique)))
    dz = numpy.min(numpy.diff(numpy.sort(z_unique)))
    if verbose:
        print("dr =", dr)
        print("dz =", dz)
        print("nr =", len(r_unique))
        print("nz =", len(z_unique))
    N = len(points)
    ## Getting W matrix
    W = numpy.zeros((N, N))
    tol = 1e-6  # floating tolerance
    for i in range(N):
        Ri, Zi = points[i]
        for j in range(N):
            if i == j:
                continue
            Rj, Zj = points[j]
            dR = Ri - Rj
            dZ = Zi - Zj
            # Cardinal R-neighbors (same Z, one step in R)
            if abs(abs(dR) - dr) < tol and abs(dZ) < tol:
                W[i, j] = (2/3) / dr**2
            # Cardinal Z-neighbors (same R, one step in Z)
            elif abs(dR) < tol and abs(abs(dZ) - dz) < tol:
                W[i, j] = (2/3) / dz**2
            # Diagonal neighbors (one step in both R and Z)
            elif abs(abs(dR) - dr) < tol and abs(abs(dZ) - dz) < tol:
                 W[i, j] = (1/6) / (dr**2 + dz**2)
    ## Getting D matrix 
    D = numpy.zeros((N, N))
    for i in range(N):
        D[i, i] = numpy.sum(W[i, :])
    ## Getting Laplacian matrix
    lap_mat = D - W
    return lap_mat, N

def find_limiting_point(lcfs_points, limiter, touch_tol=0.005):
    
    dists = numpy.min(
        numpy.linalg.norm(lcfs_points[:, None, :] - limiter[None, :, :], axis=2),
        axis=1
    )
    
    idx = numpy.argmin(dists)
    min_dist = dists[idx]
    
    if min_dist > touch_tol:
        return None, None, None
    
    return lcfs_points[idx, 0], lcfs_points[idx, 1], min_dist

    
def calc_lcfs_geo(lcfs_points):
    """
    Calculate area and geometric centroid of the LCFS.

    Parameters:
        lcfs_points: array-like of shape (2, N)
                     lcfs_points[0] = R coordinates
                     lcfs_points[1] = Z coordinates
    Returns:
        area     (float): Enclosed area in m²
        R_c      (float): Centroid R coordinate in m
        Z_c      (float): Centroid Z coordinate in m
    """
    if lcfs_points is None:
        return None, None, None

    R = numpy.array(lcfs_points[:, 0])
    Z = numpy.array(lcfs_points[:, 1])

    # Ensure contour is closed
    if not (numpy.isclose(R[0], R[-1]) and numpy.isclose(Z[0], Z[-1])):
        R = numpy.append(R, R[0])
        Z = numpy.append(Z, Z[0])

    cross = R[:-1] * Z[1:] - R[1:] * Z[:-1]    # Calculates area using Shoelace method, Shoelace terms

    # Area
    area = 0.5 * numpy.abs(numpy.sum(cross))

    # Centroid
    R_c = numpy.abs(numpy.sum((R[:-1] + R[1:]) * cross)) / (6 * area)
    Z_c = numpy.abs(numpy.sum((Z[:-1] + Z[1:]) * cross)) / (6 * area)

    return area, (R_c, Z_c) 

def calc_current_centroid(R_fil, Z_fil, I):
    '''! Calculate the current centroid of a filament grid using signed current weights
        This variant uses signed currents as weights, so opposing currents can partially
        cancel out the centroid position. Useful for computing the net moment of the
        current distribution.
        
        Calculation: R_c = sum(I_i * R_i) / sum(I_i)
                    Z_c = sum(I_i * Z_i) / sum(I_i)
        
        @param R 1D or 2D array of R (radial) coordinates of filament points `[m]`
        @param Z 1D or 2D array of Z (vertical) coordinates of filament points `[m]`
        @param I 1D or 2D array of currents at each filament point `[A]`, must match shape of R and Z
        @result R coordinate of the current centroid `[m]`
        @result Z coordinate of the current centroid `[m]`
        @result Net current (algebraic sum) `[A]`
    '''
    R = numpy.asarray(R_fil)  # Ensure inputs are numpy arrays
    Z = numpy.asarray(Z_fil)
    I = numpy.asarray(I)
    
    # Verify shapes match
    if not (R.shape == Z.shape == I.shape):
        raise ValueError(
            f"R, Z, and I must have the same shape. "
            f"Got R: {R.shape}, Z: {Z.shape}, I: {I.shape}"
        )
    
    R_flat = R.ravel()
    Z_flat = Z.ravel()
    I_flat = I.ravel()
    
    I_net = numpy.sum(I_flat)
    if I_net == 0:
        raise ValueError("Net current is zero - cannot calculate centroid")
    
    R_centroid = numpy.sum(I_flat * R_flat) / I_net
    Z_centroid = numpy.sum(I_flat * Z_flat) / I_net

    return (R_centroid, Z_centroid)


def get_inside_limiter_pts(meshfile_tokamaker, myOFT, verbose = False): 
    mygs = TokaMaker(myOFT)
    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(meshfile_tokamaker)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.setup(order=2, F0= 1 * 1)
    r_pts_grid = mygs.r[:,0]
    z_pts_grid = mygs.r[:,1]
    r_pts_lim = mygs.lim_contour[:,0]
    z_pts_lim = mygs.lim_contour[:,1]
    lim_path = Path(numpy.column_stack([r_pts_lim, z_pts_lim]))
    all_pts = numpy.column_stack([r_pts_grid, z_pts_grid])
    inside_mask = lim_path.contains_points(all_pts) #boolean grid where 1 means inside limiter and 0 means outside
    used_pts = numpy.where(inside_mask)[0] #returns indicies of points that are inside limiter
    inside_lim_pts = numpy.column_stack((mygs.r[used_pts, 0], mygs.r[used_pts, 1])) #grabs pts of the grid that are inside the limiter

    if verbose:
        fig, ax = plt.subplots()
        ax.scatter(r_pts_grid, z_pts_grid, c=inside_mask, cmap='RdYlGn', s=10, alpha=0.7)
        ax.plot(r_pts_lim, z_pts_lim, 'k-', label='Limiter Contour')
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        ax.set_title('Grid Points Inside Limiter')
        ax.legend()
        plt.colorbar(ax.collections[0], ax=ax, label='Inside Limiter')
        plt.tight_layout()
        plt.show()

    return inside_lim_pts, inside_mask, mygs.r







