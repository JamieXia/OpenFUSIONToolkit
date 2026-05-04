#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
'''! Core definitions for Jamfit - filament reconstruction
@authors Jamie Xia
@date April 2026
'''

## IMPORTING EXTERNAL LIBRARIES ##
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

pv.set_jupyter_backend('static')  # Comment to enable interactive plots

from ._core import ThinCurr, ThinCurr_reduced
from .._core import OFT_env
from .sensor import Mirnov, save_sensors, circular_flux_loop
from ..util import mu0
from ..io import histfile


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

        @param time_array np.ndarray, array of time points
        @param totalip np.ndarray, total plasma current at each time point
        @param coil_currs np.ndarray, coil currents with shape (ntimes, ncoils)
        @param r_list list, R position of plasma centroid at each time step
        @param z_list list, Z position of plasma centroid at each time step
        @param sigma_r float, Gaussian spread in the R direction
        @param sigma_z float, Gaussian spread in the Z direction
        @param rgrid np.ndarray, R coordinates of the filament grid
        @param zgrid np.ndarray, Z coordinates of the filament grid
        @result np.ndarray, combined time + coil + plasma current array for run_td
        '''
        coil_curr_wtime = np.hstack([time_array, coil_currs])
        plasma_curr_wtime = setup_synthetic_current(time_array, totalip, sigma_r, sigma_z, r_list, z_list, rgrid, zgrid)
        plasma_curr = plasma_curr_wtime[:, 1:]
        final_coil_currs = np.hstack((coil_curr_wtime, plasma_curr))
        return final_coil_currs

    def gen_synthetic_data(self, coil_currs, dt, nsteps, verbose = False, hodlr_path = 'full_HOLDR_L.save',s_freq = 10, p_freq=10):
        '''! Run a synthetic time-dependent simulation and compute sensor signals.
        Computes the inductance and resistance matrices, runs the time-dependent
        simulation, plots results, and saves the B matrix and sensor history.

        @param coil_currs np.ndarray, combined coil + plasma current array (with time column)
        @param dt float, time step size in seconds
        @param nsteps int, number of time steps
        '''
        self.torus.run_td(dt, nsteps, coil_currs=coil_currs, sensor_obj=self.sensor_obj, status_freq= s_freq, plot_freq=p_freq)
        self.torus.plot_td(nsteps, sensor_obj=self.sensor_obj)
        hist_file = histfile('floops.hist') ## have it return the hist_file object. 
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
        @param coil_currs np.ndarray, combined coil + plasma current array (with time column)
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
           # self.torus.build_XDMF()


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

        @param eig_vecs np.ndarray, array of eigenvectors to use for the reduced model
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

        eig_vecs_wfreq = np.copy(self.eig_vecs)
        for target_fil in specific_fil_array:
            Mcoil = self.torus.compute_Mcoil()
            driver = np.zeros((2, self.torus.nelems))
            driver[0, :] = Mcoil[target_fil, :]
            result = self.torus.compute_freq_response(driver, freq=1.E3)
            eig_vecs_wfreq = np.concatenate((eig_vecs_wfreq, result[:1, :]), axis=0)
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

    def run_reconstruction_lstsq(self, Psi_at_time, ip_at_time, num_non_fil_coils, coil_curr_at_time, ip_weight, sigma, reg_factor_fil, reg_factor_wall):
        '''! Run the filament current reconstruction with the lstsq method.
        @param Psi_at_time np.ndarray, sensor flux measurements at time
        @param ip_at_time float, total plasma current measurement at time
        @param num_non_fil_coils int, number of non-filament coils in the system
        @param ip_weight float, weight for the total plasma current in the reconstruction
        @param sigma np.ndarray, array of standard deviations for each sensor measurement (for weighting)
        @param reg_factor_fil float, regularization factor for filament currents
        @param reg_factor_wall float, regularization factor for wall currents
        @result tuple, containing the solution, residual, Ax, and B
        '''
        if not self.reduced_created_flag:
            raise ValueError("Reduced model has not been created yet. Please create or initialize a reduced model before running reconstruction.")
        
        # intializing the Ms and Msc matrices with the appropriate weighting and scaling based off sigma
        Ms_weighted = self.torus_reduced.Ms/sigma[:]
        Msc_weighted = self.torus_reduced.Msc/sigma[:]
        Msc_weighted_fil = Msc_weighted[num_non_fil_coils:, :]

        # This section of code scales the total ip constraint row of the matrix to ensure it has a comparable influence on the 
        # least squares solution as the magnetic measurements, based on the provided ip_weight and the magnitude of ip_at_time
        ip_row_scale = 1.0 / abs(ip_at_time)
        ip_col_ms = np.zeros((Ms_weighted.shape[0], 1))
        ip_col_msc = np.ones((Msc_weighted_fil.shape[0], 1)) * (ip_weight * ip_row_scale)

        # appending the ip constraint as an additional row to the Ms and Msc matrices, with appropriate scaling
        Ms_final = np.append(Ms_weighted, ip_col_ms, axis=1)
        Msc_final = np.append(Msc_weighted_fil, ip_col_msc, axis=1)
        combined_matrix_A = np.vstack((Ms_final, Msc_final)).T

        # here we prepare the the B vector by subtacting the non plasma filament contribution from the sensor measurements
        # we also scale by sigma here as well (to ensure magnetic sensor signals are normalized to each other - one sensor doesnt dominate)
        # finally we append the ip cosntraint row as well 
        B_weighted = (Psi_at_time - coil_curr_at_time @ self.torus_reduced.Msc[:num_non_fil_coils, :]) / sigma[:] 
        B_weighted = np.append(B_weighted, [ip_at_time * ip_weight * ip_row_scale])

        # here we apply tikonov regularization to both the filament and wall component of the lstq 
        # note that we must use the unmodified shapes of Ms to construct the identity matrices for regularization
        # the first half of the identity matrix corresponds to the wall currents and the second half corresponds to the filament currents, so we slice accordingly when stacking them below
        # we then add the reg rows to B as well to match the extra rows that we stacked to A 
        num_to_solve = Ms_final.shape[0] + Msc_final.shape[0]
        reg_identity = np.eye(num_to_solve)
        A = np.vstack([combined_matrix_A, reg_factor_wall * reg_identity[0:Ms_weighted.shape[0], :], reg_factor_fil * reg_identity[Ms_weighted.shape[0]:, :]])
        B= np.concatenate([B_weighted, np.zeros(A.shape[1])])

        # solving the least squares problem
        AtA = A.T @ A
        AtB = A.T @ B
        solution = np.linalg.solve(AtA, AtB)
        Ax = np.dot(A, solution)
        residual = np.sqrt(np.sum((B - Ax)**2))

        return solution, residual, Ax, B

    def run_reconstruction_svd_laplace(self, Psi_at_time, ip_at_time, coil_curr_at_time, Msc_coils, Ms, U, ls_mat, ls_mat_fil, lap_proj, ip_row, N, nModes=19, lam=None, lap_lam=1e-8, reg_wall=1e-5):
        '''! Run the filament current reconstruction with the svd + laplacian method.
        @param Psi_at_time np.ndarray, sensor flux measurements at time

        '''
                
        num_Ms = Ms.shape[0]
        if lam is None: 
            ip_row_now_normalized = np.linalg.norm(ip_row) / abs(ip_at_time)
            probe_row_norm = np.median(np.linalg.norm(ls_mat, axis=1))
            lam = probe_row_norm / ip_row_now_normalized
        ip_scale = np.linalg.norm(U[:, :nModes].sum(axis=0))
        B = Psi_at_time - coil_curr_at_time @ Msc_coils
        reg_mat = np.vstack([
            ls_mat,                                    # measurements
            lam * ip_row,                                                        # Ip constraint (wall=0)
            np.hstack([reg_wall * np.eye(num_Ms), np.zeros((num_Ms, ls_mat_fil.shape[1]))]),      # wall Tikhonov
            np.hstack([np.zeros((N, num_Ms)), lap_lam * lap_proj])                   # fil Laplacian
        ])
        psi_reg = np.concatenate([
            B,
            np.array([lam * ip_at_time / ip_scale]),
            np.zeros(num_Ms),
            np.zeros(N)
        ])

        AtA = reg_mat.T @ reg_mat
        AtB = reg_mat.T @ psi_reg
        curr_weights = np.linalg.solve(AtA, AtB)
        curr_expand = U[:, :nModes] @ curr_weights[num_Ms:]  # only the filament part contributes to the expansion
        wall_expand = curr_weights[:num_Ms]  # wall coefficients
        psi_expand  = ls_mat @ curr_weights
        Ax          = ls_mat @ curr_weights  # same thing
        
        ip_reconstructed = curr_expand.sum()
        ip_actual        = ip_at_time.sum()
        ip_error_pct     = 100 * np.abs(ip_reconstructed - ip_actual) / (abs(ip_actual) + 1e-30)
        fit_residual     = np.linalg.norm(Ax - B)
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

        return curr_expand, wall_expand, psi_expand, diagnostics







def reconstruct_current(time_index,PsiAtTime,totalip,coil_curr,Msc_coils,Ms,U,ls_mat,ls_mat_wall, ls_mat_fil, lap_proj,ip_row,rgrid,zgrid,N,nModes=19,lam=None,lap_lam=1e-8,reg_wall=1e-5, plot=False):
    curr_test = totalip[time_index]
    num_Ms = Ms.shape[0]

    # --- Auto-scale lam if not provided ---
    if lam is None:
        # probe_row_norm = np.median(np.linalg.norm(ls_mat, axis=1))       
        # print(f"probe_row_norm = {probe_row_norm}")
        # ip_row_norm    = np.linalg.norm(ip_row)
        # print(f"ip_row_norm = {ip_row_norm}")
        # lam_temp = probe_row_norm / ip_row_norm
        # print(f"lam_temp = {lam_temp}")
        # ip_scale = abs(curr_test)
        # print(f"ip_scale = {ip_scale}")
        # lam = lam_temp/ip_scale
        
        # print(f'lam = {lam}')

        ip_row_norm_normalized = np.linalg.norm(ip_row) / abs(curr_test.sum())
        probe_row_norm = np.median(np.linalg.norm(ls_mat, axis=1))
        lam = probe_row_norm / ip_row_norm_normalized
    ip_scale = np.linalg.norm(U[:, :nModes].sum(axis=0))
    # --- Build RHS ---
    B = PsiAtTime - coil_curr[time_index] @ Msc_coils

    reg_mat = np.vstack([
    ls_mat,                                    # measurements
    lam * ip_row,                                                        # Ip constraint (wall=0)
    np.hstack([reg_wall * np.eye(num_Ms), np.zeros((num_Ms, ls_mat_fil.shape[1]))]),      # wall Tikhonov
    np.hstack([np.zeros((N, num_Ms)), lap_lam * lap_proj])                   # fil Laplacian
    ])

    psi_reg = np.concatenate([
    B,
    np.array([lam * curr_test / ip_scale]),
    np.zeros(num_Ms),
    np.zeros(N)
    ])

    # --- Solve ---

    AtA = reg_mat.T @ reg_mat
    AtB = reg_mat.T @ psi_reg
    curr_weights = np.linalg.solve(AtA, AtB)

    #curr_weights, _, rank, sv = np.linalg.lstsq(reg_mat, psi_reg, rcond=None)
    # --- Project back ---
    curr_expand = U[:, :nModes] @ curr_weights[num_Ms:]  # only the filament part contributes to the expansion
    wall_expand = curr_weights[:num_Ms]  # wall coefficients
    psi_expand  = ls_mat @ curr_weights
    Ax          = ls_mat @ curr_weights  # same thing

    # --- Diagnostics ---
    ip_reconstructed = curr_expand.sum()
    ip_actual        = curr_test.sum()
    ip_error_pct     = 100 * np.abs(ip_reconstructed - ip_actual) / (abs(ip_actual) + 1e-30)
    fit_residual     = np.linalg.norm(Ax - B)
    fit_residual_nonorm = Ax - B

    diagnostics = {
        'lam':              lam,
        'lap_lam':          lap_lam,
        'ip_reconstructed': ip_reconstructed,
        'ip_actual':        ip_actual,
        'ip_error_pct':     ip_error_pct,
        'fit_residual':     fit_residual,
       # 'rank':             rank,
        'curr_weights':     curr_weights,
        'fit_residual_nonorm': fit_residual_nonorm,
        'ip_row':           ip_row,
    }

    # --- Optional plot ---
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(B,   'k',   label='Measured')
        axes[0].plot(Ax,  'r--', label='Reconstructed')
        ## temp 
        axes[0].set_xlim(0, 34)
        axes[0].set_title(f't_idx={time_index} | res={fit_residual:.3e}')
        axes[0].set_xlabel('Mirnov Probe index')
        axes[0].set_ylabel('Flux')
        axes[0].legend()

        triang = tri.Triangulation(rgrid, zgrid)
        axes[1].tricontourf(triang, curr_expand, levels=30, cmap='RdBu_r')
        axes[1].set_title(f'Ip err={ip_error_pct:.1f}%')
        axes[1].set_xlabel('R')
        axes[1].set_ylabel('Z')
        axes[1].set_aspect('equal')
        axes[1].plot(limiter[:, 0], limiter[:, 1], 'k-', label='Limiter')



        axes[2].plot(B,   'k',   label='Measured')
        axes[2].plot(Ax,  'r--', label='Reconstructed')
        axes[2].set_title(f't_idx={time_index} | res={fit_residual:.3e}')
        axes[2].set_xlim(34, 68)
        axes[2].set_xlabel('Flux Probe index')
        axes[2].set_ylabel('Flux')
        axes[2].legend()

        plt.colorbar(axes[1].collections[0], ax=axes[1])

        plt.tight_layout()
        plt.show()

    return curr_expand, wall_expand, psi_expand, diagnostics
    
# ========================================================
# Jamfit Depreciated Functions to be worked on or removed 
# ========================================================

    def plot_sensors(self, sensor_points_mirnov_array, sensor_points_flux, orientations):
        '''! Visualize sensor positions on the ThinCurr mesh using PyVista.

        Renders Mirnov probes as oriented arrows and flux loops as spheres
        overlaid on the wall mesh.

        @param sensor_points_mirnov_array np.ndarray, shape (N, 3) array of Mirnov probe positions
        @param sensor_points_flux dict, dict of flux loop data with keys 'x', 'y', 'z' per loop name
        @param orientations np.ndarray, shape (N, 3) array of normal vectors for each Mirnov probe
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
            positions = np.column_stack([data['x'], data['y'], data['z']])
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
        scale = 1 / (np.linalg.norm(Jfull, axis=1)).max()
        arrows = grid.glyph(scale="vectors", orient="vectors", factor=scale)
        p.add_mesh(grid, color="white", opacity=0.75, show_edges=True)
        p.add_mesh(arrows, cmap="turbo", scalar_bar_args={
            'title': "|J|", "vertical": True, "position_y": 0.25, "position_x": 0.0
        })
        p.show()


# ===============================
# Jamfit Helper Functions
# ===============================

def setup_synthetic_current(timepoints, ip_list, sigma_r, sigma_z, r0, z0, rmesh, zmesh):
    '''! Generate synthetic filament currents using a Gaussian plasma distribution.

    At each time step, spreads the total plasma current across the filament grid
    using a 2D Gaussian centered at (r0, z0) with widths (sigma_r, sigma_z).

    @param timepoints list or np.ndarray, array of time values
    @param ip_list list or np.ndarray, total plasma current at each time step
    @param sigma_r float, Gaussian width in the R direction
    @param sigma_z float, Gaussian width in the Z direction
    @param r0 list or np.ndarray, R position of the plasma centroid at each time step
    @param z0 list or np.ndarray, Z position of the plasma centroid at each time step
    @param rmesh np.ndarray, R coordinates of the filament mesh
    @param zmesh np.ndarray, Z coordinates of the filament mesh
    @result np.ndarray, shape (ntimes, 1 + nfilaments), time column followed by filament currents
    '''
    coil_curr = []
    for i in range(len(timepoints)):
        gaussian_raw = np.exp(
            -((rmesh - r0[i])**2 / (2 * sigma_r**2) + (zmesh - z0[i])**2 / (2 * sigma_z**2))
        )
        gaussian_values = ip_list[i] * (gaussian_raw / np.sum(gaussian_raw))
        coil_curr.append(gaussian_values)

    coil_curr = np.array(coil_curr)
    time_column = np.array(timepoints).reshape(-1, 1)
    coil_curr = np.hstack((time_column, coil_curr))
    return coil_curr


def interpolate_total_current(coil_currs, nsteps, verbose=False):
    '''! Interpolate total plasma current to a higher-resolution time grid.

    Sums the sensor currents at each time step and interpolates the total
    onto a finer time grid using linear interpolation.

    @param coil_currs np.ndarray, shape (ntimes, nsensors+1), first column is time
    @param nsteps int, number of high-resolution steps between first and last time
    @param verbose bool, if True, plots the original and interpolated total current (default: False)
    @result tuple of (high_res_time, total_current_high_res) as np.ndarrays
    '''
    times = coil_currs[:, 0]
    sensor_currents = coil_currs[:, 1:]

    high_res_time = np.linspace(times[0], times[-1], nsteps + 1)

    interpolated_currents = np.array([
        np.interp(high_res_time, times, sensor)
        for sensor in sensor_currents.T
    ]).T  # shape: (nsteps+1, nsensors)

    total_current_high_res = np.sum(interpolated_currents, axis=1)

    if verbose:
        plt.figure(figsize=(8, 5))
        plt.scatter(times, np.sum(sensor_currents, axis=1), color='red', label='Original Data')
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
    
    @param rgrid np.ndarray, R coordinates of the filament grid
    @param zgrid np.ndarray, Z coordinates of the filament grid
    @return lap_mat np.ndarray, the constructed Laplacian matrix for the filament grid
    @ return N int, the number of points in the filament grid
    '''
    points = np.array(list(zip(rgrid, zgrid)))
    R_vals = rgrid 
    Z_vals = zgrid 
    r_unique = np.unique(R_vals)                      
    z_unique = np.unique(Z_vals)
    dr = np.min(np.diff(np.sort(r_unique)))
    dz = np.min(np.diff(np.sort(z_unique)))
    if verbose:
        print("dr =", dr)
        print("dz =", dz)
        print("nr =", len(r_unique))
        print("nz =", len(z_unique))
    N = len(points)
    ## Getting W matrix
    W = np.zeros((N, N))
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
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = np.sum(W[i, :])
    ## Getting Laplacian matrix
    lap_mat = D - W
    return lap_mat, N