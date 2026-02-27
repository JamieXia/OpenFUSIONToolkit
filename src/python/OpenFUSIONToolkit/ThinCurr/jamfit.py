#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
'''! Core definitions for Jamfit - filament reconstruction
@authors Jamie Xia
@date Feb 2026
'''

## IMPORTING EXTERNAL LIBRARIES ##
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pyvista as pv
import h5py
from collections import OrderedDict

pv.set_jupyter_backend('static')  # Comment to enable interactive plots

sys.path.insert(0, '/Applications/OpenFUSIONToolkit/python')
from OpenFUSIONToolkit.ThinCurr import ThinCurr, ThinCurr_reduced
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors, circular_flux_loop
from OpenFUSIONToolkit.util import mu0
from OpenFUSIONToolkit.io import histfile


# ===============================
# Jamfit Base Class
# ===============================

class Jamfit():
    '''! Main class for Jamfit filament reconstruction.

    Manages setup, synthetic data generation, reduced model creation,
    and eventual reconstruction of plasma filament currents using the
    ThinCurr framework.
    '''

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

    def setup_jamfit(self, floops_path):
        '''! Set up the ThinCurr model, I/O, and sensor/coil mutual inductance matrices.
        @param floops_path str, path to the sensor locations file
        '''
        self.torus = ThinCurr(self.myOFT)
        self.torus.setup_model(mesh_file=self.thincurr_meshfile, xml_filename=self.xml_file)
        self.torus.setup_io()
        self.Msensor, self.Msc, self.sensor_obj = self.torus.compute_Msensor(floops_path)
        self.torus.compute_Mcoil(cache_file='full_HOLDR_M.save')
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

    def gen_synthetic_data(self, coil_currs, dt, nsteps):
        '''! Run a synthetic time-dependent simulation and compute sensor signals.

        Computes the inductance and resistance matrices, runs the time-dependent
        simulation, plots results, and saves the B matrix and sensor history.

        @param coil_currs np.ndarray, combined coil + plasma current array (with time column)
        @param dt float, time step size in seconds
        @param nsteps int, number of time steps
        '''
        self.torus.compute_Lmat(use_hodlr=True, cache_file='full_HOLDR_L.save')
        self.torus.compute_Rmat(copy_out=True)
        self.torus.run_td(dt, nsteps, coil_currs=coil_currs, sensor_obj=self.sensor_obj)
        self.torus.plot_td(nsteps, sensor_obj=self.sensor_obj)
        hist_file = histfile('floops.hist') ## have it return the hist_file object. 
        # mb extract from histfile, sensor data etc. 
        return hist_file, 

    def create_from_runTD_top_modes(self, num_modes, reduced_filename, coil_currs, dt, nsteps, initial_num_eigs=50, verbose=False):
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
        sensors_measurement, currents = torus_first_reduced.run_td(dt, nsteps, coil_currs, status_freq=10)

        temp_curr = currents['curr']
        max_weights = [abs(temp_curr[:, i]).max() for i in range(temp_curr.shape[1])]
        top_modenum_indices = sorted(range(len(max_weights)), key=lambda i: max_weights[i], reverse=True)[:num_modes]

        eig_inds = []
        weight_amplitude = []

        if verbose:
            fig, ax = plt.subplots(1, 1)

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

        self.reduced_torus = self.create_reduced_model(self.eig_vecs[eig_inds, :], reduced_filename, compute_B=False)
        self.reduced_created_flag = True
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

    def initialize_reduced_model(self, reduced_filename):
        '''! Load a previously saved reduced model from file.
        @param reduced_filename str, path to the HDF5 reduced model file
        @result str, confirmation message on successful load
        '''
        self.torus_reduced = ThinCurr_reduced(reduced_filename)
        self.reduced_created_flag = True
        return "Reduced model initialized from file."

    def run_reconstruction_lstsq(self, Psi, totalip, num_coils, ip_weight, magnetics_weight, reg_factor_fil, reg_factor_wall):
        '''! Run the filament current reconstruction.
        @param Psi np.ndarray, sensor flux measurements
        @param totalip np.ndarray, total plasma current measurement
        @param num_coils int, number of coils in the system (not plasma filaments)
        @param ip_weight float, weight for the total plasma current in the reconstruction
        @param magnetics_weight float, weight for the magnetic measurements in the reconstruction
        @param reg_factor_fil float, regularization factor for filament currents
        @param reg_factor_wall float, regularization factor for wall currents
        @result tuple, containing the solution, residual, Ax, and B
        '''
        if not self.reduced_created_flag:
            raise ValueError("Reduced model has not been created yet. Please create or initialize a reduced model before running reconstruction.")
        Ms = np.append(self.torus_reduced.Ms, np.zeros((self.torus_reduced.Ms.shape[0],1)),axis=1) 
        new_col = np.ones((self.torus_reduced.Msc.shape[0], 1))*ip_weight
        new_col[:num_coils] = 0 
        Msc = np.append(self.torus_reduced.Msc, new_col, axis=1)
        combined_matrix = np.vstack((Ms, Msc))
        combined_matrix = combined_matrix.T
        num_Ms = self.torus_reduced.Ms.shape[0]
        num_Msc = self.torus_reduced.Msc.shape[0]
        B = Psi*magnetics_weight
        B = np.append(B, totalip*ip_weight)
        A = combined_matrix 
        reg_identity_fil = reg_factor_fil*np.eye(A.shape[1])
        reg_identity_wall = reg_factor_wall*np.eye(A.shape[1])  
        A = np.vstack([A, reg_identity_wall[:num_Ms, :], reg_identity_fil[num_Ms:num_Ms+num_Msc, :]]) 
        B = np.concatenate([B, np.zeros(A.shape[1])])
        AtA = A.T @ A
        AtB = A.T @ B
        solution = np.linalg.solve(AtA, AtB)  
        Ax = np.dot(A, solution)
        residual = np.sqrt(np.sum((B - Ax)**2)) 
        return solution, residual, Ax, B
    


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