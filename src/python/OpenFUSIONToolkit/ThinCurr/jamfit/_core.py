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
import pyvista 
from collections import OrderedDict
pyvista.set_jupyter_backend('static') # Comment to enable interactive plots

sys.path.insert(0, '/Applications/OpenFUSIONToolkit/python')
from OpenFUSIONToolkit.ThinCurr import ThinCurr, ThinCurr_reduced
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors, circular_flux_loop
from OpenFUSIONToolkit.util import mu0
from OpenFUSIONToolkit.io import histfile
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_mhdin, read_kfile
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh



# ===============================
# Jamfit Base Class
# ===============================

class Jamfit(): 
    def __init__(self,xml_file, thincurr_meshfile, tokamaker_meshfile, nthreads = None, oft_env = None):
        if oft_env is not None:
            self.myOFT = oft_env
        else:
            if nthreads is None:
                raise ValueError("Either oft_env or nthreads must be provided")
            self.myOFT = OFT_env(nthreads=nthreads)
        
        self.xml_file = xml_file
        self.thincurr_meshfile = thincurr_meshfile
        self.tokamaker_meshfile = tokamaker_meshfile
        self.reduced_created_flag = False 

    def set_xml(self, xml_file):
        self.xml_file = xml_file
    
    def set_sensors_manually(self, sensor_array, floops_path = 'floops.loc'): 
        self.sensors = sensor_array
        save_sensors(self.sensors, filename=floops_path)
        return floops_path

    def setup_jamfit(self, floops_path): 
        self.torus = ThinCurr(self.myOFT)
        self.torus.setup_model(mesh_file = self.thincurr_meshfile, xml_filename = self.xml_file)
        self.torus.setup_io()
        if self.sensors is not None:
            self.Msensor, self.Msc, self.sensor_obj = self.torus.compute_Msensor(floops_path)
            self.torus.compute_Mcoil(cache_file = 'full_HOLDR_M.save')
            self.torus.compute_Lmat(use_hodlr=True, cache_file = 'full_HOLDR_L.save') # Compute the inductance matrix, hodlr is recommended for speed and memory
            self.torus.compute_Rmat() 
        else:
            print("No sensors have been set up yet. Mutual inductance matricies have not been computed.")
        print('Jamfit setup complete.')

    def load_synthetic_data(self, time_array, num_fil_points, coil_current_array, nsteps, intial_r0, intial_z0, totalip, sigma_r, sigma_z, R, Z, num_eigs = 50, verbose = False):
        #assuming that time_array is in milliseconds
        # setting up time parameters
        timeoffset = time_array[0]/1E3 # this needs to be in seconds 
        self.dt = ((time_array[-1]/1000) - timeoffset)/nsteps # change in time for run_td
        self.nsteps = nsteps
        # setting up coils 
        coil_time = ((time_array/1E3) - timeoffset).reshape(-1, 1)  #time array with time offset to start at 0
        coil_curr = np.hstack([coil_time, coil_current_array])
        #setting up filaments/plasma current
        self.num_points = num_fil_points
        r0_list = intial_r0*np.ones(self.num_points) # note this is where the current centroid is assumed to be stationary (which is unlikely during a disruption)
        z0_list = intial_z0 * np.ones(self.num_points)
        plasma_curr = setup_synthetic_current(time_array, totalip, sigma_r, sigma_z, r0_list, z0_list, R, Z)
        high_res_time, total_current_high_res = interpolate_total_current(plasma_curr, nsteps, verbose = verbose)
        plasma_curr = plasma_curr[:, 1:] #removing time column for thincurr run

        self.final_coil_currs = np.hstack((coil_curr, plasma_curr)) #getting final coil currents with plasma currents added on for run_td 
        self.torus.run_td(self.dt,self.nsteps,status_freq=10,coil_currs=self.final_coil_currs,sensor_obj=self.sensor_obj) 

        self.eig_vals, self.eig_vecs = self.torus.get_eigs(num_eigs,False)
        self.torus.plot_td(self.nsteps,compute_B=True,sensor_obj=self.sensor_obj)
        _, self.Bc = self.torus.compute_Bmat(cache_file='HODLR_B.save') #mb abstract to user input defined named file? 
        hist_file = histfile('floops.hist') # hist file is storing the sensor signals and their response in time
        print(f'Synthetic time dependent run complete and {num_eigs} eigenvalues computed.')


    def create_reduced(self, num_modes, reduced_filename, verbose = False):
        torus_first_reduced = self.torus.build_reduced_model(self.eig_vecs, filename = 'first_reduced_model_temp.h5', sensor_obj=self.sensor_obj) ##user input 
        sensors_measurement, currents = torus_first_reduced.run_td(self.dt,self.nsteps,self.final_coil_currs, status_freq=10)
        temp_curr = currents['curr']
        max_weights = [abs(temp_curr[:,i]).max() for i in range(temp_curr.shape[1])]
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
        self.reduced_torus = self.torus.build_reduced_model(self.eig_vecs[eig_inds,:], filename = reduced_filename, compute_B=False, sensor_obj=self.sensor_obj)
        print(f"Reduced model created with {num_modes} modes")
        self.reduced_created_flag = True 
        return self.reduced_torus


    def add_freq_eigenvalues(self, specific_fil_array): 
        from IPython.display import clear_output
        freq_count = 0
        if self.eig_vecs is None:
            raise ValueError("Eigenvalues and eigenvectors have not been computed yet.")
        eig_vecs_wfreq = np.copy(self.eig_vecs)
        for i in specific_fil_array:
            target_fil = i 
            Mcoil = self.torus.compute_Mcoil()
            driver = np.zeros((2,self.torus.nelems))
            driver[0,:] = Mcoil[target_fil,:]
            result = self.torus.compute_freq_response(driver,freq=1.E3)
            eig_vecs_wfreq = np.concatenate((eig_vecs_wfreq, result[:1,:]), axis = 0) 
            clear_output(wait=True)
            freq_count+=1 
        self.eig_vecs = eig_vecs_wfreq
        return "computed frequency response eigenvalues"
    

    def plot_sensors(self, sensor_points_mirnov_array, sensor_points_flux, orientations):
        plot_data = self.torus.build_XDMF()
        grid = plot_data['ThinCurr']['smesh'].get_pyvista_grid()
        p = pv.Plotter()
        p.camera.up = [0, 0, -100]  
        p.camera_position = [(9, 6, 2), (0, 0, 0), (0, 0, 1)] 

        # Add your main grid
        p.add_mesh(grid, color="white", opacity=0.75, show_edges=False)
        # Add mirnov sensor arrows
        sensor_points_mirnov = pv.PolyData(sensor_points_mirnov_array)
        sensor_points_mirnov.point_data['vectors'] = orientations
        arrow = pv.Arrow(tip_length=0.5, tip_radius=0.2, shaft_radius=0.05)
        arrows = sensor_points_mirnov.glyph(orient='vectors', scale='vectors', factor=0.5, geom=arrow)
        p.add_mesh(arrows, color='blue', show_scalar_bar=False)

        # Add flux sensor spheres
        for name, data in sensor_points_flux.items():
            positions = np.column_stack([data['x'], data['y'], data['z']])
            flux_polydata = pv.PolyData(positions)
            spheres_flux = flux_polydata.glyph(geom=pv.Sphere(radius=0.05))
            p.add_mesh(spheres_flux, color='red')

        p.add_axes(interactive=False)
        # Optional bounds: p.show_bounds(grid='front', location='outer', all_edges=True)
        p.show()

    def plot_wall_currents(self, time_index):
        plot_data = self.torus.build_XDMF()
        plot_times = (plot_data['ThinCurr']['smesh'].times)
        grid = plot_data['ThinCurr']['smesh'].get_pyvista_grid()
        Jfull = plot_data['ThinCurr']['smesh'].get_field('J_v',  plot_times[time_index])  
        grid["vectors"] = Jfull
        grid.set_active_vectors("vectors")
        p = pyvista.Plotter()
        scale = 1/(np.linalg.norm(Jfull,axis=1)).max()
        arrows = grid.glyph(scale="vectors", orient="vectors", factor=scale)
        p.add_mesh(grid, color="white", opacity=0.75, show_edges=True)
        p.add_mesh(arrows, cmap="turbo", scalar_bar_args={'title': "|J|", "vertical": True, "position_y":0.25, "position_x": 0.0})
        p.show()

    def intialized_reduced_model(self, reduced_filename): 
        self.torus_reduced = ThinCurr_reduced(reduced_filename) 
        self.reduced_created_flag = True
        return "Reduced model initialized from file."

    def run_reconstruction(self, PsiFull):
        if not self.reduced_created_flag:
            raise ValueError("Reduced model has not been created yet. Please create or intialize a reduced model before running reconstruction.")
        return "working in progress"

# ===============================
# Jamfit Helper functions
# ===============================
# applies a gassuian distribution based on an inputed ip list and spreads it with a sigmas distrubtion and centers it at r0 and z0 
# returns lisst of coil currents for filaments where the first index of every row is the time 
def setup_synthetic_current(timepoints, ip_list, sigma_r, sigma_z, r0, z0, rmesh, zmesh):
    coil_curr = [] 
    for i in range(len(timepoints)): 
        gaussian_raw = np.exp(-((rmesh - r0[i])**2 / (2 * sigma_r**2) + (zmesh - z0[i])**2 / (2 * sigma_z**2)))
        gaussian_values = ip_list[i]*(gaussian_raw/np.sum(gaussian_raw))
        coil_curr.append(gaussian_values) 
    coil_curr = np.array(coil_curr)             
    time_column = np.array(timepoints).reshape(-1, 1)
    coil_curr = np.hstack((time_column, coil_curr))
    return coil_curr


def interpolate_total_current(coil_currs, nsteps, verbose=False):
    """
    Interpolates total current from coil sensor data to a higher-resolution time grid.
    
    Parameters:
    - coil_currs: np.ndarray, shape (ntimes, nsensors+1)
        First column is time, remaining columns are sensor currents.
    - nsteps: int
        Number of high-resolution steps between the first and last time.
    
    Returns:
    - high_res_time: np.ndarray
    - total_current_high_res: np.ndarray
    """
    # Separate time and sensor currents
    times = coil_currs[:, 0]
    sensor_currents = coil_currs[:, 1:]
    
    # Generate high-resolution time grid
    high_res_time = np.linspace(times[0], times[-1], nsteps + 1)
    
    # Interpolate each sensor's current to high-res time using numpy.interp
    interpolated_currents = np.array([
        np.interp(high_res_time, times, sensor)
        for sensor in sensor_currents.T
    ]).T  # shape: (nsteps+1, nsensors)
    
    # Sum currents across sensors at each high-res time step
    total_current_high_res = np.sum(interpolated_currents, axis=1)
    
    # Plotting
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