"""Subspace-Net 
Details
----------
Name: system_model.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the SystemModel class for defining the settings of the DoA estimation system model.
"""

# Imports
import numpy as np
from pathlib import Path
from scipy.io import loadmat

from typing import Optional

from src.steering_vector_generator import SteeringVectorGenerator
from src.sparse_array import get_array_locations, get_virtual_ula_array, get_difference_co_array
from src.config.simulation_config import SystemModelParams

# Cache for loaded antenna pattern data to avoid redundant loading
_antenna_pattern_cache = {}


class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        """Class used for defining the settings of the system model.

        Args:
        -----
            system_model_params (SystemModelParams): Parsed configuration holding
                array/signal settings (N, M, field_type, signal_type, array_form,
                antenna_pattern, etc.).

        Attributes:
        -----------
            field_type (str): Field environment approximation type. Options: "Far", "Near".
            signal_type (str): Signals type. Options: "NarrowBand", "Broadband".
            N (int): Number of sensors.
            M (int): Number of sources.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.
            min_freq (dict): Minimal frequency value for different scenarios.
            max_freq (dict): Maximal frequency value for different scenarios.
            f_rng (dict): Frequency range of interest for different scenarios.
            f_sampling (dict): Sampling rate for different scenarios.
            time_axis (dict): Time axis for different scenarios.
            dist (dict): Distance between array elements for different scenarios.
            array (np.ndarray): Array of sensor locations.

        Methods:
        --------
            define_scenario_params(freq_values: list): Defines the signal_type parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.array = None
        self.virtual_array_ula_seg = None

        self.dist_array_elems = None
        self.time_axis = None
        self.f_sampling = None
        self.max_freq = None
        self.min_freq = None
        self.f_rng = None
        self.is_sparse_array = False if system_model_params.array_form.lower() == 'ula' else True
        self.params = system_model_params
        # Assign signal type parameters
        self.define_scenario_params()

        # # Define array indices
        self.create_array(system_model_params.array_form)
        # Calculation for the Fraunhofer and Fresnel
        self.fraunhofer, self.fresnel = self.calc_fresnel_fraunhofer_distance()

        self.sv_generator = SteeringVectorGenerator(
            array=self.array,
            dist_array_elems=self.dist_array_elems,
            params=self.params
        )
        
        # Load antenna pattern data if enabled (use cache to avoid redundant loading)
        self.pattern_data = None
        if self.params.antenna_pattern:
            # Use default path if not specified
            pattern_file = self.params.antenna_pattern_file
            if pattern_file is None:
                # Default path — real measured ULA3 steering data (Mid band: 136-550 MHz)
                pattern_file = r"C:\GitHub\Hof\Auxiliary\Data\Steering\ULA3\SteeringData_Mid.mat"
            
            if pattern_file:
                # Check cache first
                if pattern_file in _antenna_pattern_cache:
                    self.pattern_data = _antenna_pattern_cache[pattern_file]
                else:
                    self.pattern_data = self.load_antenna_pattern(pattern_file)
                    _antenna_pattern_cache[pattern_file] = self.pattern_data


    def define_scenario_params(self):
        """Defines the signal type parameters based on the specified frequency values."""
        freq_values = self.params.freq_values
        # Define minimal frequency value
        self.min_freq = {"NarrowBand": None, "Broadband": freq_values[0]}
        # Define maximal frequency value
        self.max_freq = {"NarrowBand": None, "Broadband": freq_values[1]}
        # Frequency range of interest
        self.f_rng = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                start=self.min_freq["Broadband"],
                stop=self.max_freq["Broadband"],
                num=self.max_freq["Broadband"] - self.min_freq["Broadband"],
                endpoint=False,
            ),
        }
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {
            "NarrowBand": None,
            "Broadband": 2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"]),
        }
        # Define time axis
        self.time_axis = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                0, 1, self.f_sampling["Broadband"], endpoint=False
            ),
        }
        # distance between array elements
        self.dist_array_elems = {
            "NarrowBand": 1 / 2,
            "Broadband": 1
                         / (2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"])),
        }

    def create_array(self, array_form: str):
        """create an array of sensors locations, around to origin.

        Sets self.array (and self.virtual_array_ula_seg for sparse arrays).

        Args:
            array_form (str): Array geometry, e.g. "ula" or a sparse form such as "mra-N".

        Raises:
            ValueError: If the array form is not supported.
        """
        if array_form.lower() == 'ula':
            self.array = np.linspace(0, self.params.N, self.params.N, endpoint=False)
        elif self.is_sparse_array:
            self.array = get_array_locations(array_form)
            self.virtual_array_ula_seg = get_virtual_ula_array(self.array)
        else:
            raise ValueError(f"{array_form} isn't supported")

    def calc_fresnel_fraunhofer_distance(self) -> tuple:
        """
        In the Far and Near field scenrios, those distances are relevant for the distance grid creation.
        wavelength = 1
        spacing = wavelength / 2
        diemeter = (N-1) * spacing
        Fraunhofer  = 2 * diemeter ** 2 / wavelength
        Fresnel = 0.62 * (diemeter ** 3 / wavelength) ** 0.5
        Returns:
            tuple: fraunhofer(float), fresnel(float)
        """
        wavelength = 1
        spacing = wavelength / 2
        diemeter = (self.params.N - 1) * spacing
        fraunhofer = 2 * diemeter ** 2 / wavelength
        fresnel = 0.62 * (diemeter ** 3 / wavelength) ** 0.5
        # fresnel = ((diemeter ** 4) / (8 * wavelength)) ** (1 / 3)

        return fraunhofer, fresnel

    def load_antenna_pattern(self, mat_file_path: str):
        """
        Load antenna pattern data from a .mat file.
        
        Expected format:
        - freq: Frequency vector (Nfreqs x 1)
        - phi: Azimuth angle vector in degrees (Nazimuth x 1)
        - A: Complex phasor matrix [Nelements, Nazimuth, Nfreqs]
        
        Args:
            mat_file_path (str): Path to the .mat file containing antenna pattern data
            
        Returns:
            dict: Dictionary containing:
                - 'freq': Frequency vector (1D array)
                - 'phi': Azimuth angle vector in degrees (1D array)
                - 'A': Complex phasor matrix [Nelements, Nazimuth, Nfreqs]
                - 'amplitude': Amplitude in linear scale [Nelements, Nazimuth, Nfreqs]
                - 'phase': Phase in radians [Nelements, Nazimuth, Nfreqs]
                   
        Raises:
            FileNotFoundError: If the .mat file doesn't exist
            KeyError: If required variables are not found in the .mat file
        """
        mat_path = Path(mat_file_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"Antenna pattern file not found: {mat_file_path}")
        
        mat_data = loadmat(mat_file_path)
        
        # Check if sSteering struct exists
        if 'sSteering' not in mat_data:
            # Remove MATLAB metadata keys (keys starting with '__')
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            raise KeyError(f"Could not find 'sSteering' struct in {mat_file_path}. "
                          f"Available keys: {data_keys}")
        
        steering_data = mat_data['sSteering']
        
        # Helper function to extract field from MATLAB struct
        def get_struct_field(struct, field_name):
            """Extract field from MATLAB struct (structured array).

            Args:
                struct: Loaded MATLAB struct (numpy structured array) or dict.
                field_name (str): Name of the field to extract.

            Returns:
                The field's value, or None if the field is absent.
            """
            if hasattr(struct, 'dtype') and struct.dtype.names and field_name in struct.dtype.names:
                # It's a structured array - for scalar structs (1x1), use [0, 0]
                if struct.size == 1:
                    return struct[field_name][0, 0]
                else:
                    return struct[field_name]
            elif isinstance(struct, dict) and field_name in struct:
                return struct[field_name]
            else:
                return None
        
        # Get available field names for error messages
        if hasattr(steering_data, 'dtype') and steering_data.dtype.names:
            available_keys = list(steering_data.dtype.names)
        elif isinstance(steering_data, dict):
            available_keys = list(steering_data.keys())
        else:
            available_keys = []
        
        # Find frequency vector
        freq = None
        for key in ['freq', 'frequency', 'f', 'frequencies']:
            freq_data = get_struct_field(steering_data, key)
            if freq_data is not None:
                freq = np.squeeze(freq_data)
                break
        
        # Find azimuth/angle vector
        phi = None
        for key in ['phi', 'azimuth', 'azimuth_base_array', 'az', 'theta', 'angles']:
            phi_data = get_struct_field(steering_data, key)
            if phi_data is not None:
                phi = np.squeeze(phi_data)
                break
        
        # Find complex phasor matrix
        A = None
        for key in ['A', 'pattern', 'phasors', 'antenna_pattern']:
            A_data = get_struct_field(steering_data, key)
            if A_data is not None:
                A = np.array(A_data)
                break
        
        if freq is None:
            raise KeyError(f"Could not find frequency vector in sSteering struct. "
                          f"Available fields: {available_keys}")
        if phi is None:
            raise KeyError(f"Could not find azimuth/angle vector in sSteering struct. "
                          f"Available fields: {available_keys}")
        if A is None:
            raise KeyError(f"Could not find complex phasor matrix in sSteering struct. "
                          f"Available fields: {available_keys}")
        
        # Ensure arrays are properly shaped
        freq = np.atleast_1d(freq).flatten()
        phi = np.atleast_1d(phi).flatten()
        A = np.array(A)
        
        # Verify dimensions: A should be [Nelements, Nazimuth, Nfreqs]
        if A.ndim != 3:
            raise ValueError(f"Expected 3D complex matrix A, got shape {A.shape}")
        
        Nelements, Nazimuth, Nfreqs = A.shape
        
        # Verify dimensions match
        if len(phi) != Nazimuth:
            raise ValueError(f"Dimension mismatch: phi has {len(phi)} elements, "
                           f"but A has {Nazimuth} azimuth points")
        if len(freq) != Nfreqs:
            raise ValueError(f"Dimension mismatch: freq has {len(freq)} elements, "
                           f"but A has {Nfreqs} frequency points")
        
        # Extract amplitude and phase from complex matrix
        amplitude = np.abs(A)  # Amplitude in linear scale
        phase = np.angle(A)    # Phase in radians
        
        pattern_dict = {
            'freq': freq,
            'phi': phi,
            'A': A,
            'amplitude': amplitude,
            'phase': phase,
            'Nelements': Nelements,
            'Nazimuth': Nazimuth,
            'Nfreqs': Nfreqs
        }
        
        print(f"Loaded antenna pattern from {mat_file_path}:")
        print(f"  Elements: {Nelements}, Azimuth points: {Nazimuth}, Frequency points: {Nfreqs}")
        print(f"  Frequency range: {freq.min():.2f} - {freq.max():.2f}")
        print(f"  Azimuth range: {phi.min():.2f}° - {phi.max():.2f}°")
        
        return pattern_dict

    def steering_vec(self, theta, *, distance: Optional[np.ndarray] = None,
                     f: float = 1, nominal=False,
                     pattern_data=None, generate_search_grid=False):
        """Computes the array steering vector(s) for the given angle(s), dispatching to the generator.

        Args:
            theta (np.ndarray): Direction-of-arrival angle(s) in radians.
            distance (np.ndarray, optional): Source range(s) for near-field models.
                Defaults to None.
            f (float): Frequency value/index (used for broadband). Defaults to 1.
            nominal (bool): If True, generate noiseless (nominal) steering vectors.
                Defaults to False.
            pattern_data (dict | tuple, optional): Antenna pattern data; falls back to
                self.pattern_data when antenna_pattern is enabled. Defaults to None.
            generate_search_grid (bool): If True, build a full angle/distance grid
                rather than the diagonal. Defaults to False.

        Returns:
            np.ndarray: Complex steering vector(s), shape [N] for a single angle or
                [N, ...] across the requested angle/distance grid.
        """
        # Use loaded pattern_data if antenna_pattern is enabled and pattern_data is not explicitly provided
        if pattern_data is None and self.params.antenna_pattern and self.pattern_data is not None:
            pattern_data = self.pattern_data
            
        return self.sv_generator.generate(
            theta,
            distance=distance,
            f=f,
            nominal=nominal,
            pattern_data=pattern_data,
            generate_search_grid=generate_search_grid
        )