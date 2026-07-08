import numpy as np
from scipy import interpolate


class SteeringVectorGenerator:
    _instance = None  # shared across all calls

    def __new__(cls, array, dist_array_elems, params):
        """Returns the singleton instance, initializing it on first construction.

        Args:
            array (np.ndarray): Sensor position indices, shape [N].
            dist_array_elems (dict): Inter-element spacing keyed by signal type.
            params (SystemModelParams): System model parameters (N, eta, bias, field_type, ...).

        Returns:
            SteeringVectorGenerator: The shared singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance(array, dist_array_elems, params)
        return cls._instance

    def _init_instance(self, array, dist_array_elems, params):
        """Initializes singleton state, drawing random per-array bias and mislocation noise.

        Args:
            array (np.ndarray): Sensor position indices, shape [N].
            dist_array_elems (dict): Inter-element spacing keyed by signal type.
            params (SystemModelParams): System model parameters (N, eta, bias, signal_type, ...).
        """
        self.array = array
        self.params = params
        self.dist_array_elems = dist_array_elems
        dist = dist_array_elems[params.signal_type]
        print("Initializing Steering Vector Generator with eta={}".format(self.params.eta))
        self._uniform_bias = np.random.uniform(-self.params.bias, self.params.bias, size=1)
        self._mis_distance = np.random.uniform(-self.params.eta * dist, self.params.eta * dist, size=self.params.N)


    def generate(self, theta, *, distance=None, f=1, nominal=False,
                 pattern_data=None, generate_search_grid=False):
        """Smart dispatch based on params.field_type and use of antenna pattern.

        Args:
            theta (np.ndarray): DoA angle(s) in radians.
            distance (np.ndarray, optional): Source range(s) for near-field. Defaults to None.
            f (float): Frequency value/index for broadband. Defaults to 1.
            nominal (bool): If True, generate noiseless steering vectors. Defaults to False.
            pattern_data (dict | tuple, optional): Antenna pattern data; if given, routes
                to the antenna-pattern generator. Defaults to None.
            generate_search_grid (bool): If True, build a full angle/distance grid. Defaults to False.

        Returns:
            np.ndarray: Complex steering vector(s) produced by the selected generator.

        Raises:
            ValueError: If params.field_type is neither "far" nor "near".
        """
        field_type = self.params.field_type.lower()

        if pattern_data is not None:
            return self._generate_antenna_pattern(
                theta, pattern_data, f=f, signal_type=self.params.signal_type
            )

        if field_type == "far":
            return self._generate_far_field(theta, f=f)
        elif field_type == "near":
            return self._generate_near_field(theta, distance=distance, f=f,
                                   nominal=nominal, generate_search_grid=generate_search_grid)
        else:
            raise ValueError(f"Unknown field type: {self.params.field_type}")

    def _generate_far_field(self, theta, f=1):
        """Computes the far-field plane-wave steering vector with bias/mislocation/noise effects.

        Args:
            theta (float | np.ndarray): DoA angle(s) in radians.
            f (float): Frequency scaling for broadband signals. Defaults to 1.

        Returns:
            np.ndarray: Complex steering vector, shape [N] (or broadcast over theta).
        """
        f_sv = {"NarrowBand": 1, "Broadband": f}
        mis_geometry_noise = np.sqrt(self.params.sv_noise_var) * np.random.randn(self.params.N) if self.params.sv_noise_var else 0
        dist = self.dist_array_elems[self.params.signal_type]
        theta_arr = np.atleast_1d(theta)
        # Azimuth sign convention: +2j aligns the analytic steering with the recorded ULA3 manifold
        # (SteeringData_Mid). The original -2j was the mirror (a_analytic(theta)=a_recorded(-theta)),
        # which made the cross-manifold domain-gap eval (analytic-trained -> recorded-tested) predict
        # -theta -> ~90% MD. ESPRIT's readout (+arcsin) and the a_ideal in calibration are flipped to
        # match, so self-consistent recorded results are unchanged.
        if theta_arr.size == 1:
            # Single angle — preserve the original [N] shape exactly.
            return (
                np.exp(
                    2j * np.pi * f_sv[self.params.signal_type]
                    * (self._uniform_bias + self._mis_distance + dist)
                    * self.array * np.sin(theta)
                )
                + mis_geometry_noise
            )
        # Vector of angles (M sources or a grid dictionary) -> [N, Ntheta] via outer product.
        coeff = (self._uniform_bias + self._mis_distance + dist) * self.array  # [N]
        sv = np.exp(2j * np.pi * f_sv[self.params.signal_type]
                    * coeff[:, None] * np.sin(theta_arr)[None, :])              # [N, Ntheta]
        if np.ndim(mis_geometry_noise):
            sv = sv + mis_geometry_noise[:, None]
        else:
            sv = sv + mis_geometry_noise
        return sv

    def _generate_near_field(self, theta: np.ndarray, distance: np.ndarray, f: float = 1,
                             nominal=False, generate_search_grid: bool = False):
        """Computes near-field steering vectors using the second-order (Fresnel) phase model.

        Args:
            theta (np.ndarray): DoA angle(s) in radians, length Ntheta.
            distance (np.ndarray): Source range(s), length Ndist.
            f (float): Frequency scaling for broadband signals. Defaults to 1.
            nominal (bool): If True, omit steering-vector noise. Defaults to False.
            generate_search_grid (bool): If True, return the full [N, Ntheta, Ndist]
                grid; otherwise return the per-source diagonal. Defaults to False.

        Returns:
            np.ndarray: Complex steering vectors, shape [N, Ntheta, Ndist] when
                generate_search_grid is True, else [N, Ntheta].
        """
        f_sv = {"NarrowBand": 1, "Broadband": f}

        theta = np.atleast_1d(theta)[:, np.newaxis]
        distance = np.atleast_1d(distance)[:, np.newaxis]
        array = self.array[:, np.newaxis]
        array_square = np.power(array, 2)
        dist_array_elems = self.dist_array_elems[self.params.signal_type]
        dist_array_elems += self._mis_distance
        dist_array_elems = dist_array_elems[:, np.newaxis]

        first_order = np.einsum("nm, na -> na",
                                array,
                                np.tile(np.sin(theta), (1, self.params.N)).T * dist_array_elems)
        first_order = np.tile(first_order[:, :, np.newaxis], (1, 1, len(distance)))

        second_order = -0.5 * np.divide(np.power(np.outer(np.cos(theta), dist_array_elems), 2)[:, None, :],
                                        distance.T[:, :, None])
        second_order = np.einsum("nm, nkl -> nkl",
                                 array_square,
                                 np.transpose(second_order, (2, 0, 1)))

        time_delay = first_order + second_order

        if not generate_search_grid:
            time_delay = np.diagonal(time_delay, axis1=1, axis2=2)

        # need to divide here by the wavelength, seems that for the narrowband scenario,
        # wavelength = 1.
        if not nominal:
            # Calculate additional steering vector noise
            mis_geometry_noise = ((np.sqrt(2) / 2) * np.sqrt(self.params.sv_noise_var)
                                  * (np.random.randn(*time_delay.shape) + 1j * np.random.randn(*time_delay.shape)))
            return np.exp(2 * -1j * np.pi * time_delay) + mis_geometry_noise
        return np.exp(2 * -1j * np.pi * time_delay)

    @staticmethod
    def _generate_antenna_pattern(theta, antenna_pattern_data, f=1, signal_type="NarrowBand"):
        """
        Generate steering vector from antenna pattern data.
        
        Supports two formats:
        1. Dictionary format (new): frequency-dependent patterns
           - 'freq': frequency vector
           - 'phi': azimuth angle vector in degrees
           - 'A': complex phasor matrix [Nelements, Nazimuth, Nfreqs]
           - 'amplitude': amplitude in linear scale [Nelements, Nazimuth, Nfreqs]
           - 'phase': phase in radians [Nelements, Nazimuth, Nfreqs]
        
        2. Tuple format (legacy): simple angle-dependent patterns
           - (azimuth_base_array, phase_array, amps_array)
        
        Args:
            theta: Angle(s) in radians
            antenna_pattern_data: Dictionary or tuple containing pattern data
            f: Frequency index/value (default=1, used for broadband signals)
            signal_type: Signal type ("NarrowBand" or "Broadband")

        Returns:
            np.ndarray: Complex steering vector(s), shape [Nelements] for a single
                angle or [Nelements, Ntheta] for multiple angles.
        """
        theta_deg = np.rad2deg(theta)
        theta_deg = np.atleast_1d(theta_deg)
        
        # Check if it's the new dictionary format
        if isinstance(antenna_pattern_data, dict):
            return SteeringVectorGenerator._generate_antenna_pattern_dict(
                theta_deg, antenna_pattern_data, f, signal_type
            )
        else:
            # Legacy tuple format for backward compatibility
            return SteeringVectorGenerator._generate_antenna_pattern_tuple(
                theta_deg, antenna_pattern_data
            )
    
    @staticmethod
    def _generate_antenna_pattern_dict(theta_deg, pattern_dict, f=1, signal_type="NarrowBand"):
        """Generate steering vector from dictionary format (frequency-dependent).

        Selects a frequency slice then linearly interpolates the complex pattern in azimuth.

        Args:
            theta_deg (np.ndarray): DoA angle(s) in degrees, length Ntheta.
            pattern_dict (dict): Pattern data with 'freq', 'phi', and 'A'
                [Nelements, Nazimuth, Nfreqs].
            f (float): Frequency index (mapped to a slice of A). Defaults to 1.
            signal_type (str): "NarrowBand" or "Broadband". Defaults to "NarrowBand".

        Returns:
            np.ndarray: Complex steering vector, shape [Nelements] for a single angle
                or [Nelements, Ntheta] otherwise.
        """
        freq = pattern_dict['freq']
        phi = pattern_dict['phi']
        A = pattern_dict['A']  # [Nelements, Nazimuth, Nfreqs]
        Nelements, Nazimuth, Nfreqs = A.shape
        
        # Map frequency f to frequency index
        if signal_type.startswith("NarrowBand"):
            # For narrowband, use middle frequency or first frequency
            freq_idx = Nfreqs // 2 if Nfreqs > 1 else 0
        else:
            # For broadband, map f to frequency index
            # f might be negative (for negative frequencies in FFT)
            # Normalize to valid range
            freq_idx = int(np.clip(f, 0, Nfreqs - 1))
        
        # Interpolate in angle for each element at the selected frequency
        steering_vec = np.zeros((Nelements, len(theta_deg)), dtype=complex)
        
        for elem_idx in range(Nelements):
            # Extract pattern for this element at selected frequency: [Nazimuth]
            A_elem_freq = A[elem_idx, :, freq_idx]
            
            # Interpolate complex values
            # Use real and imaginary parts separately for interpolation
            interp_real = interpolate.interp1d(
                phi, np.real(A_elem_freq),
                bounds_error=False, fill_value="extrapolate", kind='linear'
            )
            interp_imag = interpolate.interp1d(
                phi, np.imag(A_elem_freq),
                bounds_error=False, fill_value="extrapolate", kind='linear'
            )
            
            steering_vec[elem_idx, :] = (
                interp_real(theta_deg) + 1j * interp_imag(theta_deg)
            )
        
        # If single angle, return 1D array; otherwise return 2D
        if len(theta_deg) == 1:
            return steering_vec[:, 0]
        return steering_vec
    
    @staticmethod
    def _generate_antenna_pattern_tuple(theta_deg, antenna_pattern_data):
        """Generate steering vector from tuple format (legacy, angle-dependent only).

        Interpolates phase (deg) and amplitude (dB) over azimuth, then forms amps*exp(j*phase).

        Args:
            theta_deg (np.ndarray): DoA angle(s) in degrees, length Ntheta.
            antenna_pattern_data (tuple): (azimuth_base_array, phase_array, amps_array).

        Returns:
            np.ndarray: Complex steering vector, shape [Nelements] for a single angle
                or [Ntheta, Nelements] for multiple angles.
        """
        azimuth_base_array, phase_array, amps_array = antenna_pattern_data

        interp_phase = interpolate.interp1d(azimuth_base_array, phase_array, axis=0,
                                            bounds_error=False, fill_value="extrapolate")
        interp_amps = interpolate.interp1d(azimuth_base_array, amps_array, axis=0,
                                           bounds_error=False, fill_value="extrapolate")

        phase = np.deg2rad(interp_phase(theta_deg))
        amps = 10 ** (interp_amps(theta_deg) / 20)
        result = amps * np.exp(1j * phase)
        
        # Return 1D if single angle, otherwise return as is
        if len(theta_deg) == 1:
            return result.flatten()
        return result

    @classmethod
    def reset_instance(cls):
        """Clears the cached singleton so the next construction reinitializes it.

        Call between simulations that use different system-model configurations.
        """
        cls._instance = None
