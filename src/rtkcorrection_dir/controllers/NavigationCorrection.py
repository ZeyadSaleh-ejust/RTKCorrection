import numpy as np
import pandas as pd
import math
from math import sin, sqrt
from typing import Dict, List, Tuple, Optional


class NavigationCorrection:
    def __init__(self):
                # Constants
        self.C = 299792458.0  # Speed of light in m/s
        self.F_REL = -4.442807633e-10  # Relativistic correction constant
        self.MU = 3.986005e14  # Earth's gravitational constant
        self.OMEGA_DOT_E = 7.2921151467e-5  # Earth rotation rate in rad/s
        self.SECONDS_PER_WEEK = 604800
        self.HALF_WEEK = self.SECONDS_PER_WEEK / 2.0
    def parse_nav_data(self, nav_csv_path: str) -> Dict[str, List[Dict]]:
        """
        Parse navigation data from CSV file and organize by satellite ID
        
        Args:
            nav_csv_path: Path to navigation data CSV file
            
        Returns:
            Dictionary with satellite IDs as keys and lists of navigation records as values
        """
        # Read navigation data
        nav_df = pd.read_csv(nav_csv_path)
        
        # Group by satellite ID
        nav_data = {}
        for sat_id, group in nav_df.groupby('PRN'):
            nav_data[sat_id] = group.to_dict('records')
        
        return nav_data

    def wrap_time(self, dt):
        """Wrap dt into range [-HALF_WEEK, +HALF_WEEK] by +/- 1 week if needed."""
        if dt > self.HALF_WEEK:
            return dt - self.SECONDS_PER_WEEK
        if dt < - self.HALF_WEEK:
            return dt + self.SECONDS_PER_WEEK
        return dt

    def solve_kepler(self,M, e, tol=1e-12, max_iter=50):
        """
        Solve Kepler's equation E - e*sin(E) = M for eccentric anomaly E.
        Uses Newton-Raphson with safe initial guess.
        """
        # Normalize M to [-pi, pi] for better numerical stability
        M = (M + math.pi) % (2.0 * math.pi) - math.pi

        # initial guess
        if abs(e) < 1e-8:
            E = M
        else:
            # good starting point for Newton:
            E = M if e < 0.8 else math.pi

        for _ in range(max_iter):
            f = E - e * math.sin(E) - M
            fp = 1.0 - e * math.cos(E)
            if abs(fp) < 1e-12:
                # avoid division by very small fp
                break
            dE = -f / fp
            E += dE
            if abs(dE) < tol:
                return E
        # fallback to last value
        return E

    def calculate_satellite_clock_correction(self,nav_record: dict, transmit_time: float) -> float:
        """
        Calculate satellite clock correction in seconds.

        Args:
            nav_record: dictionary with required nav fields (see below).
                Required keys (example key strings):
                    'Clock Bias (s)'       -> a0
                    'Clock Drift (s/s)'    -> a1
                    'Clock Drift Rate (s/s^2)' -> a2
                    'toc (s)'              -> clock data reference time (GPS s of week)
                    'Eccentricity'         -> e
                    '√A (m^0.5)'           -> sqrt_a
                    'M0 (rad)'             -> M0
                    'Δn (rad/s)'           -> delta_n
                    'toe (s)'              -> t_oe
            transmit_time: GPS seconds of week at which correction is required.

        Returns:
            Satellite clock correction (seconds).  (Relativistic correction included.)
        """
        # extract coefficients (rename as you store them)
        a0 = nav_record['Clock Bias (s)']
        a1 = nav_record['Clock Drift (s/s)']
        a2 = nav_record['Clock Drift Rate (s/s^2)']
        toc = nav_record['toc (s)']

        # polynomial clock bias: handle week wrap
        dt = transmit_time - toc
        dt = self.wrap_time(dt)
        clock_poly = a0 + a1 * dt + a2 * (dt ** 2)

        # relativistic term
        e = nav_record['Eccentricity']
        sqrt_a = nav_record['√A (m^0.5)']
        M0 = nav_record['M0 (rad)']
        delta_n = nav_record['Δn (rad/s)']
        toe = nav_record['toe (s)']

        # semi-major axis
        a = sqrt_a * sqrt_a

        # computed mean motion
        n0 = math.sqrt(self.MU / (a ** 3))
        n = n0 + delta_n

        # time from ephemeris reference (wrap)
        tk = transmit_time - toe
        tk = self.wrap_time(tk)

        # mean anomaly at transmit_time
        M = M0 + n * tk
        # normalize M (optional)
        # M = (M + math.pi) % (2.0 * math.pi) - math.pi

        # solve Kepler for eccentric anomaly E
        E = self.solve_kepler(M, e)

        # relativistic correction (seconds)
        dt_rel = self.F_REL * e * sqrt_a * math.sin(E)

        # total clock correction (seconds)
        dt_sv = clock_poly + dt_rel

        # Note: TGD (Group delay) is NOT added here by default;
        # apply TGD separately when correcting pseudorange if needed.
        return dt_sv

    def find_best_nav_record(self, nav_records: List[Dict], obs_time: float, max_time_diff: float = 7200) -> Optional[Dict]:
        """
        Find the best navigation record for a given observation time
        
        Args:
            nav_records: List of navigation records for a satellite
            obs_time: Observation time in GPS seconds of week
            max_time_diff: Maximum allowed time difference between observation and nav record (seconds)
            
        Returns:
            Best matching navigation record or None if none found within time tolerance
        """
        best_record = None
        min_time_diff = float('inf')
        
        for record in nav_records:
            t_oc = record['toc (s)']
            time_diff = abs(obs_time - t_oc)
            
            if time_diff < min_time_diff and time_diff <= max_time_diff:
                min_time_diff = time_diff
                best_record = record
        
        return best_record

    def apply_clock_correction_to_observations(self, obs_csv_path: str, nav_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Apply clock corrections to observation data
        
        Args:
            obs_csv_path: Path to observation data CSV file
            nav_data: Dictionary of navigation data organized by satellite ID
            
        Returns:
            DataFrame with corrected observations
        """
        # Read observation data
        obs_df = pd.read_csv(obs_csv_path)
        
        # Add columns for corrected values
        obs_df['C1C_corrected'] = obs_df['C1C']
        obs_df['L1C_corrected'] = obs_df['L1C']
        obs_df['correction_applied'] = False
        obs_df['nav_time_diff'] = np.nan
        
        # Process each observation
            
        for idx, row in obs_df.iterrows():
            # Convert SatelliteID float -> int -> str
            sat_id = str(int(row['SatelliteID']))
            obs_time = row['GPST_seconds']

            if sat_id not in nav_data:
                continue  # Skip if no nav data

            nav_record = self.find_best_nav_record(nav_data[sat_id], obs_time)        
                    
            if nav_record is None:
                continue  # Skip if no suitable navigation record found
                
            # Calculate satellite clock correction
            try:
                dt_sv = self.calculate_satellite_clock_correction(nav_record, obs_time)
            except (ValueError, KeyError):
                continue  # Skip if calculation fails
                
            # Calculate range error due to clock error
            range_error = dt_sv * self.C
            
            # Apply correction to pseudorange
            obs_df.at[idx, 'C1C_corrected'] = row['C1C'] - range_error
            
            # Apply correction to carrier phase (add phase correction)
            # Phase correction in cycles = dt_sv * frequency
            f_l1 = 1575.42e6  # L1 frequency in Hz
            phase_correction = dt_sv * f_l1
            obs_df.at[idx, 'L1C_corrected'] = row['L1C'] + phase_correction
            
            # Record that correction was applied
            obs_df.at[idx, 'correction_applied'] = True
            obs_df.at[idx, 'nav_time_diff'] = obs_time - nav_record['t_oc']
        
        return obs_df

