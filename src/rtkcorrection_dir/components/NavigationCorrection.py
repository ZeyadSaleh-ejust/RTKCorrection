import numpy as np
import pandas as pd
from math import sin, sqrt
from typing import Dict, List, Tuple, Optional


class NavigationCorrection:
    def __init__(self):
                # Constants
        self.C = 299792458.0  # Speed of light in m/s
        self.F_REL = -4.442807633e-10  # Relativistic correction constant
        self.MU = 3.986005e14  # Earth's gravitational constant
        self.OMEGA_DOT_E = 7.2921151467e-5  # Earth rotation rate in rad/s
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
        for sat_id, group in nav_df.groupby('SatelliteID'):
            nav_data[sat_id] = group.to_dict('records')
        
        return nav_data

    def calculate_satellite_clock_correction(self,nav_record: Dict, transmit_time: float) -> float:
        """
        Calculate satellite clock correction using navigation data
        
        Args:
            nav_record: Dictionary containing navigation parameters
            transmit_time: Signal transmission time in GPS seconds of week
            
        Returns:
            Satellite clock correction in seconds
        """
        # Extract clock parameters
        a0 = nav_record['a0']
        a1 = nav_record['a1']
        a2 = nav_record['a2']
        t_oc = nav_record['t_oc']
        
        # Calculate time difference
        dt = transmit_time - t_oc
        
        # Polynomial clock correction
        clock_poly = a0 + a1 * dt + a2 * dt**2
        
        # Relativistic correction
        # Extract orbital parameters
        e = nav_record['e']
        sqrt_a = nav_record['sqrtA']
        m0 = nav_record['M0']
        delta_n = nav_record['deltaN']
        t_oe = nav_record['t_oe']
        
        # Calculate mean motion
        n0 = sqrt(self.MU) / (sqrt_a ** 3)
        n = n0 + delta_n
        
        # Calculate mean anomaly
        m = m0 + n * (transmit_time - t_oe)
        
        # Solve Kepler's equation for eccentric anomaly (using iterative approximation)
        ek = m  # Initial guess
        for _ in range(5):  # 5 iterations should be sufficient
            ek = m + e * sin(ek)
        
        # Calculate relativistic correction
        dt_rel = self.F_REL * e * sqrt_a * sin(ek)
        
        # Total clock correction
        dt_sv = clock_poly + dt_rel
        
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
            t_oc = record['t_oc']
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
            sat_id = str(row['SatelliteID'])
            obs_time = row['GPST_seconds']
            
            # Check if we have navigation data for this satellite
            if sat_id not in nav_data:
                continue  # Skip if no navigation data
                
            # Find the best navigation record for this observation time
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

