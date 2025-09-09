from fastapi import FastAPI
from src.routes import base,data
#from src.rtkcorrection_dir.controllers import NavigationDataToCSV
#from src.rtkcorrection_dir.constants.DataFilesPath import DataFilesPath
#from src.rtkcorrection_dir.controllers.NavigationCorrection import NavigationCorrection


app = FastAPI()

app.include_router(base.base_router)
app.include_router(data.data_router)

"""
if __name__ == "__main__":
    # --- File paths ---

    #############################
    # ** STEP 1: Convert RINEX navigation file to CSV **
    #############################
    input_NAV_file = DataFilesPath.ROVER_NAVIGATION_PARAMETERS.value  # RINEX navigation file
    output_NAV_file = DataFilesPath.ROVER_FULL_NAVIGATION_PARAMETERS_CSV.value  # Output CSV file

    NavigationDataToCSV.navToCSV(input_NAV_file, output_NAV_file)

    #############################
    # ** STEP 2: Apply clock corrections to observation data **
    #############################

    navigation_correction = NavigationCorrection()
    
    nav_data = navigation_correction.parse_nav_data(output_NAV_file)

    print(nav_data)
    
        # Apply corrections to observation data
    #corrected_obs_df = navigation_correction.apply_clock_correction_to_observations('observation_data.csv', nav_data)
        
        # Save corrected data
    #corrected_obs_df.to_csv('corrected_observations.csv', index=False)
        
        # Print summary
    #n_corrected = corrected_obs_df['correction_applied'].sum()
    #n_total = len(corrected_obs_df)
    #print(f"Applied corrections to {n_corrected} of {n_total} observations ({n_corrected/n_total*100:.1f}%)")
        
        # Show some examples
    #print("\nSample of corrected observations:")
    #sample_df = corrected_obs_df[corrected_obs_df['correction_applied']].head()
    #print(sample_df[['SatelliteID', 'GPST_seconds', 'C1C', 'C1C_corrected', 'nav_time_diff']])

"""
    