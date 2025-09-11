from fastapi import FastAPI,APIRouter, Depends,UploadFile, status, Request
from fastapi.responses import JSONResponse
from ..helpers.config import get_settings,Settings
from ..rtkcorrection_dir.controllers import NavigationDataToCSV, NavigationCorrection
import aiofiles
from ..rtkcorrection_dir.models.ResponseEnums import ResponseSignal
import logging
from ..rtkcorrection_dir.constants.DataFilesPath import DataFilesPath

logger = logging.getLogger("uvicorn.error")

data_router = APIRouter()

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: str,file: UploadFile,
                      app_settings: Settings=Depends(get_settings)):

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
    corrected_obs_df = navigation_correction.apply_clock_correction_to_observations(DataFilesPath.INPUT_PREPROCESSED.value,nav_data)
    print(type(corrected_obs_df))

    corrected_obs_df.to_csv('observation_navigation_correction.csv', index=False)
        
    return corrected_obs_df    
