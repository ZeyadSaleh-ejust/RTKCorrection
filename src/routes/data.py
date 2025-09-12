from fastapi import FastAPI,APIRouter, Depends,UploadFile, status, Request
from fastapi.responses import JSONResponse
from helpers.config import get_settings,Settings # mod
from rtkcorrection_dir.controllers import NavigationDataToCSV, NavigationCorrection # mod
import io
import csv
from rtkcorrection_dir.models.ResponseEnums import ResponseSignal # mod
import logging
from rtkcorrection_dir.constants.DataFilesPath import DataFilesPath # mod
import pandas as pd

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

    #navigation_correction = NavigationCorrection()
    
    #nav_data = navigation_correction.parse_nav_data(output_NAV_file)
    #corrected_obs_df = navigation_correction.apply_clock_correction_to_observations(DataFilesPath.INPUT_PREPROCESSED.value,nav_data)
    

    #corrected_obs_df.to_csv('observation_navigation_correction.csv', index=False)
        
    #return corrected_obs_df    
    return JSONResponse(
        status_code=200,
        content ={"saving": "done correctly"}
    )

    



@data_router.post("/savingDB/{table_name}")
async def saving_db(
    request: Request,
    file: UploadFile,
    table_name: str,
    app_settings: Settings = Depends(get_settings)
):
    try:
        # 1. Read uploaded CSV into a pandas DataFrame
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # 2. Ensure proper dtypes
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                if (df[col].dropna() % 1 == 0).all():
                    df[col] = df[col].astype("Int64")  # nullable integer
        df.rename(columns=lambda x: x.strip(), inplace=True)

        # ✅ Special handling for navigation table
        if table_name == "navigation" and "Epoch (UTC)" in df.columns:
                df["Epoch (UTC)"] = pd.to_datetime(df["Epoch (UTC)"], errors="coerce")
                df["Epoch (UTC)"] = df["Epoch (UTC)"].dt.tz_localize("UTC")

        # 3. Build SQL query dynamically
        headers = list(df.columns)
        placeholders = ", ".join([f"${i+1}" for i in range(len(headers))])
        columns = ", ".join([f'"{col}"' for col in headers])
        insert_query = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'

        # 4. Insert into database
        pool = request.app.postgre_sql_pool
        async with pool.acquire() as connection:
            async with connection.transaction():
                for row in df.itertuples(index=False, name=None):
                    await connection.execute(insert_query, *row)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Data inserted successfully into table {table_name}",
                "rows_inserted": len(df)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
