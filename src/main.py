from fastapi import FastAPI
from src.routes import base,data
import asyncpg
from helpers.config import get_settings
#from src.rtkcorrection_dir.controllers import NavigationDataToCSV
#from src.rtkcorrection_dir.constants.DataFilesPath import DataFilesPath
#from src.rtkcorrection_dir.controllers.NavigationCorrection import NavigationCorrection


app = FastAPI()

@app.on_event("startup")
async def startup_span():
    setting = get_settings()
    app.postgre_sql_pool = await asyncpg.create_pool(
        f"{setting.POSTGRESQL_URL}/{setting.POSTGRESQL_DATABASE}"
    )

@app.on_event("shutdown")
async def shutdown_span():
    await app.postgre_sql_pool.close()


app.include_router(base.base_router)
app.include_router(data.data_router)

