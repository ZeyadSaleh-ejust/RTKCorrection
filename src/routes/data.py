from fastapi import FastAPI,APIRouter, Depends,UploadFile, status, Request
from fastapi.responses import JSONResponse
from ..helpers.config import get_settings,Settings
from ..rtkcorrection_dir.controllers import NavigationDataToCSV, NavigationCorrection
import aiofiles
from ..rtkcorrection_dir.models.ResponseEnums import ResponseSignal
import logging

logger = logging.getLogger("uvicorn.error")

data_router = APIRouter()

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: str,file: UploadFile,
                      app_settings: Settings=Depends(get_settings)):
    return project_id

