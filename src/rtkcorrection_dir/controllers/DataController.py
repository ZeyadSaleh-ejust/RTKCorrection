from fastapi import FastAPI,APIRouter, Depends,UploadFile
from .ProjectController import ProjectController
from .BaseController import BaseController
from ..models import ResponseSignal
import re
import os

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1048579 # convert MB to bytes