from enum import Enum
from ..controllers.BaseController import BaseController
from ...helpers.config import get_settings

class DataFilesPath(Enum):
# Base directory for better maintainability
    BASE_DIR = get_settings().PROJECT_DIR

    ROVER_NAVIGATION_PARAMETERS = f"{BASE_DIR}/data/navigation/RoverStationNav.nav"
    INPUT_PREPROCESSED = f"{BASE_DIR}/data/input_preprocesed.csv"
    ROVER_FULL_NAVIGATION_PARAMETERS_CSV = f"{BASE_DIR}/data/navigation/preprocessed/ROVER_FullNavigation_parameters.csv"