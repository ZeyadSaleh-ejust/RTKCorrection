from enum import Enum

class RinexEnum(Enum):
    RINEX_V2_10 = "RINEX_V2_10"
    RINEX_V2_11 = "RINEX_V2_11"
    RINEX_V2_12 = "RINEX_V2_12"
    RINEX_V3_00 = "RINEX_V3_00"
    RINEX_V3_02 = "RINEX_V3_02"
    RINEX_V3_03 = "RINEX_V3_03"
    RINEX_V3_04 = "RINEX_V3_04"

class RinexConstellations(Enum):
    GPS = "GPS"
    GLONASS = "GAL"
    GALILEO = "GAL"
    QZSS = "QZS"
    NAVIC = "NavIc"
    SBAS = "SBS"
    BEIDOU = "BDS"
