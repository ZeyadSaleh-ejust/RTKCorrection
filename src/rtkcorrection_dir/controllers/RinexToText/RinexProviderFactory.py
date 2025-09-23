from RinexEnums import RinexEnum,RinexConstellations
from providers.Rinex_V3_02 import Rinex_V3_02

class RinexFactory:
    def __init__(self, version: str, constellation: RinexConstellations, filename: str):
        self.version = version
        self.constellation = constellation
        self.filename = filename

    def parse(self):
        # Map versions to their classes
        version_map = {
            "3.02": Rinex_V3_02,
            # "4.00": Rinex_V4_00,  # Example for future
        }

        if self.version not in version_map:
            raise ValueError(f"Unsupported RINEX version: {self.version}")

        # Create the right RINEX parser class
        rinex_class = version_map[self.version]
        rinex = rinex_class(self.filename)

        # Map constellation to method name
        constellation_map = {
            RinexConstellations.GPS: rinex.GPS,
            RinexConstellations.GALILEO: rinex.Galileo,
            RinexConstellations.GLONASS: rinex.GLONASS,
            RinexConstellations.BEIDOU: rinex.BeiDou,
            RinexConstellations.NAVIC: rinex.NavIC,
            RinexConstellations.QZSS: rinex.QZSS,
            RinexConstellations.SBAS: rinex.SBAS,
        }

        if self.constellation not in constellation_map:
            raise ValueError(f"Unsupported constellation: {self.constellation}")

        # Call the correct method dynamically
        return constellation_map[self.constellation]()
