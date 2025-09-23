import pandas as pd
from datetime import datetime
from ..RinexInterface import RinexInterface


class Rinex_V3_02(RinexInterface):
    def __init__(self, filename: str):
        super().__init__(filename)

    # -----------------------------
    # Generic parser for any constellation
    # -----------------------------
    def _parse_constellation(self, constellation: str) -> pd.DataFrame:
        records = []
        obs_types = {}  # Store constellation -> list of obs codes

        with open(self.filename, "r") as f:
            lines = f.readlines()

        # Step 1: Parse header to extract observation types
        start_idx = 0
        for i, line in enumerate(lines):
            if "SYS / # / OBS TYPES" in line:
                sys = line[0]  # constellation letter
                num_types = int(line[4:6])
                obs_list = line[7:60].split()

                # Some constellations may span multiple lines
                while len(obs_list) < num_types:
                    i += 1
                    next_line = lines[i]
                    obs_list.extend(next_line[7:60].split())

                obs_types[sys] = obs_list

            if "END OF HEADER" in line:
                start_idx = i + 1
                break

        if constellation not in obs_types:
            raise ValueError(f"Constellation {constellation} not found in RINEX header")

        # Get the observation columns for requested constellation
        columns = obs_types[constellation]

        # Step 2: Parse body
        i = start_idx
        current_time = None

        while i < len(lines):
            line = lines[i].rstrip("\n")

            # Epoch line starts with ">"
            if line.startswith(">"):
                parts = line.split()
                year, month, day = map(int, parts[1:4])
                hour, minute = map(int, parts[4:6])
                sec = float(parts[6])
                current_time = datetime(year, month, day, hour, minute, int(sec))
                i += 1
                continue

            # Observation lines for the selected constellation
            if line.startswith(constellation):
                sat_id = int(line[1:3])

                values = []
                for j in range(len(columns)):
                    start = 3 + j * 16
                    end = start + 16
                    val = line[start:end].strip()
                    values.append(float(val) if val else None)

                time_seconds = (
                    current_time.hour * 3600
                    + current_time.minute * 60
                    + current_time.second
                )

                records.append([time_seconds, sat_id] + values)

            i += 1

        # Step 3: Create DataFrame dynamically
        df = pd.DataFrame(records, columns=["Time_seconds", "SatelliteID"] + columns)
        return df

    # -----------------------------
    # Specific constellation wrappers
    # -----------------------------
    def GPS(self):      # "G"
        return self._parse_constellation("G")

    def Galileo(self):  # "E"
        return self._parse_constellation("E")

    def GLONASS(self):  # "R"
        return self._parse_constellation("R")

    def BeiDou(self):   # "C"
        return self._parse_constellation("C")

    def NavIC(self):    # "I"
        return self._parse_constellation("I")

    def QZSS(self):     # "J"
        return self._parse_constellation("J")
    
    def SBAS(self):
        return self._parse_constellation("s")


"""
if __name__=="__main__":
    filename = r"C:\Users\User\Desktop\rinex_out\20240214-041908.obs"
    rinex = rinex_V3_02()
    df = rinex.parse_rinex_obs(filename,"G")

    # Save as tab-delimited file
    df.to_csv("test_RTK1.txt", sep="\t", index=False)

    print(df.head())"""
