import pandas as pd
from datetime import datetime
import pandas as pd
from datetime import datetime
import math

class rinex_V3_02:
    def parse_rinex_obs(self, filename, constellation: str = None):
        records = []
        obs_types = {}  # Store constellation -> list of obs codes

        with open(filename, "r") as f:
            lines = f.readlines()

        # -----------------------
        # Step 1: Parse header
        # -----------------------
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

        # Get obs column names for requested constellation
        columns = obs_types[constellation]
        num_obs = len(columns)
        lines_per_sat = math.ceil(num_obs / 5)  # 5 obs per line max (80 chars)

        # -----------------------
        # Step 2: Parse body
        # -----------------------
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

            # If line starts with the requested constellation
            # If line starts with the requested constellation
            if line.startswith(constellation):
                sat_id = int(line[1:3])

                sat_values = []
                # Read all lines for this satellite, but stop if file ends early
                for k in range(lines_per_sat):
                    if i + k >= len(lines):
                        break  # avoid IndexError at file end

                    sat_line = lines[i + k]

                    # Each line (after sat id) holds up to 5 obs, each 16 chars
                    start = 3 if k == 0 else 0  # skip constellation+id only on first line
                    for j in range(0, 80, 16):
                        val = sat_line[start + j:start + j + 16].strip()
                        if val:
                            try:
                                sat_values.append(float(val))
                            except ValueError:
                                sat_values.append(None)
                        else:
                            sat_values.append(None)

                # Trim to expected num_obs (in case of missing/extra fields)
                sat_values = sat_values[:num_obs]

                time_seconds = (
                    current_time.hour * 3600
                    + current_time.minute * 60
                    + current_time.second
                )

                records.append([time_seconds, sat_id] + sat_values)

                # Skip all lines we consumed for this satellite
                i += lines_per_sat
                continue


            i += 1

        # -----------------------
        # Step 3: Create DataFrame
        # -----------------------
        df = pd.DataFrame(records, columns=["Time_seconds", "SatelliteID"] + columns)
        return df


if __name__=="__main__":
    filename = r"C:\Users\User\Desktop\rinex_out\20240214-041908.obs"
    rinex = rinex_V3_02()
    df = rinex.parse_rinex_obs(filename,"G")

    # Save as tab-delimited file
    df.to_csv("test_RTK1.txt", sep="\t", index=False)

    print(df.head())
