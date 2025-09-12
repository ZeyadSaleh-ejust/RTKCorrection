import csv
import pandas as pd
from ..constants.DataFilesPath import DataFilesPath

def navToCSV(input_file: str, output_file: str):
    """
    Extracts GPS navigation data from a RINEX navigation file and saves it to a CSV file.
    """
    # --- Processing ---
    data_rows = []
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Skip header (usually first 5 lines)
    data_lines = lines[5:]

    # Each satellite entry = 8 lines
    for i in range(0, len(data_lines), 8):
        block = data_lines[i:i+8]
        if len(block) < 8:
            continue  # incomplete record at end of file

        # -------- Line 1 --------
        line1 = block[0]
        prn_str = line1[0:3].strip()  # e.g. G04
        if not prn_str or prn_str[0] != "G":
            continue

        gnss_type = prn_str[0]
        prn_number = int(prn_str[1:])

        year = int(line1[4:8].strip())
        month = int(line1[9:11].strip())
        day = int(line1[12:14].strip())
        hour = int(line1[15:17].strip())
        minute = int(line1[18:20].strip())
        second = float(line1[21:23].strip())

        epoch_timestamp = pd.Timestamp(year=year, month=month, day=day, 
                                      hour=hour, minute=minute, second=int(second))

        clock_bias = float(line1[23:42].strip())
        clock_drift = float(line1[42:61].strip())
        clock_drift_rate = float(line1[61:80].strip())

        # -------- Line 2 --------
        line2 = block[1]
        IODE = float(line2[4:23].strip())
        Crs = float(line2[23:42].strip())
        delta_n = float(line2[42:61].strip())
        M0 = float(line2[61:80].strip())

        # -------- Line 3 --------
        line3 = block[2]
        Cuc = float(line3[4:23].strip())
        e = float(line3[23:42].strip())
        Cus = float(line3[42:61].strip())
        sqrtA = float(line3[61:80].strip())

        # -------- Line 4 --------
        line4 = block[3]
        toe = float(line4[4:23].strip())
        Cic = float(line4[23:42].strip())
        Omega0 = float(line4[42:61].strip())
        Cis = float(line4[61:80].strip())

        # -------- Line 5 --------
        line5 = block[4]
        i0 = float(line5[4:23].strip())
        Crc = float(line5[23:42].strip())
        omega = float(line5[42:61].strip())
        Omega_dot = float(line5[61:80].strip())

        # -------- Line 6 --------
        line6 = block[5]
        IDOT = float(line6[4:23].strip())
        codes_L2 = float(line6[23:42].strip())
        gps_week = float(line6[42:61].strip())
        L2P_flag = float(line6[61:80].strip())

        # -------- Line 7 --------
        line7 = block[6]
        SV_accuracy = float(line7[4:23].strip())
        SV_health = float(line7[23:42].strip())
        TGD = float(line7[42:61].strip())
        IODC = float(line7[61:80].strip())

        # -------- Line 8 --------
        line8 = block[7]
        toc = float(line8[4:23].strip())
        fit_interval = float(line8[23:42].strip())

        # Collect all fields
        data_rows.append([
            gnss_type, prn_number, epoch_timestamp,
            clock_bias, clock_drift, clock_drift_rate,
            IODE, Crs, delta_n, M0,
            Cuc, e, Cus, sqrtA,
            toe, Cic, Omega0, Cis,
            i0, Crc, omega, Omega_dot,
            IDOT, codes_L2, gps_week, L2P_flag,
            SV_accuracy, SV_health, TGD, IODC,
            toc, fit_interval
        ])

    # --- Save to CSV ---
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "System", "PRN", "Epoch (UTC)",
            "Clock Bias (s)", "Clock Drift (s/s)", "Clock Drift Rate (s/s²)",
            "IODE", "Crs (m)", "Δn (rad/s)", "M0 (rad)",
            "Cuc (rad)", "Eccentricity", "Cus (rad)", "√A (m^0.5)",
            "toe (s)", "Cic (rad)", "Ω0 (rad)", "Cis (rad)",
            "i0 (rad)", "Crc (m)", "ω (rad)", "Ω̇ (rad/s)",
            "IDOT (rad/s)", "Codes L2", "GPS Week", "L2P flag",
            "SV accuracy (m)", "SV health", "TGD (s)", "IODC",
            "toc (s)", "Fit interval (h)"
        ])
        writer.writerows(data_rows)

    print(f"✅ Extraction complete. Data saved to {output_file}")

