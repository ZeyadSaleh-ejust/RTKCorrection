import subprocess
from pathlib import Path
import time
import re

class Convbin:
    def __init__(self):
        # Supported RINEX versions
        self.SUPPORTED_RINEX_VERSIONS = [
            "2.10", "2.11", "2.12",
            "3.00", "3.01", "3.02", "3.03", "3.04"
        ]

        # Supported input formats
        self.SUPPORTED_INPUT_FORMATS = {
            "rtcm2": "RTCM 2",
            "rtcm3": "RTCM 3",
            "nov": "NovAtel OEM7 / OEM3",
            "ubx": "u-blox UBX",
            "ss2": "Superstar II",
            "hemis": "Hemisphere",
            "stq": "SkyTraq",
            "jps": "Javad GREIS",
            "nvs": "NVS BINR",
            "bnx": "BINEX",
            "rt17": "Trimble RT17",
            "sbf": "Septentrino SBF",
            "rinex": "RINEX"
        }

        # Supported constellations mapping (with correct convbin codes)
        self.SUPPORTED_CONSTELLATIONS = {
            "GPS": "G",
            "GLO": "R",
            "GAL": "E",
            "QZS": "J",
            "BDS": "C",
            "NavIC": "I",
            "SBS": "S"
        }

    def raw_to_rinex(self, input_file: str, output_dir: str, convbin_path: str,
                     rinex_version: str = "3.02", input_format: str = "ubx",
                     constellations: list[str] = None):
        """
        Convert GNSS binary file to RINEX using convbin from RTKLIB 2.4.3+
        """

        # --- Validate RINEX version ---
        if rinex_version not in self.SUPPORTED_RINEX_VERSIONS:
            raise ValueError(
                f"Invalid RINEX version '{rinex_version}'. "
                f"Supported versions: {', '.join(self.SUPPORTED_RINEX_VERSIONS)}"
            )

        # --- Validate input format ---
        if input_format not in self.SUPPORTED_INPUT_FORMATS:
            raise ValueError(
                f"Invalid input format '{input_format}'. "
                f"Supported formats: {', '.join(self.SUPPORTED_INPUT_FORMATS.keys())}"
            )

        # --- Validate constellations ---
        constellation_flags = ""
        if constellations:
            invalid = [c for c in constellations if c not in self.SUPPORTED_CONSTELLATIONS]
            if invalid:
                raise ValueError(
                    f"Invalid constellations: {', '.join(invalid)}. "
                    f"Supported: {', '.join(self.SUPPORTED_CONSTELLATIONS.keys())}"
                )
            constellation_flags = "".join(self.SUPPORTED_CONSTELLATIONS[c] for c in constellations)
            print(f"Constellation flags: '{constellation_flags}'")  # DEBUG

        input_path = Path(input_file).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Command for RTKLIB convbin
        cmd = [
            str(Path(convbin_path).resolve()),
            str(input_path),
            "-r", input_format,     # Input format
            "-v", rinex_version,    # RINEX version
            "-od", "-os",           # Include Doppler and SNR
            "-d", str(out_dir)      # Output directory
        ]

        # Add constellation filter if specified
        if constellation_flags:
            cmd += ["-y", constellation_flags]

        print("Running:", " ".join(cmd))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            if result.returncode != 0:
                raise RuntimeError(f"convbin failed with code {result.returncode}")
            
            time.sleep(1)
            self.verify_rinex_version(out_dir, rinex_version)
            
            # NEW: Verify constellations in output files
            if constellations:
                self.verify_constellations(out_dir, constellations)

            print("Conversion finished. Files written to:", out_dir)
            return list(out_dir.glob("*.*"))
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("convbin timed out after 120 seconds")
        except FileNotFoundError:
            raise RuntimeError(f"convbin executable not found at: {convbin_path}")

    def verify_constellations(self, output_dir: Path, expected_constellations: list[str]):
        """
        Verify that the generated files contain only the expected constellations
        """
        output_dir = Path(output_dir)
        
        for obs_file in output_dir.glob("*.obs"):
            try:
                with open(obs_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for constellation markers in RINEX file
                found_constellations = set()
                for line in content.split('\n'):
                    if line.startswith('>'):
                        # Observation epoch line
                        if len(line) > 30:
                            sys_char = line[31]
                            for name, code in self.SUPPORTED_CONSTELLATIONS.items():
                                if code == sys_char:
                                    found_constellations.add(name)
                                    break
                
                expected_codes = {self.SUPPORTED_CONSTELLATIONS[c] for c in expected_constellations}
                found_codes = {self.SUPPORTED_CONSTELLATIONS.get(c, c) for c in found_constellations}
                
                print(f"File: {obs_file.name}")
                print(f"Expected constellations: {expected_constellations}")
                print(f"Found constellations: {list(found_constellations)}")
                
                # Check for unexpected constellations
                unexpected = found_codes - expected_codes
                if unexpected:
                    print(f"⚠ Warning: Found unexpected constellations: {unexpected}")
                else:
                    print("✓ Constellation filter working correctly")
                    
            except Exception as e:
                print(f"Error reading {obs_file}: {e}")

    def verify_rinex_version(self, output_dir: Path, expected_version: str):
        """
        Verify that the generated files are the expected RINEX version
        """
        output_dir = Path(output_dir)
        
        for obs_file in output_dir.glob("*.obs"):
            try:
                with open(obs_file, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                
                if expected_version in first_line:
                    print(f"✓ Verified: {obs_file.name} is RINEX {expected_version}")
                else:
                    print(f"⚠ Warning: {obs_file.name} is not RINEX {expected_version} (found: {first_line})")
                    
            except Exception as e:
                print(f"Error reading {obs_file}: {e}")

    def check_convbin_version(self, convbin_path: str):
        """
        Check the convbin version to ensure it supports RINEX 3.02+
        """
        try:
            cmd = [str(Path(convbin_path).resolve()), "-h"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if "2.4.3" in result.stdout:
                print("✓ convbin version appears to support RINEX 3.02+")
            else:
                print("ℹ convbin version info:", result.stdout[:200] + "...")
                
        except Exception as e:
            print(f"Error checking convbin version: {e}")

"""
# Example usage
if __name__ == "__main__":
    convbin_path = r"C:\Users\User\Desktop\softwares\RTKLIB_bin-rtklib_2.4.3\RTKLIB_bin-rtklib_2.4.3\bin\convbin.exe"
    convbin_exe = Convbin()

    # First check convbin version
    convbin_exe.check_convbin_version(convbin_path)
    
    # Perform conversion (UBX → RINEX 3.02, GPS + GLONASS)
    files = convbin_exe.raw_to_rinex(
        input_file=r"C:\Users\User\Desktop\softwares\RTKLIB_bin-rtklib_2.4.3\RTKLIB_bin-rtklib_2.4.3\bin\20240214-041908.UBX",
        output_dir=r"rinex_out_test",
        convbin_path=convbin_path,
        rinex_version="3.02",
        input_format="ubx",
        constellations=["GPS","GLO","QZS"]  # Test with multiple constellations
    )
    print("Generated files:", [f.name for f in files])"""