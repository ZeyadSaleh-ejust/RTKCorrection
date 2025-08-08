filename = "/MATLAB Drive/rnx3/20240214-041908.obs";  
data = rinexread(filename);

% Extract relevant columns and convert to table
df = data.GPS(:, ["SatelliteID","C1C","L1C","D1C","S1C"]);
df = timetable2table(df);

% Convert time to total seconds
x = second(df.Time);
y = minute(df.Time);
z = hour(df.Time);
Time_seconds = (y * 60) + (z * 3600) + x;

% Add Time_seconds and remove original Time
df = addvars(df, Time_seconds, 'Before', 'SatelliteID');
df = removevars(df, "Time");

% Reorder columns to desired output format
df = df(:, ["Time_seconds", "SatelliteID", "C1C", "L1C", "D1C", "S1C"]);

% Write to file with tab delimiter
filename = 'test_RTK1.txt';
writetable(df, filename, 'Delimiter', '\t');


