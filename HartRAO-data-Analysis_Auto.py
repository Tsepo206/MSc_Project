import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from scipy.stats import pearsonr

inclino_file = 'Oct/inclino2_HartRAO_data_18Oct_Dishrim_sample1s.csv'
wind_file = r'/Users/tsepo/OneDrive/Documents/HartRAO/Wind/wind.log.20241018'

inclino_df = pd.read_csv(inclino_file)

wind_df = pd.read_csv(wind_file, sep=r'\s+', skiprows=1, header=None, 
names=['Year', 'Day_of_Year', 'Hours', 'Minutes', 'Seconds', 'Wind_speed', 'Wind_direction'])

#===Creating a Datetime object time stamp=====
# Clean and preprocess wind data
wind_df = wind_df.replace(',', '', regex=True)
wind_df = wind_df.astype({
    'Year': int, 'Day_of_Year': int, 'Hours': int, 'Minutes': int, 
    'Seconds': int, 'Wind_speed': float, 'Wind_direction': float
})

# Create 'DateTime' for wind data
wind_df['DateTime'] = pd.to_datetime(
    wind_df['Year'].astype(str) + ' ' + wind_df['Day_of_Year'].astype(str), 
    format='%Y %j'
) + pd.to_timedelta(
    wind_df['Hours'], unit='h'
) + pd.to_timedelta(
    wind_df['Minutes'], unit='m'
) + pd.to_timedelta(
    wind_df['Seconds'], unit='s'
)

    
# === Convert Wind Time to SAST (UTC+2) ===
wind_df['timestamp_sast'] = pd.to_datetime(wind_df['DateTime'] + timedelta(hours=2, minutes=0, seconds=0))

# ========= Calculating avg mean wind speed magnitude ========
wind_df['wind_speed_avg'] = wind_df['Wind_speed'].rolling(window=20, min_periods=1).mean()

# ===========Preparing to Calculate STD ===
inclino_df = inclino_df.sort_values('DateTime')

# ======= Calculating the inclinometer standard deviation =====
inclino_df['angle_std'] = inclino_df['Angle'].rolling(window=20, min_periods=1).std()

# ======== Filtering for angle STD > 0.04 and wind speed>3 m/s ======
inclino_fil = inclino_df[inclino_df['angle_std']>0.0]
wind_fil = wind_df[wind_df['wind_speed_avg']>3]

# ==============To make sure that the DateTime are of datatype datetime64[ns]
inclino_fil['DateTime'] = pd.to_datetime(inclino_fil['DateTime'])
wind_fil['DateTime'] = pd.to_datetime(wind_fil['timestamp_sast']).dt.tz_localize(None)

# =========== Time slicing ===============
start_time = pd.to_datetime('2024-10-18 09:58:00')
end_time = pd.to_datetime('2024-10-18 10:05:00')

# ============= Data slicing =================
inclino_filtered = inclino_fil[(inclino_fil['DateTime']>=start_time) & 
                                    (inclino_fil['DateTime']<=end_time)].copy()
wind_filtered = wind_fil[(wind_fil['DateTime']>=start_time) & 
                              (wind_fil['DateTime']<=end_time)].copy()

merged = pd.merge_asof(wind_filtered.sort_values('DateTime'), 
                       inclino_filtered[['DateTime', 'angle_std']],
                       on='DateTime', direction='nearest', 
                       tolerance=pd.Timedelta(seconds=1))

# =============== Calculation Correlation ===============
corr = merged['wind_speed_avg'].corr(merged['angle_std'], method = 'pearson')

#============== Plotting ===================
fig, axs = plt.subplots(2, 1, figsize=(8, 7))

# Subplot 1: Angle vs. Time
ax1 = axs[0]
ax1.plot(merged['DateTime'], merged['angle_std'], marker='o', color='blue', alpha=0.6)
ax1.set_title('Rolling STD_inclinometer and Rolling_AVG_Wind Data')
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Angle (°)')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(merged['DateTime'], merged['wind_speed_avg'], label='Avg_Wind Speed', color='red', alpha=0.6)
ax2.set_ylabel('Wind Speed (m/s)', color='red')

ax3 = axs[1]
axs[1].plot(merged['DateTime'], merged['Wind_direction'], 
            label='Wind Direction', color='green', alpha=0.6)
axs[1].set_title('Wind Direction Over Time')
axs[1].set_xlabel('DateTime')
axs[1].set_ylabel('Wind Direction (°)')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

# ========== Calculate lagged correlations with confidence intervals ==========
max_lag = 200  # Maximum lag in seconds
correlations = []
p_values = []

for lag in range(-max_lag, max_lag + 1):
    # Apply the lag to WIND SPEED instead of inclination
    shifted_wind = merged['wind_speed_avg'].shift(lag)
    valid_data = merged[['angle_std']].copy()
    valid_data['Shifted_wind'] = shifted_wind
    valid_data = valid_data.dropna()

    if len(valid_data) > 10:  # Ensure sufficient data points
        correlation, p_val = pearsonr(valid_data['Shifted_wind'], valid_data['angle_std'])
        correlations.append(correlation)
        p_values.append(p_val)
    else:
        correlations.append(np.nan)
        p_values.append(np.nan)

# Convert to DataFrame
correlation_df = pd.DataFrame({
    'Lag': range(-max_lag, max_lag + 1),
    'Correlation': correlations,
    'P_value': p_values
})

# Identify statistically significant correlations (p < 0.05)
correlation_df['Significant'] = correlation_df['P_value'] < 0.05

from scipy.signal import find_peaks

# Plot the correlations
plt.figure(figsize=(10, 5))

# Extract correlation and lag data
lags = correlation_df['Lag']
corrs = correlation_df['Correlation']

# Find peaks in the correlation
#peaks, _ = find_peaks(corrs.fillna(method='ffill'))  # Avoid NaNs
peaks, properties = find_peaks(corrs.fillna(method='ffill'), prominence=0.1)
peak_corrs = corrs.iloc[peaks]
peak_lags = lags.iloc[peaks]

# If peaks were found, select the one with the max correlation value
if not peak_corrs.empty:
    true_peak_idx = peak_corrs.idxmax()
    true_peak_corr = correlation_df.loc[true_peak_idx, 'Correlation']
    true_peak_lag = correlation_df.loc[true_peak_idx, 'Lag']
else:
    true_peak_corr = np.nan
    true_peak_lag = np.nan

# Plot the full correlation curve
plt.plot(lags, corrs, linewidth=2, label='Correlation vs Lag', color='blue')

# Highlight all local maxima with red dots and annotate with arrows
for x, y in zip(peak_lags, peak_corrs):
    plt.plot(x, y, 'ro')  # red dot
    plt.annotate(f'Lag: {x:.1f}s\nCorr: {y:.2f}',
                 xy=(x, y),
                 xytext=(x + 8, y - 0.2),  # you can adjust offset if needed
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=8, color='red')

# Highlight the true peak with a dashed vertical line (green)
if not np.isnan(true_peak_lag):
    plt.axvline(true_peak_lag, color='green', linestyle='--', linewidth=1.5,
                label=f'True Peak Lag = {true_peak_lag:.1f}s')

# Plot styling
plt.xlabel('Lag (Seconds)')
plt.ylabel('Pearson Correlation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
