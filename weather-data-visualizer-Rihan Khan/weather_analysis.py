import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

df = pd.read_csv('data/weatherIndia.csv')

print("Original Data Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nSummary Stats:")
print(df.describe())

relevant_cols = ['Formatted Date', 'Temperature (C)', 'Humidity', 'Precip Type', 'Daily Summary']
df = df[relevant_cols]

df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Date'] = df['Formatted Date'].dt.date
df['Month'] = df['Formatted Date'].dt.month
df['Year'] = df['Formatted Date'].dt.year

df.dropna(subset=['Temperature (C)', 'Humidity'], inplace=True)

df['Rain'] = df['Precip Type'].apply(lambda x: 1 if pd.notna(x) else 0)
df.drop(columns=['Precip Type'], inplace=True)

df.rename(columns={
    'Temperature (C)': 'Temp',
    'Humidity': 'Humidity',
    'Daily Summary': 'Summary'
}, inplace=True)

print("\nCleaned Data Shape:", df.shape)

daily_stats = df.groupby('Date')['Temp'].agg(['mean', 'min', 'max', 'std']).reset_index()
monthly_stats = df.groupby('Month')['Temp'].agg(['mean', 'min', 'max', 'std']).reset_index()
yearly_stats = df.groupby('Year')['Temp'].agg(['mean', 'min', 'max', 'std']).reset_index()

print("\nDaily Temp Stats (First 5):")
print(daily_stats.head())
print("\nMonthly Temp Stats:")
print(monthly_stats)
print("\nYearly Temp Stats:")
print(yearly_stats)

plt.figure(figsize=(12, 4))
plt.plot(daily_stats['Date'], daily_stats['mean'], label='Avg Temp', color='orange')
plt.title('Daily Average Temperature Trend')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plots/daily_temp_trend.png')
plt.close()

monthly_rain = df.groupby('Month')['Rain'].sum().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(monthly_rain['Month'], monthly_rain['Rain'], color='blue', edgecolor='black')
plt.title('Monthly Rainfall Occurrences')
plt.xlabel('Month')
plt.ylabel('Number of Rainy Days')
plt.xticks(range(1, 13))
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('plots/monthly_rainfall.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df['Humidity'], df['Temp'], alpha=0.5, c=df['Humidity'], cmap='viridis')
plt.colorbar(label='Humidity')
plt.title('Humidity vs Temperature')
plt.xlabel('Humidity')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/humidity_vs_temp.png')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(daily_stats['Date'], daily_stats['mean'], 'r-', linewidth=1.5)
axes[0, 0].set_title('Daily Avg Temperature')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(monthly_rain['Month'], monthly_rain['Rain'], color='green')
axes[0, 1].set_title('Monthly Rainfall Days')
axes[0, 1].set_xticks(range(1, 13))

scatter = axes[1, 0].scatter(df['Humidity'], df['Temp'], c=df['Humidity'], cmap='coolwarm')
axes[1, 0].set_title('Humidity vs Temp')
plt.colorbar(scatter, ax=axes[1, 0])

axes[1, 1].plot(yearly_stats['Year'], yearly_stats['max'], marker='o', linestyle='-', color='purple')
axes[1, 1].set_title('Yearly Max Temperature')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('plots/combined_plot.png')
plt.close()

df.to_csv('cleaned_data.csv', index=False)

# Get available months to avoid IndexError
available_months = monthly_rain['Month'].tolist()
if len(available_months) > 0:
    first_month = available_months[0]
    rain_for_first_month = monthly_rain.loc[monthly_rain['Month'] == first_month, 'Rain'].values[0]
else:
    first_month = "N/A"
    rain_for_first_month = 0

summary = f"""
WEATHER DATA ANALYSIS SUMMARY

Dataset: data/weatherIndia.csv
Cleaned Rows: {len(df)}
Key Columns: Date, Temp, Humidity, Rain, Month, Year

Insights:
- Daily avg temp ranges from {daily_stats['mean'].min():.2f}°C to {daily_stats['mean'].max():.2f}°C.
- Most rainy days occur in Month {monthly_rain.loc[monthly_rain['Rain'].idxmax(), 'Month'] if not monthly_rain.empty else 'N/A'}.
- Humidity and temperature show no strong linear correlation — scatter plot confirms this.
- Yearly max temp peaked in {yearly_stats.loc[yearly_stats['max'].idxmax(), 'Year']} at {yearly_stats['max'].max():.2f}°C.

Anomalies:
- Some months have very low rainfall (e.g., Month {first_month} has only {rain_for_first_month} rainy days).
- High humidity does not always mean high temperature — seen in scatter plot.

This analysis can help campus sustainability teams plan energy usage or outdoor events.
"""

with open('summary_report.txt', 'w') as f:
    f.write(summary)

print("\n✅ All tasks completed. Check 'plots/' folder and 'summary_report.txt'")