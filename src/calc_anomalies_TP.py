#!/usr/bin/env python

import argparse
import xarray as xr
import sys
import os

def calc_tp_anomalies(input_file, output_prefix):
    print(f"Processing Precipitation anomalies for {input_file}...")
    
    ds_new = xr.open_dataset(input_file).squeeze()
    
    if 'valid_time' in ds_new.coords:
        ds_new = ds_new.rename({'valid_time': 'time'})
        
    ds_new = ds_new[['tp']]
    ds_new = ds_new.drop_vars(['number', 'step', 'surface'], errors='ignore')
    ds_new = ds_new.sel(latitude=slice(90, 50))
    
    ds_clim_monthly = xr.open_dataset('data/TP91_20_climatology_50N.nc')
    ds_clim_seasonal = xr.open_dataset('data/TP91_20_seasonal_climatology_50N.nc')
    
    # --- 1. Monthly Anomalies ---
    clim_monthly_renamed = ds_clim_monthly.rename({'time': 'month'})
    clim_monthly_renamed['month'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    # Absolute difference
    anom_monthly_diff = ds_new.groupby('time.month') - clim_monthly_renamed
    anom_monthly_diff = anom_monthly_diff.drop_vars('month')
    anom_monthly_diff.tp.attrs['long_name'] = 'Total precipitation monthly anomaly (difference)'
    anom_monthly_diff.tp.attrs['units'] = 'm'
    
    # Fractional difference (in %)
    # 100 * (New - Clim) / Clim
    anom_monthly_frac = (anom_monthly_diff.groupby('time.month') / clim_monthly_renamed) * 100.0
    anom_monthly_frac = anom_monthly_frac.drop_vars('month')
    anom_monthly_frac.tp.attrs['long_name'] = 'Total precipitation monthly anomaly (fractional)'
    anom_monthly_frac.tp.attrs['units'] = '%'
    
    out_monthly_diff = f"data/{output_prefix}_monthly_anom_diff.nc"
    out_monthly_frac = f"data/{output_prefix}_monthly_anom_frac.nc"
    anom_monthly_diff.to_netcdf(out_monthly_diff)
    anom_monthly_frac.to_netcdf(out_monthly_frac)
    print(f"Saved monthly diff to {out_monthly_diff}")
    print(f"Saved monthly frac to {out_monthly_frac}")
    
    # --- 2. Seasonal Anomalies ---
    ds_resampled = ds_new.resample(time='QS-DEC')
    ds_new_seasonal_ts = ds_resampled.mean()
    ds_new_seasonal_count = ds_resampled.count()
    
    # Only keep seasons that have all 3 consecutive months in the input data
    ds_new_seasonal_ts = ds_new_seasonal_ts.where(ds_new_seasonal_count == 3).dropna(dim='time', how='all')
    
    clim_seasonal_renamed = ds_clim_seasonal.rename({'time': 'season'})
    clim_seasonal_renamed['season'] = ['DJF', 'MAM', 'JJA', 'SON']
    
    def get_season_string(month):
        if month in [12, 1, 2]: return 'DJF'
        if month in [3, 4, 5]: return 'MAM'
        if month in [6, 7, 8]: return 'JJA'
        if month in [9, 10, 11]: return 'SON'
        
    seasons_array = [get_season_string(t.month) for t in pd.DatetimeIndex(ds_new_seasonal_ts.time.values)]
    ds_new_seasonal_ts.coords['season'] = ('time', seasons_array)
    
    # Absolute difference
    anom_seasonal_diff = ds_new_seasonal_ts.groupby('season') - clim_seasonal_renamed
    
    # Fractional difference (in %)
    anom_seasonal_frac = (anom_seasonal_diff.groupby('season') / clim_seasonal_renamed) * 100.0
    
    anom_seasonal_diff = anom_seasonal_diff.drop_vars('season')
    anom_seasonal_diff.tp.attrs['long_name'] = 'Total precipitation seasonal anomaly (difference)'
    anom_seasonal_diff.tp.attrs['units'] = 'm'
    
    anom_seasonal_frac = anom_seasonal_frac.drop_vars('season')
    anom_seasonal_frac.tp.attrs['long_name'] = 'Total precipitation seasonal anomaly (fractional)'
    anom_seasonal_frac.tp.attrs['units'] = '%'
    
    out_seasonal_diff = f"data/{output_prefix}_seasonal_anom_diff.nc"
    out_seasonal_frac = f"data/{output_prefix}_seasonal_anom_frac.nc"
    anom_seasonal_diff.to_netcdf(out_seasonal_diff)
    anom_seasonal_frac.to_netcdf(out_seasonal_frac)
    print(f"Saved seasonal diff to {out_seasonal_diff}")
    print(f"Saved seasonal frac to {out_seasonal_frac}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input NetCDF file (e.g. TP_202101_202112.nc)")
    parser.add_argument('--out_prefix', type=str, required=True, help="Prefix for output NC files")
    args = parser.parse_args()
    
    import pandas as pd
    calc_tp_anomalies(args.input, args.out_prefix)
