#!/usr/bin/env python

import argparse
import xarray as xr
import sys
import os

def calc_t2m_anomalies(input_file, output_prefix):
    print(f"Processing Temperature anomalies for {input_file}...")
    
    # 1. Load the new data and the climatologies
    ds_new = xr.open_dataset(input_file).squeeze()
    
    if 'valid_time' in ds_new.coords:
        ds_new = ds_new.rename({'valid_time': 'time'})
        
    ds_new = ds_new[['t2m']]
    # Drop unwanted coordinates
    ds_new = ds_new.drop_vars(['number', 'step', 'surface'], errors='ignore')
    # Slice to same Arctic region as climatologies
    ds_new = ds_new.sel(latitude=slice(90, 50))
    
    ds_clim_monthly = xr.open_dataset('data/T2M91_20_climatology_50N.nc')
    ds_clim_seasonal = xr.open_dataset('data/T2M91_20_seasonal_climatology_50N.nc')
    
    # 2. Monthly Anomalies (Difference)
    # xarray's groupby allows subtracting the 12-month climatology from the full time series
    # by matching the 'month' of each time step to the 'time' dimension in ds_clim_monthly
    # (Since we renamed 'month' to 'time' in climatology, we need to match it properly)
    
    # To subtract, we can group the new data by month, but since our climatology has a 'time' axis,
    # it's easiest to rename the climatology's 'time' axis back to 'month' temporarily to let xarray broadcast.
    clim_monthly_renamed = ds_clim_monthly.rename({'time': 'month'})
    clim_monthly_renamed['month'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    anom_monthly = ds_new.groupby('time.month') - clim_monthly_renamed
    anom_monthly = anom_monthly.drop_vars('month') # Clean up coordinate
    anom_monthly.t2m.attrs['long_name'] = '2 metre temperature monthly anomaly'
    anom_monthly.t2m.attrs['units'] = 'K'
    
    out_monthly = f"data/{output_prefix}_monthly_anom.nc"
    anom_monthly.to_netcdf(out_monthly)
    print(f"Saved monthly anomalies to {out_monthly}")
    
    # 3. Seasonal Anomalies (Difference)
    # First, compute the seasonal time series of the new data (resample by quarter starting in Dec)
    # QS-DEC creates quarters: DJF, MAM, JJA, SON
    ds_resampled = ds_new.resample(time='QS-DEC')
    ds_new_seasonal_ts = ds_resampled.mean()
    ds_new_seasonal_count = ds_resampled.count()
    
    # Only keep seasons that have all 3 consecutive months in the input data
    ds_new_seasonal_ts = ds_new_seasonal_ts.where(ds_new_seasonal_count == 3).dropna(dim='time', how='all')
    
    # Rename climatology's time axis back to 'season' for broadcasting
    clim_seasonal_renamed = ds_clim_seasonal.rename({'time': 'season'})
    clim_seasonal_renamed['season'] = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Match the new seasonal time series to the climatology seasons
    def get_season_string(month):
        if month in [12, 1, 2]: return 'DJF'
        if month in [3, 4, 5]: return 'MAM'
        if month in [6, 7, 8]: return 'JJA'
        if month in [9, 10, 11]: return 'SON'
        
    seasons_array = [get_season_string(t.month) for t in pd.DatetimeIndex(ds_new_seasonal_ts.time.values)]
    ds_new_seasonal_ts.coords['season'] = ('time', seasons_array)
    
    anom_seasonal = ds_new_seasonal_ts.groupby('season') - clim_seasonal_renamed
    anom_seasonal = anom_seasonal.drop_vars('season')
    anom_seasonal.t2m.attrs['long_name'] = '2 metre temperature seasonal anomaly'
    anom_seasonal.t2m.attrs['units'] = 'K'
    anom_seaonal = anom_seasonal.squeeze()

    out_seasonal = f"data/{output_prefix}_seasonal_anom.nc"
    anom_seasonal.to_netcdf(out_seasonal)
    print(f"Saved seasonal anomalies to {out_seasonal}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input NetCDF file (e.g. T2M_202101_202112.nc)")
    parser.add_argument('--out_prefix', type=str, required=True, help="Prefix for output NC files")
    args = parser.parse_args()
    
    import pandas as pd
    calc_t2m_anomalies(args.input, args.out_prefix)
