import xarray as xr
import pandas as pd
from datetime import datetime

def calc_seasonal_climatology(input_file, output_file, var_name):
    print(f"Processing {input_file}...")
    # Load the already-computed monthly NetCDF climatology
    ds = xr.open_dataset(input_file)
    
    # Calculate seasonal averages directly from the 12 months.
    # We pass keep_attrs=True to retain the CF-compliant metadata from the monthly file
    ds_climatology = ds.groupby('time.season').mean(dim='time', keep_attrs=True)
    
    # DJF -> Jan, MAM -> Apr, JJA -> Jul, SON -> Oct
    season_to_month = {
        'DJF': 1,
        'MAM': 4,
        'JJA': 7,
        'SON': 10
    }
    
    # Extract the season labels
    seasons = ds_climatology.season.values
    
    # Create corresponding nominal datetimes (e.g., 15th of the representative month in year 2000)
    nominal_times = [pd.Timestamp(year=2000, month=season_to_month[s], day=15) for s in seasons]
    
    # Rename 'season' dimension to 'time'
    ds_climatology = ds_climatology.rename({'season': 'time'})
    
    # Assign the new datetime coordinates
    ds_climatology = ds_climatology.assign_coords(time=nominal_times)
    
    # Sort chronologically (Jan, Apr, Jul, Oct)
    ds_climatology = ds_climatology.sortby('time')
    
    # Re-apply CF-compliant attributes to the new time axis
    ds_climatology.time.attrs['standard_name'] = 'time'
    ds_climatology.time.attrs['long_name'] = 'time'
    ds_climatology.time.attrs['axis'] = 'T'
    
    # Update title in global attributes
    ds_climatology.attrs['title'] = 'ERA5 30-year Seasonal Climatology (1991-2020) for Arctic Region'
    ds_climatology.attrs['history'] = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using xarray from monthly climatology"

    # Save the result as a NetCDF file
    ds_climatology.to_netcdf(output_file)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    # Process Temperature
    calc_seasonal_climatology(
        input_file='data/T2M91_20_climatology_50N.nc', 
        output_file='data/T2M91_20_seasonal_climatology_50N.nc', 
        var_name='t2m'
    )
    
    # Process Precipitation
    calc_seasonal_climatology(
        input_file='data/TP91_20_climatology_50N.nc', 
        output_file='data/TP91_20_seasonal_climatology_50N.nc', 
        var_name='tp'
    )
    
    print("Done!")
