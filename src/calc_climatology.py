import xarray as xr
import pandas as pd
from datetime import datetime

def calc_climatology(input_file, output_file, var_name, standard_name, long_name, units):
    print(f"Processing {input_file}...")
#    # Load the GRIB data
#    ds = xr.open_dataset(input_file, engine='cfgrib')
    ds = xr.open_dataset(input_file)

    # If the time coordinate is named 'valid_time', rename it to 'time'
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
        
    # Squeeze out any singleton dimensions
    ds = ds.squeeze()
    
    # Keep only the specific variable requested
    ds = ds[[var_name]]
    
    # ERA5 latitudes are usually in descending order (90 to -90)
    # We slice for the region north of 50N
    ds_arctic = ds.sel(latitude=slice(90, 50))
    
    # Calculate monthly averages (climatology) over the 30 years. 
    # This replaces the 'time' dimension with a 'month' dimension (1-12).
    ds_climatology = ds_arctic.groupby('time.month').mean(dim='time')
    
    # Rename 'month' back to 'time' so CDO recognizes it as a time dimension
    ds_climatology = ds_climatology.rename({'month': 'time'})
    
    # Create an actual datetime axis for the climatology (using a nominal year, e.g., 2000)
    # We set the time to the 15th of each month.
    nominal_times = pd.date_range('2000-01-01', periods=12, freq='MS') + pd.Timedelta(days=14)
    ds_climatology = ds_climatology.assign_coords(time=nominal_times)
    
    # Assign CF-compliant attributes to the time axis
    ds_climatology.time.attrs['standard_name'] = 'time'
    ds_climatology.time.attrs['long_name'] = 'time'
    ds_climatology.time.attrs['axis'] = 'T'
    
    # Add CF-compliant attributes to the data variable
    ds_climatology[var_name].attrs['standard_name'] = standard_name
    ds_climatology[var_name].attrs['long_name'] = long_name
    ds_climatology[var_name].attrs['units'] = units
    
    # Ensure spatial coordinates have CF-compliant attributes
    if 'latitude' in ds_climatology.coords:
        ds_climatology.latitude.attrs['standard_name'] = 'latitude'
        ds_climatology.latitude.attrs['long_name'] = 'latitude'
        ds_climatology.latitude.attrs['units'] = 'degrees_north'
        ds_climatology.latitude.attrs['axis'] = 'Y'
        
    if 'longitude' in ds_climatology.coords:
        ds_climatology.longitude.attrs['standard_name'] = 'longitude'
        ds_climatology.longitude.attrs['long_name'] = 'longitude'
        ds_climatology.longitude.attrs['units'] = 'degrees_east'
        ds_climatology.longitude.attrs['axis'] = 'X'
    
    # Add CF-compliant global attributes
    ds_climatology.attrs['Conventions'] = 'CF-1.8'
    ds_climatology.attrs['title'] = 'ERA5 30-year Monthly Climatology (1991-2020) for Arctic Region'
    ds_climatology.attrs['history'] = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using xarray"
    ds_climatology.attrs['source'] = 'ECMWF ERA5 Reanalysis'
    ds_climatology.attrs['institution'] = 'ECMWF'

    # Save the result as a NetCDF file
    # By default, xarray will automatically encode these datetimes into CF-compliant 'days since X' units
    ds_climatology.to_netcdf(output_file)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    # Process Temperature
    calc_climatology(
        input_file='data/T2M91_20.nc', 
        output_file='data/T2M91_20_climatology_50N.nc', 
        var_name='t2m',
        standard_name='air_temperature',
        long_name='2 metre temperature',
        units='K'
    )
    
    # Process Precipitation
    calc_climatology(
        input_file='data/TP91_20.nc', 
        output_file='data/TP91_20_climatology_50N.nc', 
        var_name='tp',
        standard_name='lwe_thickness_of_precipitation_amount',
        long_name='Total precipitation',
        units='m'
    )
    
    print("Done!")
