#!/usr/bin/env python

import argparse
import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as mcolors

def plot_anomalies(input_file):
    print(f"Loading {input_file}...")
    ds = xr.open_dataset(input_file)
    
    # Determine if this is a seasonal or monthly file based on filename
    is_seasonal = 'seasonal' in input_file.lower()
    is_frac = 'frac' in input_file.lower()
    
    # Determine which variable we are plotting
    if 't2m' in ds.variables:
        var = 't2m'
        cmap = 'RdBu_r'  # Red-Blue diverging colormap (reversed so red is warm, blue is cold)
    elif 'tp' in ds.variables:
        var = 'tp'
        cmap = 'BrBG'    # Brown-Green diverging colormap (brown is dry, green is wet)
        if not is_frac:
            # Convert precipitation difference from meters to millimeters
            # Save attributes since operations drop them sometimes
            attrs = ds[var].attrs
            ds[var] = ds[var] * 1000.0
            ds[var].attrs = attrs
            ds[var].attrs['units'] = 'mm'
    else:
        raise ValueError("Could not find 't2m' or 'tp' in dataset.")

    # Figure out the time range for the directory name
    times = pd.DatetimeIndex(ds.time.values)
    start_str = times[0].strftime("%Y%m")
    end_str = times[-1].strftime("%Y%m")
    
    # Create the PLOT directory
    plot_dir = f"figures/PLOT_{start_str}_{end_str}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Output directory: {plot_dir}/")
    
    # Helper to map the nominal month to the season string
    def get_season(month):
        if month in [12, 1, 2]: return 'DJF'
        if month in [3, 4, 5]: return 'MAM'
        if month in [6, 7, 8]: return 'JJA'
        if month in [9, 10, 11]: return 'SON'
        return f"M{month:02d}"

    # Calculate min/max for the colorbar
    real_vmin = float(ds[var].min())
    real_vmax = float(ds[var].max())
    
    if is_frac:
        # Cap the minimum at -100, allow maximum to be as high as needed
        vmin = max(-100.0, real_vmin)
        vmax = max(0.1, real_vmax)
        vmin = min(-0.1, vmin) # ensure vmin is negative for TwoSlopeNorm to work
        norm = mcolors.TwoSlopeNorm(vcenter=0., vmin=vmin, vmax=vmax)
    else:
        # Symmetric limits for differences so 0 is in the middle of the colormap
        vabs = max(abs(real_vmin), abs(real_vmax))
        if vabs == 0: vabs = 0.1
        norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)

    for i in range(len(times)):
        data_slice = ds[var].isel(time=i)
        t_val = times[i]
        
        # Setup Figure and Cartopy North Polar Stereographic projection
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
        
        # Set extent to Arctic (90N down to 50N)
        ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
        
        # Add coastlines and gridlines
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                          
        # Plot the data
        pcm = data_slice.plot.pcolormesh(
            ax=ax, 
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False
        )
        
        # Add Colorbar
        if is_frac:
            # Force explicit ticks for fractional plots
            pos_max = int(np.ceil(vmax / 50.0) * 50)
            if pos_max <= 100:
                pos_ticks = [50, 100]
            elif pos_max <= 200:
                pos_ticks = [100, 200]
            elif pos_max <= 500:
                pos_ticks = [100, 200, 300, 400, 500]
            else:
                pos_ticks = list(range(200, pos_max + 200, 200))
                
            ticks = [-100, -50, 0] + [t for t in pos_ticks if t <= vmax]
            cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, ticks=ticks)
        else:
            cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            
        units = data_slice.attrs.get('units', '')
        
        # Since this is an anomaly (difference), 1K difference = 1°C difference
        if var == 't2m' and units == 'K':
            units = '°C'
            
        cbar.set_label(units)
        
        # Generate Title and Filename
        long_name = data_slice.attrs.get('long_name', var)
        if is_frac:
            long_name = long_name.replace('(fractional)', '(%)')
            
        if is_seasonal:
            season_str = get_season(t_val.month)
            title = f"{long_name}\n{season_str} {t_val.year}"
            
            # Create a descriptive filename
            out_name = f"{var}_{t_val.year}_{season_str}.png"
        else:
            # %B gives the full English month name (e.g., "June")
            title = f"{long_name}\n{t_val.strftime('%B %Y')}"
            out_name = f"{var}_{t_val.strftime('%Y%m')}.png"
            
        # Differentiate between difference and fractional anomalies in the filename
        if is_frac:
            out_name = out_name.replace('.png', '_frac.png')
        elif 'diff' in input_file:
            out_name = out_name.replace('.png', '_diff.png')
            
        plt.title(title)
        
        out_path = os.path.join(plot_dir, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ERA5 Arctic anomalies.")
    parser.add_argument('--input', type=str, required=True, help="Input anomaly NetCDF file")
    args = parser.parse_args()
    
    plot_anomalies(args.input)
