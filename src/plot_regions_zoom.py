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
import shapely.wkt
from shapely.geometry import LineString

def plot_regional_zooms(input_file):
    print(f"Loading {input_file}...")
    ds = xr.open_dataset(input_file)
    
    is_seasonal = 'seasonal' in input_file.lower()
    is_frac = 'frac' in input_file.lower()
    
    if 't2m' in ds.variables:
        var = 't2m'
        cmap = 'RdBu_r'
    elif 'tp' in ds.variables:
        var = 'tp'
        cmap = 'BrBG'
        if not is_frac:
            attrs = ds[var].attrs
            ds[var] = ds[var] * 1000.0
            ds[var].attrs = attrs
            ds[var].attrs['units'] = 'mm'
    else:
        raise ValueError("Could not find 't2m' or 'tp' in dataset.")

    times = pd.DatetimeIndex(ds.time.values)
    start_str = times[0].strftime("%Y%m")
    end_str = times[-1].strftime("%Y%m")
    
    def get_season(month):
        if month in [12, 1, 2]: return 'DJF'
        if month in [3, 4, 5]: return 'MAM'
        if month in [6, 7, 8]: return 'JJA'
        if month in [9, 10, 11]: return 'SON'
        return f"M{month:02d}"

    # Calculate min/max for the colorbar globally so it is consistent across all plots
    real_vmin = float(ds[var].min())
    real_vmax = float(ds[var].max())
    
    if is_frac:
        vmin = max(-100.0, real_vmin)
        vmax = max(0.1, real_vmax)
        vmin = min(-0.1, vmin)
        norm = mcolors.TwoSlopeNorm(vcenter=0., vmin=vmin, vmax=vmax)
    else:
        vabs = max(abs(real_vmin), abs(real_vmax))
        if vabs == 0: vabs = 0.1
        norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)

    # Load ArcRCC regions from CSV
    region_file = 'data/ArcRCC_regions_20200930.csv'
    regions = []
    
    if os.path.exists(region_file):
        df_regions = pd.read_csv(region_file)
        for _, row in df_regions.iterrows():
            geom_str = row['WKT']
            if geom_str.startswith('POLYGON((') and geom_str.endswith('))'):
                coords_str = geom_str[9:-2].strip()
                coords_list = []
                for c in coords_str.split(','):
                    lon, lat = map(float, c.strip().split())
                    coords_list.append((lon, lat))
                
                line = LineString(coords_list)
                regions.append({
                    'abbrev': row['Abbrev'].strip(),
                    'name': row['name'].strip(),
                    'geom': line
                })
        print(f"Loaded {len(regions)} regions for zoomed plotting.")
    else:
        print("Error: Could not find ArcRCC_regions_20200930.csv")
        return

    # Extract all geometries to draw light background boundaries
    all_geoms = [r['geom'] for r in regions]

    for reg in regions:
        abbrev = reg['abbrev']
        name = reg['name']
        geom = reg['geom']
        
        # Calculate bounding box for this region
        min_lon, min_lat, max_lon, max_lat = geom.bounds
        lon_span = max_lon - min_lon
        
        plot_dir = f"figures/PLOT_{abbrev}_{start_str}_{end_str}"
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Generating plots for {name} in {plot_dir}/")
        
        # Padding in degrees around the region
        pad = 5.0
        ext_min_lon = min_lon - pad
        ext_max_lon = max_lon + pad
        ext_min_lat = max(40, min_lat - pad)
        ext_max_lat = min(90, max_lat + pad)

        for i in range(len(times)):
            data_slice = ds[var].isel(time=i)
            t_val = times[i]
            
            fig = plt.figure(figsize=(10, 8))
            
            # If the region wraps all the way around the pole (like Central Arctic)
            if lon_span > 300:
                central_lon = 0
                ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=central_lon))
                # Set extent to encompass the entire polar region above the minimum latitude
                ax.set_extent([-180, 180, ext_min_lat, 90], crs=ccrs.PlateCarree())
            else:
                central_lon = (min_lon + max_lon) / 2.0
                ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=central_lon))
                # Set extent tightly around the specific region's bounding box
                ax.set_extent([ext_min_lon, ext_max_lon, ext_min_lat, ext_max_lat], crs=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            
            # Draw all boundaries thinly
            ax.add_geometries(all_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.5)
            
            # Draw THIS region's boundary thickly to highlight it
            ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=2.5)

            pcm = data_slice.plot.pcolormesh(
                ax=ax, 
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                add_colorbar=False
            )
            
            # Add Colorbar
            if is_frac:
                pos_max = int(np.ceil(vmax / 50.0) * 50)
                if pos_max <= 100: pos_ticks = [50, 100]
                elif pos_max <= 200: pos_ticks = [100, 200]
                elif pos_max <= 500: pos_ticks = [100, 200, 300, 400, 500]
                else: pos_ticks = list(range(200, pos_max + 200, 200))
                    
                ticks = [-100, -50, 0] + [t for t in pos_ticks if t <= vmax]
                cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, ticks=ticks)
            else:
                cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                
            units = data_slice.attrs.get('units', '')
            if var == 't2m' and units == 'K': units = '°C'
            cbar.set_label(units)
            
            # Generate Title and Filename
            long_name = data_slice.attrs.get('long_name', var)
            if is_frac: long_name = long_name.replace('(fractional)', '(%)')
                
            if is_seasonal:
                season_str = get_season(t_val.month)
                title = f"{name} \n{long_name}\n{season_str} {t_val.year}"
                #title = f"{name} ({abbrev})\n{long_name}\n{season_str} {t_val.year}"
                out_name = f"{var}_{t_val.year}_{season_str}_{abbrev}.png"
            else:
                title = f"{name} \n{long_name}\n{t_val.strftime('%B %Y')}"
                # title = f"{name} ({abbrev}) \n{long_name}\n{t_val.strftime('%B %Y')}"
                out_name = f"{var}_{t_val.strftime('%Y%m')}_{abbrev}.png"
                
            if is_frac: out_name = out_name.replace('.png', '_frac.png')
            elif 'diff' in input_file: out_name = out_name.replace('.png', '_diff.png')
                
            plt.title(title)
            out_path = os.path.join(plot_dir, out_name)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ERA5 Arctic anomalies zoomed to regions.")
    parser.add_argument('--input', type=str, required=True, help="Input anomaly NetCDF file")
    args = parser.parse_args()
    
    plot_regional_zooms(args.input)
