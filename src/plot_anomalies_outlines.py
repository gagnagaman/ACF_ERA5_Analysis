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
from shapely.geometry import Polygon, LineString, LinearRing

def plot_anomalies(input_file):
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
    plot_dir = f"figures/PLOT_{start_str}_{end_str}"
    os.makedirs(plot_dir, exist_ok=True)
    
    def get_season(month):
        if month in [12, 1, 2]: return 'DJF'
        if month in [3, 4, 5]: return 'MAM'
        if month in [6, 7, 8]: return 'JJA'
        if month in [9, 10, 11]: return 'SON'
        return f"M{month:02d}"

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
    arc_regions = []
    
    # In cartopy, drawing a Polygon will draw all of its edges.
    # Because some of your regions bound the North Pole (latitude 90), but the WKT stops at latitude 85 or 73,
    # simply closing the polygon creates a straight line cutting across the map from the last coordinate 
    # to the first coordinate.
    # To fix this, instead of plotting Polygons, we will plot them as LineStrings (just the lines connecting the points).
    # Since we don't forcefully close the loop with the first point, that artificial line across the Arctic won't be drawn!
    
    if os.path.exists(region_file):
        try:
            df_regions = pd.read_csv(region_file)
            for _, row in df_regions.iterrows():
                geom_str = row['WKT']
                if geom_str.startswith('POLYGON((') and geom_str.endswith('))'):
                    # Extract the inner coordinates
                    coords_str = geom_str[9:-2].strip()
                    # Convert to a list of (lon, lat) floats
                    coords_list = []
                    for c in coords_str.split(','):
                        lon, lat = map(float, c.strip().split())
                        coords_list.append((lon, lat))
                    
                    # Create a LineString instead of a Polygon
                    line = LineString(coords_list)
                    arc_regions.append(line)
                else:
                    geom = shapely.wkt.loads(geom_str)
                    arc_regions.append(geom)
            print(f"Loaded {len(arc_regions)} ArcRCC regions as open boundaries.")
        except Exception as e:
            print(f"Warning: Could not load ArcRCC regions: {e}")

    for i in range(len(times)):
        data_slice = ds[var].isel(time=i)
        t_val = times[i]
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                          
        # Overlay ArcRCC boundaries (now as open lines)
        if arc_regions:
            ax.add_geometries(arc_regions, crs=ccrs.PlateCarree(), 
                              facecolor='none', edgecolor='black', linewidth=1.5)

        pcm = data_slice.plot.pcolormesh(
            ax=ax, 
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False
        )
        
        if is_frac:
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
        
        if var == 't2m' and units == 'K':
            units = '°C'
            
        cbar.set_label(units)
        
        long_name = data_slice.attrs.get('long_name', var)
        if is_frac:
            long_name = long_name.replace('(fractional)', '(%)')
            
        if is_seasonal:
            season_str = get_season(t_val.month)
            title = f"{long_name}\n{season_str} {t_val.year}"
            out_name = f"{var}_{t_val.year}_{season_str}.png"
        else:
            title = f"{long_name}\n{t_val.strftime('%B %Y')}"
            out_name = f"{var}_{t_val.strftime('%Y%m')}.png"
            
        if is_frac:
            out_name = out_name.replace('.png', '_frac_subregions.png')
        elif 'diff' in input_file:
            out_name = out_name.replace('.png', '_diff_subregions.png')
        else:
            out_name = out_name.replace('.png', '_subregions.png')
            
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
