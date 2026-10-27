#!/usr/bin/env python

import cdsapi
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Download ERA5 2m temperature monthly means.")
    parser.add_argument('--year_start', type=int, required=True, help="Start year (e.g., 1991)")
    parser.add_argument('--month_start', type=int, required=True, help="Start month (1-12)")
    parser.add_argument('--year_stop', type=int, required=True, help="Stop year (e.g., 2020)")
    parser.add_argument('--month_stop', type=int, required=True, help="Stop month (1-12)")
    args = parser.parse_args()

    if args.year_start > args.year_stop or (args.year_start == args.year_stop and args.month_start > args.month_stop):
        print("Error: Start date must be before stop date.")
        sys.exit(1)

    years = [str(y) for y in range(args.year_start, args.year_stop + 1)]
    
    # CDS API 'reanalysis-era5-single-levels-monthly-means' takes a list of years and a list of months.
    # It will fetch the cartesian product (all combinations) of the years and months provided.
    # Therefore, if the request spans multiple years, we request all 12 months for those years 
    # to ensure the continuous range is fully covered. You can slice the exact start/stop months later.
    if args.year_start == args.year_stop:
        months = [f"{m:02d}" for m in range(args.month_start, args.month_stop + 1)]
    else:
        print(f"Warning: Fetching across multiple years ({args.year_start}-{args.year_stop}).")
        print("CDS API fetches the cartesian product of years and months.")
        print("To capture your continuous date range, all 12 months for these years will be downloaded.")
        months = [f"{m:02d}" for m in range(1, 13)]

    # Generate a descriptive output filename
    output_file = f"data/T2M_{args.year_start}{args.month_start:02d}_{args.year_stop}{args.month_stop:02d}.nc"
    
    print(f"Requesting data from CDS API...")
    print(f"Years: {years}")
    print(f"Months: {months}")
    print(f"Output will be saved to: {output_file}")
    
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': ['2m_temperature'],
            'year': years,
            'month': months,
            'time': '00:00',
        },
        output_file)
    print(f"Successfully downloaded {output_file}")

if __name__ == "__main__":
    main()
