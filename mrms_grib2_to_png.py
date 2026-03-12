import requests
import pygrib
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import gc  # Import garbage collector

def download_file(url, output_path):
    """Download a file from a URL to a specified output path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded to {output_path}")
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

def process_grib_to_png(grib_file, png_file):
    """Process a GRIB2 file and save the data as a PNG image with a Web Mercator projection for Mapbox compatibility."""
    grbs = pygrib.open(grib_file)
    grb = grbs.select()[0]  # Select the first GRIB message

    data, lats, lons = grb.data()

    # Apply a Gaussian filter to smooth the data
    data = gaussian_filter(data, sigma=1)  # Adjust sigma for the desired level of smoothing

    # Mask out values less than 1 dBZ after smoothing
    data = np.ma.masked_less(data, 1)

    # Define the colormap and normalization
    colors = [
        (100/255, 200/255, 255/255),  # 5 dBZ
        (0, 255/255, 255/255),        # 10 dBZ
        (0, 200/255, 0),              # 20 dBZ
        (255/255, 255/255, 0),        # 30 dBZ
        (255/255, 150/255, 0),        # 40 dBZ
        (255/255, 0, 0),              # 50 dBZ
        (200/255, 0, 200/255),        # 60 dBZ
        (255/255, 255/255, 255/255)   # 70 dBZ
    ]
    levels = [1, 5, 10, 20, 30, 40, 50, 60, 70]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Create the Web Mercator projection for Mapbox compatibility
    plt.figure(figsize=(10, 8), dpi=150)  # Reduced figure size and DPI for lower memory usage
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    # Plot the data without map features or colorbar
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), antialiased=True)  # Ensure anti-aliasing

    # Remove the axes and set a fully transparent background
    ax.axis('off')
    plt.gca().patch.set_alpha(0)

    # Save the figure as a PNG with transparency
    plt.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)  # High DPI and optimized bounding box
    plt.close()

    # Calculate and print the corner bounds
    top_left = (lats.max(), lons.min())
    bottom_left = (lats.min(), lons.min())
    top_right = (lats.max(), lons.max())
    bottom_right = (lats.min(), lons.max())

    print(f"Processed data saved as PNG to {png_file}")
    print(f"Top Left Corner: {top_left}")
    print(f"Bottom Left Corner: {bottom_left}")
    print(f"Top Right Corner: {top_right}")
    print(f"Bottom Right Corner: {bottom_right}")

    # Explicitly delete large variables after use
    del data, lats, lons, grb, grbs
    gc.collect()  # Force garbage collection

if __name__ == "__main__":
    # URL of the GRIB2 file
    url = "https://mrms.ncep.noaa.gov/2D/MergedBaseReflectivity/MRMS_MergedBaseReflectivity.latest.grib2.gz"
    grib_gz_file = "MRMS_MergedBaseReflectivity.latest.grib2.gz"
    grib_file = "MRMS_MergedBaseReflectivity.latest.grib2"
    png_file = "MRMS_MergedBaseReflectivity.png"

    # Download the file
    download_file(url, grib_gz_file)

    # Extract the .gz file
    import gzip
    with gzip.open(grib_gz_file, 'rb') as f_in:
        with open(grib_file, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"Extracted {grib_gz_file} to {grib_file}")

    # Process the GRIB2 file and save as a PNG
    process_grib_to_png(grib_file, png_file)

    # Free memory after downloading the file
    gc.collect()

    # Free memory after processing the GRIB2 data
    plt.close()
    gc.collect()

    # Free memory after cleaning up temporary files
    os.remove(grib_gz_file)
    os.remove(grib_file)
    gc.collect()
