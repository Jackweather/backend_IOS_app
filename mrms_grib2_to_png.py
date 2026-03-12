import requests
import pygrib
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import gc  # Import garbage collector
import psutil  # For monitoring memory usage
import time  # Import time module for adding delays

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
    plt.figure(figsize=(6, 4), dpi=150)  # Reduced figure size and DPI for lower memory usage
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    # Process data in smaller chunks
    chunk_size = 500  # Process smaller chunks to reduce memory usage
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i:i + chunk_size]
        chunk = gaussian_filter(chunk, sigma=1)  # Apply Gaussian filter to the chunk
        data[i:i + chunk_size] = chunk  # Replace the chunk in the original data
        del chunk  # Free memory after processing each chunk
        gc.collect()
        log_memory_usage(f"after processing chunk {i // chunk_size + 1}")
        time.sleep(5)  # Add a 1-second delay before processing the next chunk

    # Mask out values less than 1 dBZ after processing
    data = np.ma.masked_less(data, 1)

    # Plot the data without map features or colorbar
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), antialiased=True)  # Ensure anti-aliasing

    # Remove the axes and set a fully transparent background
    ax.axis('off')
    plt.gca().patch.set_alpha(0)

    # Save the figure as a PNG with transparency
    plt.savefig(png_file, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)  # Reduced DPI for lower memory usage
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

def log_memory_usage(stage):
    """Log memory usage at a specific stage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"Memory usage at {stage}: {mem:.2f} MB")

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
