import gc
import gzip
import os
import shutil

import matplotlib
import numpy as np
import psutil
import pygrib
import requests
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


matplotlib.use("Agg")


DOWNLOAD_URL = "https://mrms.ncep.noaa.gov/2D/MergedBaseReflectivity/MRMS_MergedBaseReflectivity.latest.grib2.gz"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
EXTRACT_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_RETRIES = int(os.getenv("MRMS_DOWNLOAD_RETRIES", "3"))
RETRY_DELAY_SECONDS = float(os.getenv("MRMS_RETRY_DELAY_SECONDS", "2.0"))
MIN_DBZ = 1.0
PNG_EXPORT_SCALE = float(os.getenv("MRMS_PNG_EXPORT_SCALE", "3.0"))

PALETTE = np.array(
    [
        [100, 200, 255, 180],
        [0, 255, 255, 190],
        [0, 200, 0, 200],
        [255, 255, 0, 210],
        [255, 150, 0, 220],
        [255, 0, 0, 230],
        [200, 0, 200, 240],
        [255, 255, 255, 255],
    ],
    dtype=np.uint8,
)
COLOR_BINS = np.array([5, 10, 20, 30, 40, 50, 60, 70], dtype=np.float32)
CONTOUR_LEVELS = np.concatenate(([MIN_DBZ], COLOR_BINS)).astype(np.float32)
CONTOUR_COLORS = [tuple(color / 255.0) for color in PALETTE]


def download_file(url, output_path):
    """Download a file from a URL to a specified output path."""
    temporary_path = f"{output_path}.part"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with open(temporary_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        file_handle.write(chunk)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(f"File downloaded to {output_path}")


def extract_gzip_file(gzip_path, output_path):
    """Extract a .gz file to disk without reading the full file into memory."""
    temporary_path = f"{output_path}.part"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    try:
        with gzip.open(gzip_path, "rb") as source, open(temporary_path, "wb") as destination:
            shutil.copyfileobj(source, destination, length=EXTRACT_CHUNK_SIZE)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(f"Extracted {gzip_path} to {output_path}")


def download_and_extract_with_retries(url, gzip_path, output_path, retries):
    """Download and extract MRMS data, retrying when the latest gzip is truncated."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            download_file(url, gzip_path)
            extract_gzip_file(gzip_path, output_path)
            return
        except (EOFError, gzip.BadGzipFile, OSError, requests.RequestException) as error:
            last_error = error
            print(f"Attempt {attempt} failed: {error}")

            for path in (gzip_path, output_path, f"{gzip_path}.part", f"{output_path}.part"):
                if os.path.exists(path):
                    os.remove(path)

            if attempt == retries:
                break

            print(f"Retrying in {RETRY_DELAY_SECONDS:.1f} seconds...")
            import time
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Unable to download a complete MRMS gzip after {retries} attempts"
    ) from last_error


def safe_grib_value(message, key):
    """Return a GRIB metadata value when available."""
    try:
        return message[key]
    except (KeyError, RuntimeError, ValueError):
        return None


def normalize_longitude(value):
    """Normalize longitudes to the [-180, 180] range for easier inspection."""
    if value is None:
        return None
    return ((float(value) + 180.0) % 360.0) - 180.0


def get_corner_bounds(message):
    """Build corner bounds from GRIB metadata without allocating full lat/lon grids."""
    first_lat = safe_grib_value(message, "latitudeOfFirstGridPointInDegrees")
    first_lon = normalize_longitude(safe_grib_value(message, "longitudeOfFirstGridPointInDegrees"))
    last_lat = safe_grib_value(message, "latitudeOfLastGridPointInDegrees")
    last_lon = normalize_longitude(safe_grib_value(message, "longitudeOfLastGridPointInDegrees"))

    if None in (first_lat, first_lon, last_lat, last_lon):
        return None

    north = max(first_lat, last_lat)
    south = min(first_lat, last_lat)
    west = min(first_lon, last_lon)
    east = max(first_lon, last_lon)

    return {
        "top_left": (north, west),
        "top_right": (north, east),
        "bottom_left": (south, west),
        "bottom_right": (south, east),
    }


def get_latitude_range(message):
    """Return the first and last latitude values from GRIB metadata."""
    first_lat = safe_grib_value(message, "latitudeOfFirstGridPointInDegrees")
    last_lat = safe_grib_value(message, "latitudeOfLastGridPointInDegrees")
    if None in (first_lat, last_lat):
        return None
    return float(first_lat), float(last_lat)


def get_longitude_range(message):
    """Return the first and last longitude values from GRIB metadata."""
    first_lon = normalize_longitude(safe_grib_value(message, "longitudeOfFirstGridPointInDegrees"))
    last_lon = normalize_longitude(safe_grib_value(message, "longitudeOfLastGridPointInDegrees"))
    if None in (first_lon, last_lon):
        return None
    return float(first_lon), float(last_lon)


def mercator_y(latitudes):
    """Convert latitude degrees to normalized Web Mercator Y coordinates."""
    clipped = np.clip(latitudes, -85.05112878, 85.05112878)
    radians = np.deg2rad(clipped)
    return np.log(np.tan((np.pi / 4.0) + (radians / 2.0)))


def render_contourf_png(data, latitude_range, longitude_range, png_file):
    """Render reflectivity data to a transparent PNG using contourf polygons."""
    if latitude_range is None or longitude_range is None:
        raise ValueError("Latitude and longitude ranges are required for contour rendering")

    valid_mask = np.isfinite(data) & (data >= MIN_DBZ)
    if not np.any(valid_mask):
        height, width = data.shape
        dpi = 100
        export_scale = max(1.0, PNG_EXPORT_SCALE)
        figure = plt.figure(
            figsize=((width * export_scale) / dpi, (height * export_scale) / dpi),
            dpi=dpi,
            frameon=False,
        )
        figure.patch.set_alpha(0.0)
        axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axis.set_axis_off()
        axis.set_facecolor((0, 0, 0, 0))
        figure.savefig(png_file, dpi=dpi, transparent=True, bbox_inches=None, pad_inches=0)
        plt.close(figure)
        return

    row_count, column_count = data.shape
    latitudes = np.linspace(latitude_range[0], latitude_range[1], row_count, dtype=np.float32)
    longitudes = np.linspace(longitude_range[0], longitude_range[1], column_count, dtype=np.float32)
    mercator_latitudes = mercator_y(latitudes)

    masked_data = np.ma.masked_invalid(data)
    masked_data = np.ma.masked_less(masked_data, MIN_DBZ)

    dpi = 100
    export_scale = max(1.0, PNG_EXPORT_SCALE)
    figure_width = max(1.0, (column_count * export_scale) / dpi)
    figure_height = max(1.0, (row_count * export_scale) / dpi)
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=dpi, frameon=False)
    figure.patch.set_alpha(0.0)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_axis_off()
    axis.set_facecolor((0, 0, 0, 0))

    cmap = ListedColormap(CONTOUR_COLORS)
    norm = BoundaryNorm(CONTOUR_LEVELS, cmap.N, clip=True)

    axis.contourf(
        longitudes,
        mercator_latitudes,
        masked_data,
        levels=CONTOUR_LEVELS,
        cmap=cmap,
        norm=norm,
        antialiased=True,
        extend="max",
        corner_mask=True,
    )
    axis.set_xlim(float(longitudes[0]), float(longitudes[-1]))
    axis.set_ylim(float(mercator_latitudes.min()), float(mercator_latitudes.max()))

    figure.savefig(png_file, dpi=dpi, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(figure)


def process_grib_to_png(grib_file, png_file):
    """Process a GRIB2 file and save the data as a transparent PNG."""
    log_memory_usage("before opening grib")

    with pygrib.open(grib_file) as grib_messages:
        message = grib_messages.select()[0]

        log_memory_usage("before reading values")
        data = message.values
        log_memory_usage("after reading values")

        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        log_memory_usage("after float32 conversion")

        latitude_range = get_latitude_range(message)
        longitude_range = get_longitude_range(message)
        render_contourf_png(data, latitude_range, longitude_range, png_file)
        del data
        gc.collect()
        log_memory_usage("after contourf render")

        corner_bounds = get_corner_bounds(message)

    print(f"Processed data saved as PNG to {png_file}")
    if corner_bounds is not None:
        print(f"Top Left Corner: {corner_bounds['top_left']}")
        print(f"Top Right Corner: {corner_bounds['top_right']}")
        print(f"Bottom Left Corner: {corner_bounds['bottom_left']}")
        print(f"Bottom Right Corner: {corner_bounds['bottom_right']}")

    gc.collect()
    log_memory_usage("after saving png")


def log_memory_usage(stage):
    """Log memory usage at a specific stage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory usage at {stage}: {memory_mb:.2f} MB")


if __name__ == "__main__":
    grib_gz_file = "MRMS_MergedBaseReflectivity.latest.grib2.gz"
    grib_file = "MRMS_MergedBaseReflectivity.latest.grib2"
    png_file = "MRMS_MergedBaseReflectivity.png"

    download_and_extract_with_retries(DOWNLOAD_URL, grib_gz_file, grib_file, DOWNLOAD_RETRIES)
    process_grib_to_png(grib_file, png_file)

    for temporary_file in (grib_gz_file, grib_file):
        if os.path.exists(temporary_file):
            os.remove(temporary_file)

    gc.collect()
