import gc
import gzip
import os
import shutil

import numpy as np
import psutil
import pygrib
import requests
from PIL import Image


DOWNLOAD_URL = "https://mrms.ncep.noaa.gov/2D/MergedBaseReflectivity/MRMS_MergedBaseReflectivity.latest.grib2.gz"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
EXTRACT_CHUNK_SIZE = 1024 * 1024
MIN_DBZ = 1.0
MAX_OUTPUT_DIMENSION = int(os.getenv("MRMS_MAX_OUTPUT_DIMENSION", "1400"))
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

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


def download_file(url, output_path):
    """Download a file from a URL to a specified output path."""
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    file_handle.write(chunk)
    print(f"File downloaded to {output_path}")


def extract_gzip_file(gzip_path, output_path):
    """Extract a .gz file to disk without reading the full file into memory."""
    with gzip.open(gzip_path, "rb") as source, open(output_path, "wb") as destination:
        shutil.copyfileobj(source, destination, length=EXTRACT_CHUNK_SIZE)
    print(f"Extracted {gzip_path} to {output_path}")


def resize_data_smoothly(data, max_dimension):
    """Reduce array size with bilinear interpolation to avoid blocky artifacts."""
    if max_dimension <= 0:
        return data

    height, width = data.shape
    max_side = max(height, width)
    if max_side <= max_dimension:
        return data

    scale = max_dimension / float(max_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    print(f"Resizing data from {width}x{height} to {new_width}x{new_height}")

    finite_mask = np.isfinite(data)
    safe_data = np.where(finite_mask, data, 0.0).astype(np.float32, copy=False)
    mask_data = finite_mask.astype(np.float32, copy=False)

    resized_data = np.array(
        Image.fromarray(safe_data, mode="F").resize((new_width, new_height), RESAMPLE_BILINEAR),
        dtype=np.float32,
    )
    resized_mask = np.array(
        Image.fromarray(mask_data, mode="F").resize((new_width, new_height), RESAMPLE_BILINEAR),
        dtype=np.float32,
    )

    resized_data[resized_mask < 0.01] = np.nan
    return resized_data


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


def mercator_y(latitudes):
    """Convert latitude degrees to normalized Web Mercator Y coordinates."""
    clipped = np.clip(latitudes, -85.05112878, 85.05112878)
    radians = np.deg2rad(clipped)
    return np.log(np.tan((np.pi / 4.0) + (radians / 2.0)))


def reproject_data_to_mercator(data, latitude_range):
    """Reproject regular lat/lon gridded data into Mercator space with row interpolation."""
    if latitude_range is None:
        return data

    first_lat, last_lat = latitude_range
    row_count = data.shape[0]
    if row_count <= 1:
        return data

    source_lats = np.linspace(first_lat, last_lat, row_count, dtype=np.float32)
    source_y = mercator_y(source_lats)

    if np.allclose(source_y[0], source_y[-1]):
        return data

    target_y = np.linspace(source_y[0], source_y[-1], row_count, dtype=np.float32)
    if source_y[0] <= source_y[-1]:
        source_axis = source_y
        source_rows = np.arange(row_count, dtype=np.float32)
    else:
        source_axis = source_y[::-1]
        source_rows = np.arange(row_count - 1, -1, -1, dtype=np.float32)

    row_positions = np.interp(target_y, source_axis, source_rows)
    lower_indexes = np.floor(row_positions).astype(np.int32)
    upper_indexes = np.clip(lower_indexes + 1, 0, row_count - 1)
    lower_indexes = np.clip(lower_indexes, 0, row_count - 1)

    upper_weight = (row_positions - lower_indexes).astype(np.float32)[:, None]
    lower_weight = 1.0 - upper_weight

    finite_mask = np.isfinite(data)
    safe_data = np.where(finite_mask, data, 0.0).astype(np.float32, copy=False)
    valid_mask = finite_mask.astype(np.float32, copy=False)

    lower_values = safe_data[lower_indexes, :]
    upper_values = safe_data[upper_indexes, :]
    lower_valid = valid_mask[lower_indexes, :]
    upper_valid = valid_mask[upper_indexes, :]

    total_weight = (lower_valid * lower_weight) + (upper_valid * upper_weight)
    blended = (lower_values * lower_valid * lower_weight) + (upper_values * upper_valid * upper_weight)

    result = np.full_like(safe_data, np.nan, dtype=np.float32)
    np.divide(blended, total_weight, out=result, where=total_weight > 0.0)
    return result


def to_rgba(data):
    """Convert reflectivity values into an RGBA image array."""
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    valid_mask = np.isfinite(data) & (data >= MIN_DBZ)
    if not np.any(valid_mask):
        return rgba

    color_indexes = np.digitize(data[valid_mask], COLOR_BINS, right=False)
    rgba[valid_mask] = PALETTE[color_indexes]
    return rgba


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
        data = resize_data_smoothly(data, MAX_OUTPUT_DIMENSION)
        log_memory_usage("after smooth resize")

        data = reproject_data_to_mercator(data, latitude_range)
        gc.collect()
        log_memory_usage("after mercator reprojection")

        rgba = to_rgba(data)
        del data
        gc.collect()
        log_memory_usage("after rgba conversion")

        image = Image.fromarray(rgba, mode="RGBA")
        image.save(png_file, optimize=True)
        image.close()

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

    download_file(DOWNLOAD_URL, grib_gz_file)
    extract_gzip_file(grib_gz_file, grib_file)
    process_grib_to_png(grib_file, png_file)

    for temporary_file in (grib_gz_file, grib_file):
        if os.path.exists(temporary_file):
            os.remove(temporary_file)

    gc.collect()
