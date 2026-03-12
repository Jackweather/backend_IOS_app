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


def downsample_stride(data, max_dimension):
    """Reduce array size using striding to avoid large intermediate allocations."""
    if max_dimension <= 0:
        return data

    max_side = max(data.shape)
    step = max(1, int(np.ceil(max_side / max_dimension)))
    if step == 1:
        return data

    print(f"Downsampling by factor {step} to reduce memory usage")
    return data[::step, ::step]


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

        data = downsample_stride(data, MAX_OUTPUT_DIMENSION)
        log_memory_usage("after downsampling")

        rgba = to_rgba(data)
        del data
        gc.collect()
        log_memory_usage("after rgba conversion")

        image = Image.fromarray(rgba, mode="RGBA")
        image.save(png_file, optimize=True)
        image.close()

        first_lat = safe_grib_value(message, "latitudeOfFirstGridPointInDegrees")
        first_lon = normalize_longitude(safe_grib_value(message, "longitudeOfFirstGridPointInDegrees"))
        last_lat = safe_grib_value(message, "latitudeOfLastGridPointInDegrees")
        last_lon = normalize_longitude(safe_grib_value(message, "longitudeOfLastGridPointInDegrees"))

    print(f"Processed data saved as PNG to {png_file}")
    if None not in (first_lat, first_lon, last_lat, last_lon):
        print(f"Top Left Corner: ({first_lat}, {first_lon})")
        print(f"Bottom Right Corner: ({last_lat}, {last_lon})")

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
