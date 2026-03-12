from flask import Flask, send_from_directory, render_template
import os
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess

app = Flask(__name__)

# Function to run the script
def run_grib_to_png():
    subprocess.run(['python', 'mrms_grib2_to_png.py'], check=True)

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_grib_to_png, 'interval', minutes=5)
scheduler.start()

# Route to serve the index.html
@app.route('/')
def serve_index():
    return render_template('index.html')

# Route to serve the radar image file
@app.route('/MRMS_MergedBaseReflectivity.png')
def serve_radar_image():
    return send_from_directory(os.getcwd(), 'MRMS_MergedBaseReflectivity.png', as_attachment=False)

# Ensure the scheduler shuts down properly when the app exits
import atexit
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(debug=True)