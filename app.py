from flask import Flask, send_from_directory, render_template
import os
import subprocess
import threading

app = Flask(__name__)

# Function to run the script
def run_grib_to_png():
    subprocess.run(['python', 'mrms_grib2_to_png.py'], check=True)

# Function to run scripts
def run_scripts(scripts, delay):
    import time
    for script, cwd in scripts:
        subprocess.run(['python', script], cwd=cwd, check=True)
        time.sleep(delay)

# Route to serve the index.html
@app.route('/')
def serve_index():
    return render_template('index.html')

# Route to serve the radar image file
@app.route('/MRMS_MergedBaseReflectivity.png')
def serve_radar_image():
    return send_from_directory(os.getcwd(), 'MRMS_MergedBaseReflectivity.png', as_attachment=False)

@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/mrms_grib2_to_png.py", "/opt/render/project/src"),
       
    ]
    threading.Thread(target=lambda: run_scripts(scripts, 1)).start()
    return "Task started in background! Check logs folder for output.", 200

if __name__ == '__main__':
    app.run(debug=True)
