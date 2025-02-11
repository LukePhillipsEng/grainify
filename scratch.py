from flask import Flask, render_template, request, flash, redirect, send_file, jsonify, url_for
import os
import cv2
import math
import numpy as np
from PIL import Image
import io
import time
import zipfile
from werkzeug.utils import secure_filename
# Store last processed parameters to prevent redundant calculations
last_params = {}
app = Flask('__main__')
app.secret_key = 'X'  # Required for flashing messages



# Folder to store uploaded and processed images
UPLOAD_FOLDER = os.path.join('static', 'images')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create and store material data for each format, and define function to solve
def solve_hall_petch(y, k, diameter): # y = original yield stress, k = material constant
    YieldStrength = (y + (k/(math.sqrt(diameter/1000000))))
    return YieldStrength

@app.route('/')
def index():
    return render_template('index.html', traced_images=None)

#Download function for all user input
@app.route('/download')
def download():
    material = request.args.get('material', 'unknown')

    # Create a ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.startswith(('traced_', 'combined_')) or filename in request.args.getlist('traced_images'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                zip_file.write(file_path, filename)


    zip_buffer.seek(0)

    # Return the ZIP file as a response
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{material}_traced_images.zip"
    )
#Updating image from input or sliders
@app.route('/update_image', methods=['POST'])
def update_image():
    global last_params


    try:
        # Retrieve parameters from the sliders
        conversionvalue = float(request.form.get('conversion', 1.0))
        sigma = float(request.form.get('sigma', 1.0))
        threshold = int(request.form.get('threshold', 50))
        high_threshold = int(request.form.get('high_threshold', 150))

        material = request.form.get('material', 'Steel')

        new_params = {
            "conversion": conversionvalue,
            "sigma": sigma,
            "threshold": threshold,
            "high_threshold": high_threshold,
            "material": material

        }

        # Check if parameters have actually changed
        if new_params == last_params:
            return "", 204  # No update needed, prevents unnecessary processing

        last_params = new_params  # Store new params

        # Find the latest uploaded image
        image_files = sorted(
            [f for f in os.listdir(UPLOAD_FOLDER) if not f.startswith(('traced_', 'combined_'))],
            key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)),  # Sort by latest modified time
            reverse=True
        )

        if not image_files:
            return jsonify({"error": "No image found"}), 400

        latest_image = image_files[0]
        image_path = os.path.join(UPLOAD_FOLDER, latest_image)

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to load image"}), 400

        height, width = image.shape[:2]  # Image dimensions
        length_middle = math.sqrt((height ** 2) + (width ** 2))  # Diagonal length

        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        edges = cv2.Canny(blurred, threshold, high_threshold)

        # Create transparent traced image
        traced_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGRA)
        traced_image[:, :, 3] = 0  # Fully transparent background
        traced_image[edges > 0] = [0, 0, 255, 255]  # Red edges, fully opaque

        # Save traced image
        traced_filename = f"traced_{latest_image}"
        traced_image_path = os.path.join(UPLOAD_FOLDER, traced_filename)
        cv2.imwrite(traced_image_path, traced_image)

        # Overlay traced image on original
        original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if original_image.shape[2] == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)  # Ensure alpha channel

        combined_image = original_image.copy()
        combined_image[traced_image[:, :, 3] == 255] = traced_image[traced_image[:, :, 3] == 255]

        # Save combined image
        combined_filename = f"combined_{latest_image}"
        combined_image_path = os.path.join(UPLOAD_FOLDER, combined_filename)
        cv2.imwrite(combined_image_path, combined_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Save with no compression

        # Debug: Check if the image was saved
        if os.path.exists(combined_image_path):
            print(f"Combined image saved successfully: {combined_image_path}")
        else:
            print(f"Failed to save combined image: {combined_image_path}")

        # form = MyForm()
        # if form.validate_on_submit():
             #   selected_option = form.dropdown.data
             #   if selected_option == 'Steel':
                  #  material_variable = 42
             #   else:
                   # material_variable = 10
               # return 0

        # print(material_variable)


        _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        intersection_count = 0
        diagonal_line = np.array([[0, 0], [width, height]], dtype=np.int32)
        for contour in contours:
            bounding_rect = cv2.boundingRect(contour)  # Get bounding box
            intersects, _, _ = cv2.clipLine(bounding_rect, tuple(diagonal_line[0]), tuple(diagonal_line[1]))
            if intersects:
                intersection_count += 1
        grain_avg_size = (length_middle*float(conversionvalue) / intersection_count) if length_middle > 0 else 0
        if material == "Steel":
            hall_petch = solve_hall_petch(125, .6, grain_avg_size) # Steel inputs, needs to be simplified into array math TO DO
        elif material == "Aluminum":
            hall_petch = solve_hall_petch(15, .007, grain_avg_size)
        elif material == "Silver":
            hall_petch = solve_hall_petch(10,18, grain_avg_size)
        else:
            hall_petch = "Unresolved Material"

        print(length_middle)
        print(hall_petch)

        # Generate cache-busting URL for new image
        combined_image_url = url_for('static', filename=f'images/{combined_filename}', _external=True) + f"?t={int(time.time())}"

        return jsonify({
            "combined_image_url": combined_image_url,
            "grain_avg_size": grain_avg_size,
            "hall_petch": hall_petch
        })

    except Exception as e:
        print("Error in /update_image:", str(e))  # Debugging
        return jsonify({"error": str(e)}), 500


#Clear Data
@app.route('/clear')
def clear():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    return redirect('/')

#Trace math
@app.route('/analyze', methods=['POST'])
def analyze():
    # Clear the upload folder before processing new files
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    if 'individual-files' not in request.files and 'folder' not in request.files:
        flash('No files uploaded', 'error')
        return redirect('/')

    individual_files = request.files.getlist('individual-files')
    folder_files = request.files.getlist('folder')
    files = individual_files + folder_files

    if not files or all(file.filename == '' for file in files):
        flash('No files selected', 'error')
        return redirect('/')

    try:
        # Get user-defined parameters
        material = request.form.get('material', 'unknown')
        structure = request.form.get('structure', 'unknown')
        scale_micrometers = request.form.get('scale_micrometers', 'unknown')
        sigma = float(request.form.get('sigma', 1.0))
        threshold = int(request.form.get('threshold', 50))
        high_threshold = int(request.form.get('high_threshold', 150))

        traced_images = []

        for file in files:
            if file.filename == '':
                continue

            # Check if the file is an image
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                flash(f'Skipped non-image file: {file.filename}', 'warning')
                continue

            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)

            if not os.path.exists(image_path):
                file.save(image_path)

            # Load the image using OpenCV
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                flash(f'Failed to process image: {file.filename}', 'error')
                continue

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)

            # Apply Canny edge detection with user-defined thresholds
            edges = cv2.Canny(blurred, threshold, high_threshold)

            # Create a fully transparent background for the traced edges
            traced_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGRA)  # Convert to 4-channel image (RGBA)
            traced_image[:, :, 3] = 0  # Set alpha channel to 0 (fully transparent)

            # Draw only the edges in red
            traced_image[edges > 0, 0] = 0  # Set blue channel to 0
            traced_image[edges > 0, 1] = 0  # Set green channel to 0
            traced_image[edges > 0, 2] = 255  # Set red channel to 255
            traced_image[edges > 0, 3] = 255  # Set alpha channel to 255 for edges

            # Save the traced image with transparency
            traced_filename = f"traced_{filename}"
            traced_image_path = os.path.join(UPLOAD_FOLDER, traced_filename)
            cv2.imwrite(traced_image_path, traced_image)

            '''height, width = traced_image.shape[:2]
            length_middle = math.sqrt((height ** 2) + (width ** 2))
            # gray = cv2.cvtColor(traced_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(traced_image, 127, 255, cv2.THRESH_BINARY)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lines = [np.array([[0, 0], [width, height]], dtype=np.int32).reshape(-1, 2)]

            for contour in contours:
                intersection_count = 0
                for line in lines:
                    pt1, pt2 = tuple(line[0]), tuple(line[1])  # Convert to tuples
                    for contour in contours:
                        rect = cv2.boundingRect(contour)  # Get contour bounding box
                        intersects, new_pt1, new_pt2 = cv2.clipLine(rect, pt1, pt2)
                        if intersects:
                            intersection_count += 1
                print("Total intersections:", intersection_count)

                print(intersection_count)
                grain_avg_size = (intersection_count / length_middle)
                print("Average grain size in pixels: ", grain_avg_size)'''

            # Create a combined image (original + traced overlay)
            original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if original_image is None:
                flash(f'Failed to load original image: {file.filename}', 'error')
                continue

            # Ensure the original image has an alpha channel
            if original_image.shape[2] == 3:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)

            # Overlay the traced image onto the original image
            combined_image = original_image.copy()
            combined_image[traced_image[:, :, 3] == 255] = traced_image[traced_image[:, :, 3] == 255]

            # Save the combined image
            combined_filename = f"combined_{filename}"
            combined_image_path = os.path.join(UPLOAD_FOLDER, combined_filename)
            cv2.imwrite(combined_image_path, combined_image)

            if os.path.exists(combined_image_path):
                os.remove(combined_image_path)
            cv2.imwrite(combined_image_path, combined_image)



            print(f"Combined image successfully saved: {combined_image_path}")  # Debugging

            # Create the URL for the combined image using url_for
            combined_image_url = url_for('static', filename='images/' + combined_filename)

            # Append the filenames to the traced_images list
            traced_images.append((filename, traced_filename, combined_image_url))

        return render_template(
            'index.html',
            traced_images=traced_images,
            structure=structure,
            scale_micrometers=scale_micrometers,
            sigma=sigma,
            threshold=threshold,
            high_threshold=high_threshold
        )

    except Exception as e:
        print("Error:", str(e))
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect('/')



if __name__ == '__main__':
    app.run(debug=True,)