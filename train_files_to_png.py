import pydicom
import pandas as pd
import numpy as np
import os
from PIL import Image

# Path to the DICOM images
dicom_images_dir = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'
# Path to the CSV file with labels
labels_csv_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
# Output directory
output_dir = '/kaggle/working/output'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file containing labels
labels_df = pd.read_csv(labels_csv_path)


# Function to convert DICOM to PNG
def convert_dicom_to_png(ds, output_path):
    img = ds.pixel_array
    img = Image.fromarray(img)
    img.save(output_path)


# Function to create txt files for bounding boxes
def create_bbox_txt(patient_id, df, img_width, img_height, output_dir):
    # Filter dataframe for the patient ID
    patient_df = df[df['patientId'] == patient_id]
    # Create a txt file with bounding box data
    txt_output_path = os.path.join(output_dir, patient_id + '.txt')

    # If there is no pneumonia, create an empty txt file or specify 'normal'
    if patient_df['Target'].sum() == 0:
        # with open(txt_output_path, 'w') as f:
        # Uncomment the line below if you want to label 'normal' without bounding box
        # f.write('1 \n') # '1' would be the class for 'normal'
        pass
    else:
        # Otherwise, there is pneumonia, write the bounding box data
        with open(txt_output_path, 'w') as f:
            for index, row in patient_df.iterrows():
                if row['Target'] == 1:
                    # YOLO format: class x_center y_center width height (normalized)
                    x_center = (row['x'] + row['width'] / 2) / img_width
                    y_center = (row['y'] + row['height'] / 2) / img_height
                    width = row['width'] / img_width
                    height = row['height'] / img_height
                    # Class 0 for pneumonia
                    f.write(f'0 {x_center} {y_center} {width} {height}\n')


# Create classes.txt file
classes_txt_path = os.path.join(output_dir, 'classes.txt')
with open(classes_txt_path, 'w') as f:
    f.write('pneumonia\n')
    # f.write('normal\n') # Add this line only if you're using the 'normal' class

# Initialize a counter
counter = 0

# Convert DICOM images to PNG and create corresponding txt files
for filename in os.listdir(dicom_images_dir):
    if filename.endswith('.dcm'):
        # Break out of the loop if 1000 images have been processed
        if counter >= 2000:
            break

        patient_id = filename.split('.')[0]
        dicom_path = os.path.join(dicom_images_dir, filename)
        ds = pydicom.dcmread(dicom_path)  # Read the DICOM image

        png_output_path = os.path.join(output_dir, patient_id + '.png')
        convert_dicom_to_png(ds, png_output_path)
        create_bbox_txt(patient_id, labels_df, ds.Columns, ds.Rows, output_dir)

        # Increment the counter
        counter += 1




#Create zip archive
import os
import zipfile
# Define the directory containing the output files
output_dir = '/kaggle/working/output'
# Define the path for the zip file
zip_path = '/kaggle/working/pneumonia_detection_output.zip'

# Create a zip file
with zipfile.ZipFile(zip_path, 'w') as zipf:
    # Walk through the directory
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            # Write each file to the zip file with its relative path
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, output_dir))

# Clear all files in the output directory after creating the zip file
for root, dirs, files in os.walk(output_dir):
    for file in files:
        os.remove(os.path.join(root, file))

# Check if the output directory is empty (should be empty if all files are cleared)
is_output_dir_empty = not os.listdir(output_dir)

is_output_dir_empty, zip_path