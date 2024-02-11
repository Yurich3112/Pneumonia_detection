import torch
import os
from PIL import Image
import sys
import requests
from io import BytesIO
import csv
from pathlib import Path


def download_weights(weights_url, weights_path):
    response = requests.get(weights_url)
    response.raise_for_status()  # ensure the download succeeded
    with open(weights_path, 'wb') as f:
        f.write(response.content)


def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    model.eval()
    return model


def run_inference(model, directory_path, output_csv_path):
    submission_data = [['patientId', 'PredictionString']]
    for image_path in Path(directory_path).glob('*.png'):
        results = model(str(image_path))
        prediction_string = ' '.join(
            f'{x[4]} {x[0]} {x[1]} {x[2] - x[0]} {x[3] - x[1]}' for x in results.pred[0]
        )
        patient_id = image_path.stem
        submission_data.append([patient_id, prediction_string.strip()])
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(submission_data)


def main(directory_path):
    weights_url = 'https://github.com/Yurich3112/Pneumonia_detection/blob/main/best.pt?raw=true'
    weights_path = '/kaggle/working/best.pt'
    download_weights(weights_url, weights_path)

    model = load_model(weights_path)
    output_csv_path = '/kaggle/working/submission.csv'
    run_inference(model, directory_path, output_csv_path)

    print(f"Submission file created: {output_csv_path}")


if __name__ == '__main__':
    directory_path = sys.argv[1]  # Directory of RGB images as an argument
    main(directory_path)
