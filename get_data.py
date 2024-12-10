import csv
import requests
import os

# Path to your CSV file and images folder
csv_file = 'kili_data/data/remote_assets.csv'
images_folder = 'kili_data/images'

# Ensure the images folder exists
os.makedirs(images_folder, exist_ok=True)

# Read CSV and download images
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image_name = row['label file'].replace('.xml', '.jpg')  # Adjust if your format differs
        image_url = row['url']
        image_path = os.path.join(images_folder, image_name)
        
        # Download and save the image
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(image_path, 'wb') as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Downloaded {image_name}")
        else:
            print(f"Failed to download {image_name}")
            print(response)

print("All images downloaded.")
