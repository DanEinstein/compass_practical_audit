# Create a file called download_compas.py
import urllib.request
import os

# Create directory
path = r"C:\Users\User\Desktop\week7_AI\venv\lib\site-packages\aif360\data\raw\compas"
os.makedirs(path, exist_ok=True)

# Download file
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
output_file = os.path.join(path, "compas-scores-two-years.csv")

print(f"Downloading to: {output_file}")
urllib.request.urlretrieve(url, output_file)
print("Download complete!")