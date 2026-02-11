import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # 下载 ZIP 文件
    with urllib.request.urlopen(url) as response:   # A
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压 ZIP 文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref: # B
        zip_ref.extractall(extracted_path)

    # 重命名原始文件
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)   # C
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
