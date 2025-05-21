import oss2
import os
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("ALI_OSS_ENDPOINT")

auth = oss2.Auth(os.getenv("ALI_ACCESSKEY"), os.getenv("ALI_SECRET"))
bucket = oss2.Bucket(auth, endpoint, os.getenv("ALI_OSS_BUCKET"))

def upload_file_to_oss(local_file_path, oss_file_path):
    bucket.put_object_from_file(oss_file_path, local_file_path)

def download_file_from_oss(oss_file_path, local_file_path):
    bucket.get_object_to_file(oss_file_path, local_file_path)