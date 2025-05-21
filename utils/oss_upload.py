import oss2
import os
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("ALI_OSS_ENDPOINT")

auth = oss2.Auth(os.getenv("ALI_ACCESSKEY"), os.getenv("ALI_SECRET"))
bucket = oss2.Bucket(auth, endpoint, os.getenv("ALI_OSS_BUCKET"))

def upload_file_to_oss(oss_file_path, local_file_path):
    file_size = os.path.getsize(local_file_path)
    if file_size < 100 * 1024 * 1024:  # 100MB以内
        # 小文件直接上传
        bucket.put_object_from_file(oss_file_path, local_file_path)
    else:
        # 大文件手动分片上传（见下方实现）
        multipart_upload_file(bucket, oss_file_path, local_file_path)

def multipart_upload_file(bucket, oss_file_path, local_file_path):
    from oss2 import SizedFileAdapter, determine_part_size
    from oss2.models import PartInfo
    total_size = os.path.getsize(local_file_path)
    part_size = determine_part_size(total_size, preferred_size=10 * 1024 * 1024)  # 10MB
    upload_id = bucket.init_multipart_upload(oss_file_path).upload_id
    parts = []
    with open(local_file_path, 'rb') as fileobj:
        part_number = 1
        offset = 0
        while offset < total_size:
            num_to_upload = min(part_size, total_size - offset)
            result = bucket.upload_part(oss_file_path, upload_id, part_number, SizedFileAdapter(fileobj, num_to_upload))
            parts.append(PartInfo(part_number, result.etag))
            offset += num_to_upload
            part_number += 1
    bucket.complete_multipart_upload(oss_file_path, upload_id, parts)

def download_file_from_oss(oss_file_path, local_file_path):
    bucket.get_object_to_file(oss_file_path, local_file_path)