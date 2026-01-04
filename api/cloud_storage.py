"""
Cloud Storage Service for CosArt
AWS S3 integration for storing generated images and models
"""

import boto3
import os
from typing import Optional, Dict, Any, BinaryIO
from botocore.exceptions import ClientError
from config.settings import settings
import uuid
from datetime import datetime
import mimetypes

class CloudStorageService:
    """AWS S3 cloud storage service"""

    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = settings.CLOUD_STORAGE_BUCKET

    def upload_file(
        self,
        file_obj: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
        folder: str = "images",
        public: bool = False
    ) -> Dict[str, Any]:
        """
        Upload a file to S3

        Args:
            file_obj: File-like object to upload
            filename: Original filename
            content_type: MIME type (auto-detected if None)
            folder: S3 folder/prefix
            public: Whether to make file publicly accessible

        Returns:
            Dict with upload details
        """
        try:
            # Generate unique filename
            file_extension = os.path.splitext(filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            s3_key = f"{folder}/{unique_filename}"

            # Detect content type
            if not content_type:
                content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

            # Upload parameters
            upload_args = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': file_obj,
                'ContentType': content_type,
                'Metadata': {
                    'original_filename': filename,
                    'uploaded_at': datetime.utcnow().isoformat(),
                    'folder': folder
                }
            }

            # Set ACL for public access
            if public:
                upload_args['ACL'] = 'public-read'

            # Upload file
            self.s3_client.upload_fileobj(**upload_args)

            # Generate URL
            url = self._generate_presigned_url(s3_key) if not public else self._generate_public_url(s3_key)

            return {
                'success': True,
                's3_key': s3_key,
                'url': url,
                'bucket': self.bucket_name,
                'size': file_obj.tell() if hasattr(file_obj, 'tell') else None,
                'content_type': content_type,
                'public': public
            }

        except ClientError as e:
            return {
                'success': False,
                'error': str(e),
                's3_key': None,
                'url': None
            }

    def upload_image(
        self,
        image_data: bytes,
        filename: str = "generated_image.png",
        user_id: Optional[str] = None,
        public: bool = True
    ) -> Dict[str, Any]:
        """
        Upload an image to S3

        Args:
            image_data: Image bytes
            filename: Image filename
            user_id: User ID for organizing files
            public: Whether image should be publicly accessible

        Returns:
            Upload result dict
        """
        from io import BytesIO

        # Create folder structure
        folder = f"users/{user_id}/images" if user_id else "images"

        # Create file-like object
        file_obj = BytesIO(image_data)
        file_obj.seek(0)

        return self.upload_file(
            file_obj=file_obj,
            filename=filename,
            content_type='image/png',
            folder=folder,
            public=public
        )

    def upload_model(
        self,
        model_data: bytes,
        model_name: str,
        version: str = "latest"
    ) -> Dict[str, Any]:
        """
        Upload a trained model to S3

        Args:
            model_data: Model file bytes
            model_name: Name of the model
            version: Model version

        Returns:
            Upload result dict
        """
        from io import BytesIO

        folder = f"models/{model_name}/{version}"
        filename = f"{model_name}_{version}.pth"

        file_obj = BytesIO(model_data)
        file_obj.seek(0)

        return self.upload_file(
            file_obj=file_obj,
            filename=filename,
            content_type='application/octet-stream',
            folder=folder,
            public=False  # Models are private
        )

    def download_file(self, s3_key: str) -> Optional[bytes]:
        """
        Download a file from S3

        Args:
            s3_key: S3 object key

        Returns:
            File content as bytes or None if failed
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError:
            return None

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3

        Args:
            s3_key: S3 object key

        Returns:
            True if deleted successfully
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def list_user_files(self, user_id: str, folder: str = "images", max_keys: int = 100) -> Dict[str, Any]:
        """
        List user's files in S3

        Args:
            user_id: User ID
            folder: Folder to list
            max_keys: Maximum number of files to return

        Returns:
            Dict with file list
        """
        try:
            prefix = f"users/{user_id}/{folder}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': self._generate_presigned_url(obj['Key'])
                    })

            return {
                'success': True,
                'files': files,
                'count': len(files),
                'truncated': response.get('IsTruncated', False)
            }

        except ClientError as e:
            return {
                'success': False,
                'error': str(e),
                'files': [],
                'count': 0
            }

    def _generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for private S3 objects

        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError:
            return ""

    def _generate_public_url(self, s3_key: str) -> str:
        """
        Generate a public URL for public S3 objects

        Args:
            s3_key: S3 object key

        Returns:
            Public URL
        """
        return f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"

# Global instance
cloud_storage = CloudStorageService()