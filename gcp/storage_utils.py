import os
from google.cloud import storage
import pandas as pd
import datetime
from io import BytesIO
from google.auth.credentials import Credentials
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from datetime import timedelta
import re
import requests


class GCStorage_broker:
    client: storage.Client

    def __init__(self, credentials: Credentials = None):
        if credentials:
            # Use as credenciais fornecidas para criar o cliente de
            # armazenamento.
            self.client = storage.Client(credentials=credentials)
            self.credentials = credentials
        else:
            # Se nenhuma credencial for fornecida, crie o cliente sem elas
            # (supondo credenciais padrão da aplicação).
            self.client = storage.Client()

    @staticmethod
    def public_url_uri(public_url: str) -> str:
        """
        Convert a public URL to a URI.

        Parameters:
        - public_url (str): Public URL of the blob.

        Returns:
        - str: URI of the blob.
        """
        return public_url.replace("http://storage.googleapis.com/", "gs://")\
                         .replace("https://storage.googleapis.com/", "gs://")

    @staticmethod
    def parse_uri(source_uri: str) -> Tuple[str, str]:
        """
        Parse the source URI and return the bucket and blob components.

        Parameters:
        - source_uri (str): Source URI of the blob.

        Returns:
        - Tuple[str, str]: Bucket and blob components.
        """
        if not source_uri.startswith("gs://"):
            raise ValueError("Invalid source URI. Must startswith 'gs://'.")

        parts = source_uri.split("gs://")[1].split("/")
        bucket_ = parts[0]
        blob_ = '/'.join(parts[1:]) if len(parts) > 1 else ''

        return bucket_, blob_

    @staticmethod
    def parse_organization_id(
            user_data_path: str,
            organization_id: str) -> str:
        """
        Parse the user data path and return the bucket object full path.

        Parameters:
        - user_data_path (str): Source URI from user.

        Returns:
        - str: Full bucket path.
        """
        # user_data_path = raw/dir/file.parquet
        # return path = gs://organizationID_raw/dir/file.parquet
        if "raw" not in user_data_path:
            raise ValueError(f"Data path should start with raw/.")
        else:
            full_object_path = "gs://" + organization_id + "_" + user_data_path

        return full_object_path

    def check_blob_is_folder(self, blob: storage.Blob) -> bool:
        """
        Check if a blob is a folder.

        Parameters:
        - blob (storage.Blob): Blob to be checked.

        Returns:
        - bool: True if the blob is a folder, False otherwise.
        """
        return blob.name.endswith('/')

    def check_blob(self, source_uri: str) -> bool:
        """
        Check if a blob exists.

        Parameters:
        - source_uri (str): Source URI of the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.

        Returns:
        - bool: True if the blob or bucket exists, False otherwise.
        """
        bucket_name, blob_name = GCStorage_broker.parse_uri(source_uri)
        if blob_name:
            source_bucket = self.client.get_bucket(bucket_name)
            blob = storage.Blob(bucket=source_bucket, name=blob_name)
            return blob.exists(self.client)
        else:
            source_bucket = self.client.get_bucket(bucket_name)
            return source_bucket.exists(self.client)

    def save_from_string(
            self,
            destination_uri: str,
            blob_string: str,
            overwrite: bool = False,
            checksum: str = None) -> bool:
        """
        Save a blob with content from a string to the given destination URI.

        Parameters:
        - destination_uri (str): Destination URI for the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - blob_string (str): Content of the blob as a string. Example: 'name,age,job\nJohn,25,Engineer\nMary,30,Data Scientist'.
        - overwrite (bool): If True, overwrite the existing blob; otherwise, raise an error if the blob already exists.
        - checksum (str) : Supported values are "md5", "crc32c" and None. The default is None.
        Returns:
        - bool: True if blob was saved successfully, False otherwise.
        """
        bucket_name, blob_filename = GCStorage_broker.parse_uri(
            destination_uri)
        bucket = self.client.get_bucket(bucket_name)
        filename = blob_filename
        blob = bucket.blob(filename)

        if not overwrite and blob.exists(self.client):
            raise ValueError(
                f"Blob already exists at {destination_uri}. Set 'overwrite' to True to overwrite.")

        blob.upload_from_string(data=blob_string, checksum=checksum)
        return self.check_blob(destination_uri)

    def save_from_file(
            self,
            destination_uri: str,
            file_path: str,
            overwrite: bool = False,
            checksum: str = None) -> bool:
        """
        Save a blob with content from a file to the given destination URI.

        Parameters:
        - destination_uri (str): Destination URI for the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - file_path (str): Path to the file to be uploaded. Example: './name_age_job.csv'.
        - overwrite (bool): If True, overwrite the existing blob; otherwise, raise an error if the blob already exists.

        Returns:
        - bool: True if the operation is successful, False otherwise.
        """
        bucket_name, blob_filename = GCStorage_broker.parse_uri(
            destination_uri)
        bucket = self.client.get_bucket(bucket_name)
        filename = blob_filename
        blob = bucket.blob(filename)

        if not overwrite and blob.exists(self.client):
            raise ValueError(
                f"Blob already exists at {destination_uri}. Set 'overwrite' to True to overwrite.")

        blob.upload_from_filename(filename=file_path, checksum=checksum)
        return self.check_blob(destination_uri)

    def get_blob_string(self, source_uri: str) -> str:
        """
        Downloads a blob from a Google Cloud Storage bucket and returns its contents as a string.

        Args:
            source_uri (str): The URI of the blob to download. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.

        Returns:
            str: The contents of the downloaded blob as a string. Example: 'name,age,job\nJohn,25,Engineer\nMary,30,Data Scientist'.
        """
        bucket_name, blob_filename = GCStorage_broker.parse_uri(source_uri)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(blob_filename)
        return blob.download_as_text()

    def get_blob_file(self, source_uri: str, file_destination: str) -> bool:
        """
        Download a blob from the source URI to the specified file destination.

        Parameters:
        - source_uri (str): Source URI of the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - file_destination (str): File path where the blob will be saved. Example: './name_age_job.csv'.

        Returns:
        - bool: True if the download is successful, False otherwise.
        """
        with open(file_destination, 'wb') as file_obj:
            self.client.download_blob_to_file(source_uri, file_obj)
        return True

    def get_blob_dataframe(self, source_uri: str) -> pd.DataFrame:
        """
        Downloads a CSV file from a Google Cloud Storage bucket and returns its contents as a pandas DataFrame.

        Args:
            source_uri (str): The URI of the CSV file in the format gs://zion-poc-raw/tests/name_age_job.csv.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the contents of the CSV file.
        """
        bucket_name, blob_filename = GCStorage_broker.parse_uri(source_uri)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(blob_filename)
        data_df = pd.read_csv(BytesIO(blob.download_as_string()))
        return data_df

    def list_blobs_tree(self, source_uri: str) -> List[Dict[str, str]]:
        """
        List blobs tree recursively in a given source URI.

        Parameters:
        - source_uri (str): Source URI of the blobs, ends with / to indicate a folder path. Example: 'gs://zion-poc-raw/

        Returns:
        A recursive list of folders items objects with subfolders

        -   List[
                FolderItem:
                    folder: str = "tests"
                    path: str = "raw/tests
                    subfolders: List[FolderItem] = [ {folder: "tests_a", path: "raw/tests/tests_a", subfolders: []}, {...} ]
            ]
        """
        bucket_name, prefix = GCStorage_broker.parse_uri(source_uri)
        blobs_list = self.client.list_blobs(
            bucket_name, prefix=prefix, delimiter='/')
        prefixes = blobs_list.prefixes
        if len(prefixes) == 0:
            blobs_list._next_page()
        response = []
        for blob in prefixes:
            blob_dict = {}
            blob_dict["folder"] = blob.split('/')[-2]
            blob_dict["path"] = blob[:-1]
            blob_dict["subfolders"] = self.list_blobs_tree(
                f"gs://{bucket_name}/{blob}")
            response.append(blob_dict)
        return response

    def list_blobs(
        self,
        source_uri: str,
        created_after: str = None,
        created_before: str = None,
        updated_after: str = None,
        updated_before: str = None,
        name_regex: str = None,
        min_size: int = None,
        max_size: int = None,
        content_type_match: str = None,
    ) -> List[Dict[str, str | int | float]]:
        """
        List blobs in a given source URI with optional filters.

        Parameters:
        - source_uri (str): Source URI of the blobs, ends with / to indicate a folder path. Example: 'gs://zion-poc-raw/tests/'.
        - created_after (str) DateTime ISO format: Filter blobs created after this datetime. Example: '2021-01-01T00:00:00'.
        - created_before(str) DateTime ISO format: Filter blobs created before this datetime. Example: '2021-01-01T00:00:00'.
        - updated_after(str) DateTime ISO format: Filter blobs updated after this datetime. Example: '2021-01-01T00:00:00'.
        - updated_before(str) DateTime ISO format: Filter blobs updated before this datetime. Example: '2021-01-01T00:00:00'.
        - name_regex (str): Regular expression to filter blobs by name. Example: '.*\\.csv' to match all CSV files.
        - min_size (int) Bytes: Minimum size limit for blobs. Example: 1000000 for 1MB.
        - max_size (int): Maximum size limit for blobs. Example: 1000000 for 1MB.
        - content_type_match (str): Content type to match for blobs. Example: 'application/octet-stream' for CSV files.

        Returns:
        - List[Dict[str, str | int]]: List of blobs objects as dict representing filtered blobs.
            properties:
                - name (str): Name of the blob.
                - content_type (str): Content type of the blob.
                - uri (str): URI of the blob.
                - time_created (str): Creation time of the blob.
                - time_updated (str): Update time of the blob.
                - size (int): Size of the blob in bytes.
                - md5_hash (str): MD5 hash of the blob.
        """
        bucket_name, prefix = GCStorage_broker.parse_uri(source_uri)
        blobs_list = self.client.list_blobs(
            bucket_name, prefix=prefix, delimiter='/')
        #blobs_list = self.client.list_blobs(bucket_name, prefix=prefix)
        # Convert datetime strings to datetime objects
        created_after_dt = datetime.fromisoformat(
            created_after) if created_after else None
        created_before_dt = datetime.fromisoformat(
            created_before) if created_before else None
        updated_after_dt = datetime.fromisoformat(
            updated_after) if updated_after else None
        updated_before_dt = datetime.fromisoformat(
            updated_before) if updated_before else None
        blobs_list_response = []
        # Listing blobs files and creating blobdto object for each blob
        for blob in blobs_list:
            if isinstance(blob, storage.Blob):
                blob_dict = {}
                blob_dict["name"] = blob.name
                blob_dict["content_type"] = blob.content_type or ""
                blob_dict["uri"] = GCStorage_broker.public_url_uri(
                    blob.public_url)
                blob_dict["time_created"] = blob.time_created
                blob_dict["time_updated"] = blob.updated
                blob_dict["size"] = blob.size
                blob_dict["md5_hash"] = blob.md5_hash
                # Apply filters
                if (
                    (created_after_dt is None or blob_dict["time_created"] > created_after_dt)
                    and (created_before_dt is None or blob_dict["time_created"] < created_before_dt)
                    and (updated_after_dt is None or blob_dict["time_updated"] > updated_after_dt)
                    and (updated_before_dt is None or blob_dict["time_updated"] < updated_before_dt)
                    and (name_regex is None or re.search(name_regex, blob_dict["name"]))
                    and (min_size is None or blob_dict["size"] >= min_size)
                    and (max_size is None or blob_dict["size"] <= max_size)
                    and (content_type_match is None or blob_dict["content_type"] == content_type_match)
                ):
                    # converting datetime values to isoformat string
                    blob_dict["time_created"] = blob_dict["time_created"].isoformat(
                    ) if blob_dict["time_created"] else None
                    blob_dict["time_updated"] = blob_dict["time_updated"].isoformat(
                    ) if blob_dict["time_updated"] else None
                    blobs_list_response.append(blob_dict)
            else:
                print("Blob isinstance false: {}".format(blob.name))
        # Getting subfolders from the blobs_list.prefixes property
        # Creating blobsto object for each subfolder
        subfolders = list(blobs_list.prefixes)
        for subfolder in subfolders:
            blobdto_folder = {
                "name": subfolder,
                "content_type": "folder",
                "uri": source_uri + subfolder,
                "time_created": "",
                "time_updated": "",
                "size": 0,
                "md5_hash": ""
            }
            blobs_list_response.append(blobdto_folder)
        return blobs_list_response

    def copy_blob(self, source_uri: str, destination_uri: str) -> str:
        """
        Copy a blob from the source URI to the destination URI and return the new blob URI.
        The path of the destination URI can be a folder path or a complete file URI.
        Parents folders ill be created if they don't exist.

        Parameters:
        - source_uri (str): Source URI of the blob. Exameple: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - destination_uri (str): Destination URI for the blob or path. Example: 'gs://zion-poc-raw/tests/copy/'. or 'gs://zion-poc-raw/tests/copy/name_age_job.csv'.

        Returns:
        - str: New blob URI.
        """
        source_bucket_name, source_blob_name = GCStorage_broker.parse_uri(
            source_uri)

        # Determine if the destination URI is a folder path or a complete file
        # URI
        if destination_uri.endswith('/'):
            # Destination URI is a folder path
            destination_blob_name = f'{destination_uri}{source_blob_name.split("/")[-1]}'
        else:
            # Destination URI is a complete file URI
            _, destination_blob_name = GCStorage_broker.parse_uri(
                destination_uri)

        source_bucket = self.client.get_bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_blob_name)

        destination_bucket_name, _ = GCStorage_broker.parse_uri(
            destination_uri)
        destination_bucket = self.client.get_bucket(destination_bucket_name)

        blob_copy = source_bucket.copy_blob(
            source_blob, destination_bucket, destination_blob_name)

        print(
            f'Blob {source_blob.name} in bucket {source_bucket.name} copied to blob {blob_copy.name} in bucket {destination_bucket.name}.'
        )
        # Get the new blob URI
        new_blob_uri = f'gs://{destination_bucket_name}/{blob_copy.name}'
        return new_blob_uri

    def rename_blob(self, source_uri: str, new_name: str) -> str:
        """
        Rename a blob in Google Cloud Storage and return the new blob URI.

        Parameters:
        - source_uri (str): Source URI of the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - new_name (str): New name for the blob. Example: 'new_name_age_job.csv'. Not URI or PATH.

        Returns:
        - str: New blob URI. Example: 'gs://zion-poc-raw/tests/new_name_age_job.csv'.
        """
        if new_name.__contains__("/"):
            raise ValueError(
                "Invalid new name. Must not uri, path, or contain '/'. Input example 'new_name_age_job.csv'.")
        bucket_name, blob_name = GCStorage_broker.parse_uri(source_uri)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        old_name = blob_name.split("/")[-1]
        _new_name = f'{blob_name.replace(old_name, new_name)}'
        new_blob = bucket.rename_blob(blob, _new_name)
        print(f'Blob {blob.name} has been renamed to {new_blob.name}')
        # Get the new blob URI
        new_blob_uri = GCStorage_broker.public_url_uri(new_blob.public_url)
        return new_blob_uri

    def delete_blob(self, source_uri: str) -> bool:
        """
        Delete a blob from the given source URI.

        Parameters:
        - source_uri (str): Source URI of the blob to be deleted. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.

        Returns:
        - bool: True if the blob was deleted successfully, False otherwise.
        """
        bucket_name, blob_name = GCStorage_broker.parse_uri(source_uri)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        return not self.check_blob(source_uri)

    def move_blob(self, source_uri: str, destination_uri: str) -> str:
        """
        Move a blob from the source URI to the destination URI and return the new blob URI.
        Destination URI must be a file or folder path

        Parameters:
        - source_uri (str): Source URI of the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - destination_uri (str): Destination URI for the blob.  Example: 'gs://zion-poc-raw/tests/moved_name_age_job.csv'. or 'gs://zion-poc-raw/tests/moved/'.

        Returns:
        - str: New blob URI. Example: 'gs://zion-poc-raw/tests/moved_name_age_job.csv'.
        """
        source_bucket_name, source_blob_name = GCStorage_broker.parse_uri(
            source_uri)
        source_bucket = self.client.get_bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_blob_name)
        file_name = source_blob_name.split("/")[-1]

        new_bucket_name, new_blob_name = GCStorage_broker.parse_uri(
            destination_uri)
        new_blob_name = f"{new_blob_name}/{file_name}" if new_blob_name.endswith(
            "/") else new_blob_name

        destination_bucket = self.client.get_bucket(new_bucket_name)
        new_blob = source_bucket.copy_blob(
            source_blob, destination_bucket, new_blob_name)
        source_blob.delete()

        print(f'File moved from {source_blob} to {new_blob_name}')

        # Get the new blob URI
        new_blob_uri = GCStorage_broker.public_url_uri(new_blob.public_url)
        return new_blob_uri

    def signed_url_for_upload(
            self,
            destination_uri: str,
            expiration_minutes: int = 15,
            method: str = "POST") -> str:
        """
        Generate a temporary upload link for a bucket object.

        Parameters:
        - destination_uri (str): URI of the object in the format 'gs://bucket_name/object_path'.
        - expiration_minutes (int): Expiration time in minutes for the URL (default is 15 minutes).
        - method (str): HTTP method for the URL (default is "POST").

        Returns:
        - str: Temporary upload URL for the bucket object.

        Usage example:
            1. Get the upload URL
                url = gcs.signed_url_for_upload(destination_uri='gs://zion-poc-raw/tests/name_age_job.csv', expiration_minutes=15)
            2. Upload the file to the URL with requests library
                import requests
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                filename = os.path.basename(file_path)
                files = {'file': (filename, file_data)}
                response = requests.post(url, files=files)
            3. Response need to be 200 or 204:
                assert (response.status_code == 200 or response.status_code == 204)
        """
        if method not in ["POST", "PUT", "RESUMABLE"]:
            method = "POST"

        client = self.client

        # Parse the object URI
        bucket_name, blob_name = GCStorage_broker.parse_uri(destination_uri)

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        expiration_time = timedelta(minutes=expiration_minutes)
        if method == "RESUMABLE":
            url = blob.create_resumable_upload_session()
        else:
            url = blob.generate_signed_url(
                credentials=self.credentials,
                version="v4",
                expiration=expiration_time,
                method=method,
            )
        return url

    def signed_url_for_download(
            self,
            source_uri: str,
            expiration_minutes: int = 15) -> str:
        """
        Generate a temporary download link for a bucket object.

        Parameters:
        - source_uri (str): URI of the object in the format 'gs://bucket_name/object_path'.
        - expiration_minutes (int): Expiration time in minutes for the URL (default is 15 minutes).

        Returns:
        - str: Temporary download URL for the bucket object.

        Usage example:
            1. Get the download URL
                url = gcs.signed_url_for_download(source_uri='gs://zion-poc-raw/tests/name_age_job.csv', expiration_minutes=15)
            2. Download the file from the URL
                response = requests.get(url)
                with open(file_destination, 'wb') as f:
                    f.write(response.content)
            3. Response need to be 200 or 204:
                assert (response.status_code == 200 or response.status_code == 204)
        """
        client = self.client

        # Parse the object URI
        bucket_name, blob_name = GCStorage_broker.parse_uri(source_uri)

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        expiration_time = timedelta(minutes=expiration_minutes)

        url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
        )
        return url

    @staticmethod
    def upload_file_to_url(file_path: str, url: str):
        """
        Upload a file to a given URL.

        Parameters:
        - file_path (str): Path to the file to be uploaded.
        - url (str): The upload URL.

        Returns:
        - HTTP response from the server.
        """
        with open(file_path, 'rb') as f:
            file_data = f.read()
        filename = os.path.basename(file_path)
        files = {'file': (filename, file_data)}
        response = requests.post(url, files=files)
        return response

    @staticmethod
    def download_file_from_url(file_destination: str, url: str):
        """
        Download a file from a given URL.

        Parameters:
        - file_destination (str): Path to the file to be saved.
        - url (str): The download URL.

        Returns:
        - HTTP response from the server.
        """
        response = requests.get(url)
        with open(file_destination, 'wb') as f:
            f.write(response.content)
        return response

    def list_buckets(self) -> List[str]:
        """
        List all buckets in the Google Cloud Storage.

        Returns:
        - List[str]: List of bucket names.
        """
        storage_client = self.client
        buckets = list(storage_client.list_buckets())
        print(buckets)
        return [bucket.name for bucket in buckets]

    def save_from_string_censored(
            self,
            destination_uri: str,
            blob_string: str,
            namespace: str,
            overwrite: bool = False,
            checksum: str = None) -> bool:
        """
        Save a blob with content from a string to the given destination URI.

        Parameters:
        - destination_uri (str): Destination URI for the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - blob_string (str): Content of the blob as a string. Example: 'name,age,job\nJohn,25,Engineer\nMary,30,Data Scientist'.
        - namespace (str): Organization ID from user token
        - overwrite (bool): If True, overwrite the existing blob; otherwise, raise an error if the blob already exists.
        - checksum (str) : Supported values are "md5", "crc32c" and None. The default is None.
        Returns:
        - bool: True if blob was saved successfully, False otherwise.
        """
        # destination_uri from request is "raw/dir/file.parquet"
        # destination_uri needed
        # "gs://34acfe82-1318-4e7f-85f6-6af1f49aa21f_raw/dir/file.parquet"

        full_uri = GCStorage_broker.parse_organization_id(
            destination_uri, namespace)
        bucket_name, blob_filename = GCStorage_broker.parse_uri(full_uri)
        bucket = self.client.get_bucket(bucket_name)
        filename = blob_filename
        blob = bucket.blob(filename)

        if not overwrite and blob.exists(self.client):
            raise ValueError(
                f"Blob already exists at {full_uri}. Set 'overwrite' to True to overwrite.")

        blob.upload_from_string(data=blob_string, checksum=checksum)
        return self.check_blob_censored(destination_uri, namespace)

    def check_blob_censored(self, source_uri: str, namespace: str) -> bool:
        """
        Check if a blob exists.

        Parameters:
        - source_uri (str): Source URI of the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.

        Returns:
        - bool: True if the blob exists, False otherwise.
        """
        full_uri = GCStorage_broker.parse_organization_id(
            source_uri, namespace)
        bucket_name, blob_name = GCStorage_broker.parse_uri(full_uri)
        source_bucket = self.client.get_bucket(bucket_name)
        blob = storage.Blob(bucket=source_bucket, name=blob_name)
        return blob.exists(self.client)

    def save_from_file_censored(
            self,
            destination_uri: str,
            file_path: str,
            namespace: str,
            overwrite: bool = False,
            checksum: str = None) -> bool:
        """
        Save a blob with content from a file to the given destination URI.

        Parameters:
        - destination_uri (str): Destination URI for the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - file_path (str): Path to the file to be uploaded. Example: './name_age_job.csv'.
        - overwrite (bool): If True, overwrite the existing blob; otherwise, raise an error if the blob already exists.

        Returns:
        - bool: True if the operation is successful, False otherwise.
        """
        full_uri = GCStorage_broker.parse_organization_id(
            destination_uri, namespace)
        bucket_name, blob_filename = GCStorage_broker.parse_uri(full_uri)
        bucket = self.client.get_bucket(bucket_name)
        filename = blob_filename
        blob = bucket.blob(filename)

        if not overwrite and blob.exists(self.client):
            raise ValueError(
                f"Blob already exists at {full_uri}. Set 'overwrite' to True to overwrite.")

        blob.upload_from_filename(filename=file_path, checksum=checksum)
        return self.check_blob_censored(destination_uri, namespace)

    def get_blob_string_censored(self, source_uri: str, namespace: str) -> str:
        """
        Downloads a blob from a Google Cloud Storage bucket and returns its contents as a string.

        Args:
            source_uri (str): The URI of the blob to download. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.

        Returns:
            str: The contents of the downloaded blob as a string. Example: 'name,age,job\nJohn,25,Engineer\nMary,30,Data Scientist'.
        """
        full_uri = GCStorage_broker.parse_organization_id(
            source_uri, namespace)
        bucket_name, blob_filename = GCStorage_broker.parse_uri(full_uri)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(blob_filename)
        return blob.download_as_text()

    def get_blob_file_censored(
            self,
            source_uri: str,
            file_destination: str,
            namespace: str) -> bool:
        """
        Download a blob from the source URI to the specified file destination.

        Parameters:
        - source_uri (str): Source URI of the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - file_destination (str): File path where the blob will be saved. Example: './name_age_job.csv'.

        Returns:
        - bool: True if the download is successful, False otherwise.
        """
        full_uri = GCStorage_broker.parse_organization_id(
            source_uri, namespace)
        with open(file_destination, 'wb') as file_obj:
            self.client.download_blob_to_file(full_uri, file_obj)
        return True

    def get_blob_dataframe_censored(
            self,
            source_uri: str,
            namespace: str) -> pd.DataFrame:
        """
        Downloads a CSV file from a Google Cloud Storage bucket and returns its contents as a pandas DataFrame.

        Args:
            source_uri (str): The URI of the CSV file in the format gs://zion-poc-raw/tests/name_age_job.csv.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the contents of the CSV file.
        """
        full_uri = GCStorage_broker.parse_organization_id(
            source_uri, namespace)
        bucket_name, blob_filename = GCStorage_broker.parse_uri(full_uri)
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(blob_filename)
        data_df = pd.read_csv(BytesIO(blob.download_as_string()))
        return data_df

    def list_blobs_censored(
        self,
        source_uri: str,
        namespace: str,
        created_after: str = None,
        created_before: str = None,
        updated_after: str = None,
        updated_before: str = None,
        name_regex: str = None,
        min_size: int = None,
        max_size: int = None,
        content_type_match: str = None,
    ) -> List[Dict[str, str | int | float]]:
        """
        List blobs in a given source URI with optional filters.

        Parameters:
        - source_uri (str): Source URI of the blobs, ends with / to indicate a folder path. Example: 'gs://zion-poc-raw/tests/'.
        - created_after (str) DateTime ISO format: Filter blobs created after this datetime. Example: '2021-01-01T00:00:00'.
        - created_before(str) DateTime ISO format: Filter blobs created before this datetime. Example: '2021-01-01T00:00:00'.
        - updated_after(str) DateTime ISO format: Filter blobs updated after this datetime. Example: '2021-01-01T00:00:00'.
        - updated_before(str) DateTime ISO format: Filter blobs updated before this datetime. Example: '2021-01-01T00:00:00'.
        - name_regex (str): Regular expression to filter blobs by name. Example: '.*\\.csv' to match all CSV files.
        - min_size (int) Bytes: Minimum size limit for blobs. Example: 1000000 for 1MB.
        - max_size (int): Maximum size limit for blobs. Example: 1000000 for 1MB.
        - content_type_match (str): Content type to match for blobs. Example: 'application/octet-stream' for CSV files.

        Returns:
        - List[Dict[str, str | int]]: List of blobs objects as dict representing filtered blobs.
            properties:
                - name (str): Name of the blob.
                - content_type (str): Content type of the blob.
                - uri (str): URI of the blob.
                - time_created (str): Creation time of the blob.
                - time_updated (str): Update time of the blob.
                - size (int): Size of the blob in bytes.
                - md5_hash (str): MD5 hash of the blob.
        """
        full_uri = GCStorage_broker.parse_organization_id(
            source_uri, namespace)
        bucket_name, prefix = GCStorage_broker.parse_uri(full_uri)
        blobs_list = self.client.list_blobs(
            bucket_name, prefix=prefix, delimiter='/')
        #blobs_list = self.client.list_blobs(bucket_name, prefix=prefix)
        # Convert datetime strings to datetime objects
        created_after_dt = datetime.fromisoformat(
            created_after) if created_after else None
        created_before_dt = datetime.fromisoformat(
            created_before) if created_before else None
        updated_after_dt = datetime.fromisoformat(
            updated_after) if updated_after else None
        updated_before_dt = datetime.fromisoformat(
            updated_before) if updated_before else None
        blobs_list_response = []
        # Listing blobs files and creating blobdto object for each blob
        for blob in blobs_list:
            if isinstance(blob, storage.Blob):
                blob_dict = {}
                blob_dict["name"] = blob.name
                blob_dict["content_type"] = blob.content_type or ""
                blob_dict["uri"] = GCStorage_broker.public_url_uri(
                    blob.public_url)
                blob_dict["time_created"] = blob.time_created
                blob_dict["time_updated"] = blob.updated
                blob_dict["size"] = blob.size
                blob_dict["md5_hash"] = blob.md5_hash
                # Apply filters
                if (
                    (created_after_dt is None or blob_dict["time_created"] > created_after_dt)
                    and (created_before_dt is None or blob_dict["time_created"] < created_before_dt)
                    and (updated_after_dt is None or blob_dict["time_updated"] > updated_after_dt)
                    and (updated_before_dt is None or blob_dict["time_updated"] < updated_before_dt)
                    and (name_regex is None or re.search(name_regex, blob_dict["name"]))
                    and (min_size is None or blob_dict["size"] >= min_size)
                    and (max_size is None or blob_dict["size"] <= max_size)
                    and (content_type_match is None or blob_dict["content_type"] == content_type_match)
                ):
                    # converting datetime values to isoformat string
                    blob_dict["time_created"] = blob_dict["time_created"].isoformat(
                    ) if blob_dict["time_created"] else None
                    blob_dict["time_updated"] = blob_dict["time_updated"].isoformat(
                    ) if blob_dict["time_updated"] else None
                    blobs_list_response.append(blob_dict)
            else:
                print("Blob isinstance false: {}".format(blob.name))
        # Getting subfolders from the blobs_list.prefixes property
        # Creating blobsto object for each subfolder
        subfolders = list(blobs_list.prefixes)
        for subfolder in subfolders:
            blobdto_folder = {
                "name": subfolder,
                "content_type": "folder",
                "uri": full_uri + subfolder,
                "time_created": "",
                "time_updated": "",
                "size": 0,
                "md5_hash": ""
            }
            blobs_list_response.append(blobdto_folder)
        return blobs_list_response

    def rename_blob_censored(
            self,
            source_uri: str,
            new_name: str,
            namespace: str) -> str:
        """
        Rename a blob in Google Cloud Storage and return the new blob URI.

        Parameters:
        - source_uri (str): Source URI of the blob. Example: 'gs://zion-poc-raw/tests/name_age_job.csv'.
        - new_name (str): New name for the blob. Example: 'new_name_age_job.csv'. Not URI or PATH.

        Returns:
        - str: New blob URI. Example: 'gs://zion-poc-raw/tests/new_name_age_job.csv'.
        """
        if new_name.__contains__("/"):
            raise ValueError(
                "Invalid new name. Must not uri, path, or contain '/'. Input example 'new_name_age_job.csv'.")
        full_uri = GCStorage_broker.parse_organization_id(
            source_uri, namespace)
        bucket_name, blob_name = GCStorage_broker.parse_uri(full_uri)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        old_name = blob_name.split("/")[-1]
        _new_name = f'{blob_name.replace(old_name, new_name)}'
        new_blob = bucket.rename_blob(blob, _new_name)
        print(f'Blob {blob.name} has been renamed to {new_blob.name}')
        # Get the new blob URI
        new_blob_uri = GCStorage_broker.public_url_uri(new_blob.public_url)
        return new_blob_uri

    def signed_url_for_upload_censored(
            self,
            destination_uri: str,
            namespace: str,
            expiration_minutes: int = 15,
            method: str = "POST") -> str:
        """
        Generate a temporary upload link for a bucket object.

        Parameters:
        - destination_uri (str): URI of the object in the format 'gs://bucket_name/object_path'.
        - expiration_minutes (int): Expiration time in minutes for the URL (default is 15 minutes).
        - method (str): HTTP method for the URL (default is "POST").

        Returns:
        - str: Temporary upload URL for the bucket object.

        Usage example:
            1. Get the upload URL
                url = gcs.signed_url_for_upload(destination_uri='gs://zion-poc-raw/tests/name_age_job.csv', expiration_minutes=15)
            2. Upload the file to the URL with requests library
                import requests
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                filename = os.path.basename(file_path)
                files = {'file': (filename, file_data)}
                response = requests.post(url, files=files)
            3. Response need to be 200 or 204:
                assert (response.status_code == 200 or response.status_code == 204)
        """
        if method not in ["POST", "PUT", "RESUMABLE"]:
            method = "POST"

        client = self.client

        # Parse the object URI
        full_uri = GCStorage_broker.parse_organization_id(
            destination_uri, namespace)
        bucket_name, blob_name = GCStorage_broker.parse_uri(full_uri)

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        expiration_time = timedelta(minutes=expiration_minutes)
        if method == "RESUMABLE":
            url = blob.create_resumable_upload_session()
        else:
            url = blob.generate_signed_url(
                credentials=self.credentials,
                version="v4",
                expiration=expiration_time,
                method=method,
            )
        return url

    def signed_url_for_download_censorde(
            self,
            source_uri: str,
            namespace: str,
            expiration_minutes: int = 15) -> str:
        """
        Generate a temporary download link for a bucket object.

        Parameters:
        - source_uri (str): URI of the object in the format 'gs://bucket_name/object_path'.
        - expiration_minutes (int): Expiration time in minutes for the URL (default is 15 minutes).

        Returns:
        - str: Temporary download URL for the bucket object.

        Usage example:
            1. Get the download URL
                url = gcs.signed_url_for_download(source_uri='gs://zion-poc-raw/tests/name_age_job.csv', expiration_minutes=15)
            2. Download the file from the URL
                response = requests.get(url)
                with open(file_destination, 'wb') as f:
                    f.write(response.content)
            3. Response need to be 200 or 204:
                assert (response.status_code == 200 or response.status_code == 204)
        """
        client = self.client

        # Parse the object URI
        full_uri = GCStorage_broker.parse_organization_id(
            source_uri, namespace)
        bucket_name, blob_name = GCStorage_broker.parse_uri(full_uri)

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        expiration_time = timedelta(minutes=expiration_minutes)

        url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
        )
        return url

    def create_bucket(self, organization_id) -> bool:
        """
        Create bucket for every new organizationID registered.
        """
        bucket_name = organization_id + "_raw"

        storage_client = self.client

        bucket = storage_client.bucket(bucket_name)
        bucket.storage_class = "STANDARD"
        new_bucket = storage_client.create_bucket(bucket)

        return "New bucket " + new_bucket.name + " created."
