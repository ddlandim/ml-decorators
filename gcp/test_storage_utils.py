import pytest
from google.auth.credentials import AnonymousCredentials
from io import StringIO
from google.cloud import storage
from storage_utils import GCStorage_broker
from credentials_utils import Credentials_broker
import pandas as pd
import storage_router
import api_dtos
import os
import sys
from typing import List, Dict


@pytest.fixture
def storage_broker() -> Credentials_broker:
    # Set GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account key file
    # If you are in VSCODE you can set it in .vscode/launch.json to run tests
    # in debug mode
    key_path = os.path.abspath("/home/douglas/keys/key.json")
    Credentials_broker.clear_instances()  # Clear instances of Credentials_broker
    creds_broker = Credentials_broker(filename=key_path)
    creds = creds_broker.get_credentials()
    return GCStorage_broker(credentials=creds)


@pytest.fixture
def mock_blob_string() -> str:
    return "name,age,job\nbob,30,engineer\nalice,20,student\nmarta,40,manager"


@pytest.fixture
def mock_blob_dataframe() -> pd.DataFrame:
    data = {
        'name': [
            'bob', 'alice', 'marta'], 'age': [
            30, 20, 40], 'job': [
                'engineer', 'student', 'manager']}
    return pd.DataFrame(data)


@pytest.fixture
def mock_blob_uri() -> str:
    return "gs://zion-poc-raw/tests/name_age_job.csv"


@pytest.fixture
def mock_local_file() -> str:
    return "/home/douglas/git/data/api-broker/name_age_job.csv"


@pytest.fixture
def mock_content_type() -> str:
    return "text/csv"


@pytest.fixture
def mock_min_size() -> str:
    return 62


@pytest.fixture
def mock_400k_blob_uri() -> str:
    return "gs://zion-poc-raw/GOVBR_ESTABELE_400k.csv"


@pytest.fixture
def mock_bucket_uri() -> str:
    return "gs://zion-poc-raw/"


@pytest.fixture
def mock_test_uri() -> str:
    return "gs://zion-poc-raw/tests/"


@pytest.fixture
def mock_blob_duplicated_uri() -> str:
    _mock_blob_move_uri = "gs://zion-poc-raw/tests/duplicated_name_age_job.csv"
    return _mock_blob_move_uri


@pytest.fixture
def mock_blob_move_uri() -> str:
    _mock_blob_move_uri = "gs://zion-poc-raw/tests/move_name_age_job.csv"
    return _mock_blob_move_uri


@pytest.fixture
def mock_blob_moved_uri() -> str:
    _mock_blob_move_uri = "gs://zion-poc-raw/tests/moved/moved_name_age_job.csv"
    return _mock_blob_move_uri


@pytest.fixture
def mock_blob_rename_uri() -> str:
    _mock_blob_rename_uri = "gs://zion-poc-raw/tests/rename_name_age_job.csv"
    return _mock_blob_rename_uri


@pytest.fixture
def mock_new_name() -> str:
    _mock_blob_rename_uri = "renamed_name_age_job.csv"
    return _mock_blob_rename_uri


@pytest.fixture
def mock_blob_renamed_uri() -> str:
    _mock_blob_rename_uri = "gs://zion-poc-raw/tests/renamed_name_age_job.csv"
    return _mock_blob_rename_uri


@pytest.fixture
def mock_blob_delete_uri() -> str:
    _mock_blob_delete_uri = "gs://zion-poc-raw/tests/delete_name_age_job.csv"
    return _mock_blob_delete_uri


@pytest.fixture
def mock_blob_upload_uri() -> str:
    _mock_blob_upload_uri = "gs://zion-poc-raw/tests/upload_signed_stream.csv"
    return _mock_blob_upload_uri


def test_save_from_string(
        storage_broker: GCStorage_broker,
        mock_blob_uri,
        mock_blob_string,
        mock_blob_move_uri,
        mock_blob_rename_uri,
        mock_blob_delete_uri):
    result = storage_broker.save_from_string(destination_uri=mock_blob_uri,
                                             blob_string=mock_blob_string,
                                             overwrite=True)
    assert result is True
    assert storage_broker.check_blob(mock_blob_uri)
    downloaded_content = storage_broker.get_blob_string(mock_blob_uri)
    assert downloaded_content == mock_blob_string
    assert storage_broker.save_from_string(destination_uri=mock_blob_move_uri,
                                           blob_string=mock_blob_string,
                                           overwrite=True)
    assert storage_broker.save_from_string(
        destination_uri=mock_blob_rename_uri,
        blob_string=mock_blob_string,
        overwrite=True)
    assert storage_broker.save_from_string(
        destination_uri=mock_blob_delete_uri,
        blob_string=mock_blob_string,
        overwrite=True)


def test_save_from_file(
        storage_broker: GCStorage_broker,
        mock_blob_uri,
        mock_blob_string,
        mock_local_file):
    file_path = mock_local_file
    with open(file_path, "w") as file:
        file.write(mock_blob_string)

    result = storage_broker.save_from_file(destination_uri=mock_blob_uri,
                                           file_path=file_path,
                                           overwrite=True)
    assert result is True
    assert storage_broker.check_blob(mock_blob_uri)
    # delete local file
    # os.remove(file_path)


def test_get_blob_string(storage_broker, mock_blob_uri, mock_blob_string):
    result = storage_broker.get_blob_string(mock_blob_uri)
    assert result == mock_blob_string


def test_get_blob_dataframe(
        storage_broker,
        mock_blob_uri,
        mock_blob_dataframe):
    result = storage_broker.get_blob_dataframe(mock_blob_uri)
    pd.testing.assert_frame_equal(result, mock_blob_dataframe)
    df2 = storage_broker.get_blob_dataframe(mock_blob_uri)


def test_list_blobs(
        storage_broker: GCStorage_broker,
        mock_test_uri: str,
        mock_blob_uri: str,
        mock_min_size,
        mock_content_type,
        mock_bucket_uri):
    # Assuming all your mocks have the same content type for simplicity
    result = storage_broker.list_blobs(
        source_uri=mock_test_uri,
        min_size=mock_min_size,
        content_type_match=mock_content_type
    )
    # Converting the result List[Dict] to a list of BlobDTO
    result_blob = [api_dtos.BlobDTO(**blob) for blob in result]
    # Check if a BlobDTO with the expected blob_uri exists in the result
    assert len(result) >= 0
    assert all(((blob.content_type == mock_content_type and blob.size >= mock_min_size)
                or
                (blob.content_type == "folder" and blob.size == 0)
                )
               for blob in result_blob)


def test_list_blobs_non_recursive(
        storage_broker: GCStorage_broker,
        mock_bucket_uri: str):
    # testing non recursive return, should only return files or folders name
    # in the root folder ###
    result = storage_broker.list_blobs(source_uri=mock_bucket_uri)
    result_blob = [api_dtos.BlobDTO(**blob) for blob in result]
    assert len(result) > 0
    assert all(((blob.content_type != "folder" and not blob.name.__contains__("/"))
                or
                (blob.content_type == "folder" and blob.name.__contains__("/"))
                )
               for blob in result_blob)


def test_storage_router(mock_test_uri: str):
    ### testing router ###
    assert isinstance(
        storage_router.gcs_list(
            source_uri=mock_test_uri),
        api_dtos.HttpResponseLBlob)


def test_save_from_file(
        storage_broker: GCStorage_broker,
        mock_blob_uri,
        mock_blob_string,
        mock_local_file):
    file_path = mock_local_file
    with open(file_path, "w") as file:
        file.write(mock_blob_string)

    result = storage_broker.save_from_file(destination_uri=mock_blob_uri,
                                           file_path=file_path,
                                           overwrite=True)
    assert result is True
    assert storage_broker.check_blob(mock_blob_uri)
    # delete local file
    os.remove(file_path)


def test_get_blob_string(storage_broker, mock_blob_uri, mock_blob_string):
    result = storage_broker.get_blob_string(mock_blob_uri)
    assert result == mock_blob_string


def test_get_blob_dataframe(
        storage_broker,
        mock_blob_uri,
        mock_blob_dataframe):
    result = storage_broker.get_blob_dataframe(mock_blob_uri)
    pd.testing.assert_frame_equal(result, mock_blob_dataframe)
    df2 = storage_broker.get_blob_dataframe(mock_blob_uri)


def test_get_blob_dataframe(storage_broker: GCStorage_broker,
                            mock_blob_uri: str, mock_blob_dataframe):
    result = storage_broker.get_blob_dataframe(mock_blob_uri)
    pd.testing.assert_frame_equal(result, mock_blob_dataframe)


def test_copy_blob(
        storage_broker: GCStorage_broker,
        mock_blob_uri,
        mock_blob_duplicated_uri):
    result = storage_broker.copy_blob(mock_blob_uri, mock_blob_duplicated_uri)
    assert result == mock_blob_duplicated_uri
    assert storage_broker.check_blob(mock_blob_duplicated_uri)


def test_rename_blob(
        storage_broker: GCStorage_broker,
        mock_blob_rename_uri,
        mock_new_name,
        mock_blob_renamed_uri):
    result = storage_broker.rename_blob(mock_blob_rename_uri, mock_new_name)
    assert result == mock_blob_renamed_uri
    assert storage_broker.check_blob(mock_blob_renamed_uri)


def test_move_blob(
        storage_broker: GCStorage_broker,
        mock_blob_move_uri,
        mock_blob_moved_uri,
        mock_blob_string):
    generate_file = storage_broker.save_from_string(
        destination_uri=mock_blob_move_uri,
        blob_string=mock_blob_string,
        overwrite=True)
    result = storage_broker.move_blob(mock_blob_move_uri, mock_blob_moved_uri)
    assert storage_broker.check_blob(result)
    assert result == mock_blob_moved_uri
    assert not storage_broker.check_blob(mock_blob_move_uri)


def test_delete_blob(
        storage_broker: GCStorage_broker,
        mock_blob_delete_uri,
        mock_blob_string):
    generate_file = storage_broker.save_from_string(
        destination_uri=mock_blob_delete_uri,
        blob_string=mock_blob_string,
        overwrite=True)
    result = storage_broker.delete_blob(mock_blob_delete_uri)
    assert result is True
    assert not storage_broker.check_blob(mock_blob_delete_uri)


def test_signed_url_for_upload(
        storage_broker: GCStorage_broker,
        mock_blob_upload_uri,
        mock_blob_uri,
        mock_local_file):
    url = storage_broker.signed_url_for_upload(mock_blob_upload_uri)
    assert url is not None
    assert storage_broker.get_blob_file(source_uri=mock_blob_uri,
                                        file_destination=mock_local_file)
    response = GCStorage_broker.upload_file_to_url(
        file_path=mock_local_file, url=url)
    assert (response.status_code == 200 or response.status_code == 204)


def test_signed_url_for_download(
        storage_broker: GCStorage_broker,
        mock_blob_upload_uri,
        mock_blob_uri,
        mock_local_file):
    url = storage_broker.signed_url_for_download(mock_blob_upload_uri)
    assert url is not None
    response = GCStorage_broker.download_file_from_url(
        file_destination=mock_local_file, url=url)
    assert os.path.exists(mock_local_file)
    assert len(response.text) > 0


def test_list_blobs_tree(storage_broker: GCStorage_broker,
                         mock_bucket_uri):
    # Assuming all your mocks have the same content type for simplicity
    source_uri = "gs://34acfe82-1318-4e7f-85f6-6af1f49aa21f_raw/"
    #source_uri = mock_bucket_uri
    result = storage_broker.list_blobs_tree(source_uri=source_uri)
    # Converting the result List[Dict] to a list of BlobDTO
    #result_blob = [api_dtos.BlobDTO(**blob) for blob in result]
    # Check if a BlobDTO with the expected blob_uri exists in the result
    assert len(result) >= 0
