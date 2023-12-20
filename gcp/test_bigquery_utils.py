import pytest
from google.auth.credentials import AnonymousCredentials
from io import StringIO
from google.cloud import storage
from bigquery_utils import Bq_broker
from storage_utils import GCStorage_broker
from credentials_utils import Credentials_broker
import bigquery_router
import pandas as pd
import storage_router
import api_dtos
import os
import sys
from typing import List, Dict


@pytest.fixture
def bigquey_broker() -> Bq_broker:
    # Set GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account key file
    # If you are in VSCODE you can set it in .vscode/launch.json to run tests
    # in debug mode
    key_path = os.path.abspath("/home/douglas/keys/key.json")
    Credentials_broker.clear_instances()  # Clear instances of Credentials_broker
    creds_broker = Credentials_broker(filename=key_path)
    creds = creds_broker.get_credentials()
    return Bq_broker(creds=creds)

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
def mock_parquet_file() -> str:
    return "gs://zion-poc-raw/tests/empresas_t.parquet"


@pytest.fixture
def mock_csv_file() -> str:
    return "gs://zion-poc-raw/Iris.csv"


@pytest.fixture
def mock_csv_iris_recursive_files() -> str:
    return "gs://zion-poc-raw/tests/irisrecursive/"


@pytest.fixture
def mock_parquet_table() -> str:
    return "opg-qa-2243.zion_native_storage_poc.empresas_t"


@pytest.fixture
def mock_csv_table() -> str:
    return "opg-qa-2243.zion_native_storage_poc.iris"


@pytest.fixture
def mock_iris_recursive_table() -> str:
    return "opg-qa-2243.zion_native_storage_poc.iris_recursive"


@pytest.fixture
def gcs_broked_csv() -> str:
    return "gs://34acfe82-1318-4e7f-85f6-6af1f49aa21f_raw/GOVBR_EMPRESA_500k.csv"


@pytest.fixture
def bq_table_id_broked() -> str:
    return "opg-qa-2243.zion_native_storage_poc.500kbroked"


def test_create_native_table_parquet(
        bigquey_broker: Bq_broker,
        mock_parquet_file: str,
        mock_parquet_table: str):
    test_table = mock_parquet_table + "_test"
    # Create a native BigQuery table from a parquet file
    result = bigquey_broker.create_native_table(
        dataframe_csv_uri=mock_parquet_file,
        table_id=test_table,
        write_disposition="WRITE_TRUNCATE")
    # Check if the table exists
    assert bigquey_broker.check_table(test_table)
    # Delete the table
    # bigquey_broker.delete_table(test_table)
    # Check if the table was deleted
    # assert bigquey_broker.check_table(test_table) == False


def test_create_native_table_csv(
        bigquey_broker: Bq_broker,
        mock_csv_file: str,
        mock_csv_table: str):
    test_table = mock_csv_table + "_test"
    # Check if the table exists
    if bigquey_broker.check_table(test_table):
        bigquey_broker.delete_table(test_table)
    # Create a native BigQuery table from a parquet file
    job_dict = bigquey_broker.create_native_table(
        dataframe_csv_uri=mock_csv_file,
        table_id=test_table,
        write_disposition="WRITE_TRUNCATE",
        source_format="CSV")
    # Check DTO Convertion
    job_dto = api_dtos.BigQueryJobDTO.from_load_job_result(job_dict)
    # Check if the table exists
    assert bigquey_broker.check_table(test_table)
    # Delete the table
    # bigquey_broker.delete_table(test_table)
    # Check if the table was deleted
    # assert bigquey_broker.check_table(test_table) == False


def test_create_native_table_csv_broked(
        bigquey_broker: Bq_broker,
        gcs_broked_csv: str,
        bq_table_id_broked: str):
    test_table = bq_table_id_broked + "_test"
    # Check if the table exists
    if bigquey_broker.check_table(test_table):
        bigquey_broker.delete_table(test_table)
    try:
        # Create a native BigQuery table from a parquet file
        response = bigquey_broker.create_native_table(
            dataframe_csv_uri=gcs_broked_csv,
            table_id=test_table,
            write_disposition="WRITE_EMPTY",
            source_format="CSV")
    except Exception as e:
        response_error = api_dtos.ApiIOHandler.handle_exceptions(e)
        assert response_error.error
        assert response_error.statusCode == 400


def test_create_native_recursive_iris(
        bigquey_broker: Bq_broker,
        mock_csv_iris_recursive_files: str,
        mock_iris_recursive_table: str):
    test_table = mock_iris_recursive_table + "_test"
    # Check if the table exists
    if bigquey_broker.check_table(test_table):
        bigquey_broker.delete_table(test_table)
    response = bigquery_router.bq_create_native_table_recursive(
        dataframe_dir_uri=mock_csv_iris_recursive_files,
        table_id=test_table,
        source_format="CSV"
    )
    table_creation = bigquey_broker.check_table(test_table)
    assert table_creation


def test_create_native_table_dto(
        bigquey_broker: Bq_broker,
        mock_parquet_file: str,
        mock_parquet_table: str):
    test_table = mock_parquet_table + "_test"
    # Create a native BigQuery table from a parquet file
    result = bigquey_broker.create_native_table(
        dataframe_csv_uri=mock_parquet_file,
        table_id=test_table,
        write_disposition="WRITE_TRUNCATE")
    # Delete the table
    bigquey_broker.delete_table(test_table)
    # check DTO constructor
    dto: api_dtos.BigQueryJobDTO = api_dtos.BigQueryJobDTO.from_load_job_result(
        result)
    print("\n utils dict returned: ", result)
    print("\n dto : ", dto)
    assert isinstance(dto, api_dtos.BigQueryJobDTO)
    http_response = api_dtos.HttpResponseBqJob(data=dto)
    assert isinstance(http_response, api_dtos.HttpResponseBqJob)


def test_export_native_table(bigquey_broker: Bq_broker, storage_broker: GCStorage_broker):
    table_to_export = "bronze.empre_x_estab2"
    destination_uri = "raw"
    format = "PARQUET"
    namespace = "34acfe82-1318-4e7f-85f6-6af1f49aa21f"
    # Export a native BigQuery table to a parquet file
    table_id = bigquey_broker.parse_query(table_to_export, namespace)
    destination_uri = storage_broker.parse_organization_id(destination_uri, namespace)
    ###

    table_id = api_dtos.apiiohandler.validate_table_id(table_id)
    destination_uri = api_dtos.apiiohandler.validate_blob_uri(destination_uri)
    format = api_dtos.apiiohandler.validate_source_format(format)
    ###
    result = api_dtos.HttpResponseStr(
        data= bigquey_broker.export_table(
                                            table_id = table_id,
                                            destination_path = destination_uri,
                                            format = format))
    assert result.statusCode == 200