from datetime import datetime
from google.cloud import bigquery
from google.auth.credentials import Credentials
import pandas as pd
from typing import BinaryIO, List, Dict, Any, Tuple
from storage_utils import GCStorage_broker
from google.api_core.exceptions import GoogleAPICallError

source_format_dict = {
    "PARQUET": bigquery.SourceFormat.PARQUET,
    "CSV": bigquery.SourceFormat.CSV,
    "JSON": bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    "ORC": bigquery.SourceFormat.ORC,
    "AVRO": bigquery.SourceFormat.AVRO,
    "DATASTORE_BACKUP": bigquery.SourceFormat.DATASTORE_BACKUP}

extension_dict = {
    "PARQUET": ".parquet",
    "CSV": ".csv",
    "JSON": ".json",
    "ORC": ".orc",
    "AVRO": ".avro",
    "DATASTORE_BACKUP": ".backup_info"}


class Bq_broker:
    """
    A class for interacting with Google BigQuery and Storage.

    Parameters:
    - credentials (Credentials): Google Cloud credentials. If not provided, uses application-default credentials.

    Methods:
    - check_table(table_id: str) -> bool: Check if a BigQuery table exists.
    - execute_query(query: str) -> List[Dict[Any, Any]]: Execute a BigQuery SQL query.
    - get_table_preview(table_id: str, preview_row: int) -> List[Dict[str, Any]]: Get a preview of rows from a BigQuery table.
    - calculate_cost(query: str) -> int: Estimate the cost of a BigQuery query.
    - create_native_table(table_id: str, dataframe: pd.DataFrame) -> str: Create a native BigQuery table from a DataFrame.
    - table_to_parquet(table_id: str, destination_path: str) -> str: Export a BigQuery table to Parquet format.
    - table_to_csv(table_id: str, destination_path: str) -> str: Export a BigQuery table to CSV format.
    """

    def __init__(self, creds: Credentials = None):
        """
        Initialize the BigQuery broker with the provided or default credentials.

        Parameters:
        - credentials (Credentials): Google Cloud credentials. If not provided, uses application-default credentials.
        """
        if creds:
            # Use the provided credentials to create a BigQuery and Storage
            # client.
            self.creds = creds
            self.client = bigquery.Client(credentials=creds)
            self.storage_broker = GCStorage_broker(creds)
        else:
            # If no credentials are provided, create clients without them
            # (assuming application-default credentials).
            self.client = bigquery.Client()
            self.storage_broker = GCStorage_broker(creds)

    def parse_query(self, source_query: str, namespace: str) -> str:
        """
        Parse the source query and return the final query to hide backend info from users.

        Parameters:
        - source_query (str): Source query from user.

        Returns:
        - str: Final query string.
        """
        gcp_project_id = "opg-qa-2243"

        # Convert namespace/organizationID("34acfe82-1318-4e7f-85f6-6af1f49aa21f")
        # to BQ dataset suffix ("34acfe82_1318_4e7f_85f6_6af1f49aa21f")
        dataset_suffix = self.parse_organization_id(namespace)

        # Source query sample "select * from bronze_bobs.lojas"
        if "bronze" in source_query:
            bronze_layer = "bronze" + "_" + dataset_suffix
            final_query = source_query.replace(
                "bronze", gcp_project_id + "." + bronze_layer)
        if "silver" in source_query:
            silver_layer = "silver" + "_" + dataset_suffix
            final_query = source_query.replace(
                "silver", gcp_project_id + "." + silver_layer)
        if "gold" in source_query:
            gold_layer = "gold" + "_" + dataset_suffix
            final_query = source_query.replace(
                "gold", gcp_project_id + "." + gold_layer)

        return final_query

    def parse_organization_id(self, organization_id: str) -> str:
        """
        Parse the organization id from auth0 to BQ dataset suffix name.

        Parameters:
        - organization_id (str): organization id from auth flow.

        Returns:
        - str: Final BQ dataset suffix name string.
        """

        # organization_id sample "34acfe82-1318-4e7f-85f6-6af1f49aa21f"
        dataset_suffix = organization_id.replace("-", "_")

        return dataset_suffix

    def check_table(self, table_id: str) -> bool:
        """
        Check if a BigQuery table exists.

        Parameters:
        - table_id (str): Full BigQuery table ID (project.dataset.table).
            Example: "bigquery-public-data.samples.shakespeare"

        Returns:
        - bool: True if the table exists, False otherwise.
        """
        client = self.client
        try:
            client.get_table(table_id)
            return True
        except GoogleAPICallError as e:
            if e.code == 404:
                return False
            else:
                raise e
        except Exception as e:
            raise e

    def check_dataset(self, dataset_id: str) -> bool:
        """
        Check if a BigQuery dataset exists.

        Parameters:
        - dataset_id (str): Full BigQuery dataset ID (project.dataset).
            Example: "bigquery-public-data.samples"

        Returns:
        - bool: True if the dataset exists, False otherwise.
        """
        client = self.client
        try:
            client.get_dataset(dataset_id)
            return True
        except GoogleAPICallError as e:
            if e.code == 404:
                return False
            else:
                raise e
        except Exception as e:
            raise e

    def execute_query(self, query: str) -> List[Dict[Any, Any]]:
        """
        Execute a BigQuery SQL query.

        Parameters:
        - query (str): The SQL query string.
            Example: "SELECT * FROM `bigquery-public-data.samples.shakespeare`"

        Returns:
        - List[Dict[Any, Any]]: List of dictionaries representing the query results.
            Example: [{'word': 'honey', 'word_count': 1}, {'word': 'honey', 'word_count': 1}, ...]
        """
        query_job = self.client.query(query)
        results = query_job.result()

        # Convert the results to a list of dictionaries for JSON serialization.
        results_list = [dict(row) for row in results]

        return results_list

    def get_table_preview(self, table_id: str,
                          preview_row: int) -> List[Dict[str, Any]]:
        """
        Get a preview of rows from a BigQuery table.

        Parameters:
        - table_id (str): Full BigQuery table ID (project.dataset.table).
        - preview_row (int): Number of rows to preview.

        Returns:
        - List[Dict[str, Any]]: List of dictionaries representing the previewed rows.
        """
        client = self.client

        rows_iter = client.list_rows(table_id, max_results=preview_row)
        results_list = [dict(row) for row in rows_iter]

        return results_list

    def calculate_cost(self, query: str) -> int:
        """
        Estimate the cost of a BigQuery query.

        Parameters:
        - query (str): The SQL query string. Example: "SELECT * FROM `bigquery-public-data.samples.shakespeare`"

        Returns:
        - int: Estimated cost in bytes of processing the query. Example: 1000000000
        """
        job_config = bigquery.QueryJobConfig()
        job_config.dry_run = True
        job_config.use_query_cache = False
        query_job = self.client.query(
            query,
            job_config=job_config,
        )
        total_bytes_processed = query_job.total_bytes_processed

        return total_bytes_processed

    def create_native_table(self, dataframe_csv_uri: str, table_id: str,
                            schema: List[Tuple] = None,
                            write_disposition: str = "WRITE_EMPTY",
                            source_format: str = "PARQUET"
                            ) -> Dict[str, str | int | float]:
        """
        Create a native BigQuery table from a csv file with pandas DataFrame csv loader.

        Parameters:
        - dataframe_csv_uri (str) required: Uri of any file to import into BigQuery table.
            Example: "gs://zion_native_storage_poc/bronze_bobs/empresas_t.csv"
        - table_id (str) required: Full BigQuery table ID (project.dataset.table).
            Example: "zion_native_storage_poc.bronze_bobs.empresas_t"
        - schema (List[Tuple]) optional: List of tuples with column name and type.
            Example: [("name", "STRING"), ("age", "INTEGER")]
            default: None
        - write_disposition (str) optional: Write disposition for the table.
            Example: "WRITE_TRUNCATE" for overwrite, "WRITE_APPEND" for append, "WRITE_EMPTY" for error if table exists.
            default: "WRITE_EMPTY"  if not provided or different from the options.
        - source_format (str) optional: Source format for the table.
            Example: "PARQUET" for parquet or "CSV" or "JSON" or "ORC" or "AVRO" or "DATASTORE_BACKUP".
            default: "PARQUET" if not provided or different from the options.

        Returns:
        - Dict: Job ID info
        """
        client = self.client

        # table_id = project + '.' + dataset + '.' + table_name
        project_id = table_id.split('.')[0]
        dataset_id = table_id.split('.')[1]
        table_name = table_id.split('.')[2]

        table_ref = bigquery.DatasetReference(
            project_id, dataset_id).table(table_name)
        job_config = bigquery.LoadJobConfig()
        write_dict = {
            "WRITE_TRUNCATE": bigquery.WriteDisposition.WRITE_TRUNCATE,
            "WRITE_APPEND": bigquery.WriteDisposition.WRITE_APPEND,
            "WRITE_EMPTY": bigquery.WriteDisposition.WRITE_EMPTY}
        job_config.write_disposition = write_dict.get(
            write_disposition, bigquery.WriteDisposition.WRITE_EMPTY)
        if schema is None:
            job_config.autodetect = True
        else:
            job_config.schema = schema
        job_config.source_format = source_format_dict.get(
            source_format, bigquery.SourceFormat.PARQUET)
        load_job = client.load_table_from_uri(
            dataframe_csv_uri,
            table_ref,
            job_config=job_config
        )
        job_result = load_job.result()
        # Waits for table load to complete.
        response = job_result.__dict__
        return response

    def append_native_table(
            self,
            dataframe_csv_uri: str,
            table_id: str) -> str:
        """
        Create a native BigQuery table from a csv file with pandas DataFrame csv loader,
        append new table in existing table.

        Parameters:
        - csv_uri (str): Uri of csv file to convert into BigQuery table.
        - table_id (str): Full BigQuery table ID (project.dataset.table).

        Returns:
        - str: Job ID of the table creation process.
        """
        client = self.client

        dataframe = self.storage_broker.get_blob_dataframe(dataframe_csv_uri)

        # table_id = project + '.' + dataset + '.' + table_name
        project_id = table_id.split('.')[0]
        dataset_id = table_id.split('.')[1]
        table_name = table_id.split('.')[2]

        table_ref = bigquery.DatasetReference(
            project_id, dataset_id).table(table_name)

        job_config = bigquery.LoadJobConfig()
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

        load_job = client.load_table_from_dataframe(
            dataframe, table_ref, job_config=job_config)

        job_result = load_job.result()

        # Waits for table load to complete.
        return str(job_result.job_id)

    def create_native_table_from_file_obj(
            self,
            fileobj: BinaryIO,
            table_id: str,
            structured: bool = False) -> str:
        """
        Create a native BigQuery table from a file object.

        Parameters:
        - fileobj (fileobj): File object to convert into BigQuery table.
        - table_id (str): Full BigQuery table ID (project.dataset.table).
        - structured (bool): If True, the file is assumed to be structured to imported as a structured table, if not, the file ill be structured with pandas first.
        Returns:
        - str: Job ID of the table creation process.
        """
        client = self.client

        if not structured:
            dataframe = pd.read_csv(fileobj)

            # table_id = project + '.' + dataset + '.' + table_name
            project_id = table_id.split('.')[0]
            dataset_id = table_id.split('.')[1]
            table_name = table_id.split('.')[2]

            table_ref = bigquery.DatasetReference(
                project_id, dataset_id).table(table_name)

            job_config = bigquery.LoadJobConfig()
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

            load_job = client.load_table_from_dataframe(
                dataframe, table_ref, job_config=job_config)

            # Waits for table load to complete.
            return str(load_job.result().job_id)

    def export_table(
            self,
            table_id: str,
            destination_path: str,
            format: str = "PARQUET") -> str:
        """
        Export a BigQuery table to Parquet format.

        Parameters:
        - table_id (str): Full BigQuery table ID (project.dataset.table).
        - destination_path (str): Destination path for the exported file.
        - format (str): Format of the exported file. Options: "PARQUET", "CSV", "JSON", "ORC", "AVRO", "DATASTORE_BACKUP".

        Returns:
        - str: Job ID of the export process.
        """
        client = self.client
        format = format.upper()

        # table_id = project + '.' + dataset + '.' + table_name
        project_id = table_id.split('.')[0]
        dataset_id = table_id.split('.')[1]
        table_name = table_id.split('.')[2]
        table_ref = bigquery.DatasetReference(
            project_id, dataset_id).table(table_name)

        # Set export file format
        job_config = bigquery.job.ExtractJobConfig()
        job_config.destination_format = source_format_dict.get(format)
        
        if not destination_path.endswith("/"):
            destination_path = destination_path + "/"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_path = destination_path + timestamp + "/"
        # Set export destination uri
        destination_uri = destination_path + \
            table_name + extension_dict.get(format)
        extract_job = client.extract_table(
            table_ref,
            destination_uri,
            job_config=job_config,
            location="US",
        )  # API request
        return str(extract_job.result().job_id)

    def list_datasets(self):
        """
        List all datasets which current SA has read permission.

        Parameters:
        - No parameter for poc version
        - Should receive user_id/client_id for access control

        Returns:
        - list[str]: list of dataset names.
        """
        client = self.client

        datasets = list(client.list_datasets())
        dataset_ids = []

        for dataset in datasets:
            dataset_ids.append(dataset.dataset_id)

        return dataset_ids

    def list_tables(self, dataset_id: str) -> List[str]:
        """
        List all tables from given dataset.

        Parameters:
        - dataset_id (str): Full BigQuery dataset ID (project.dataset).
        - Should receive user_id/client_id for access control

        Returns:
        - list[str]: list of table names.
        """
        client = self.client

        tables = client.list_tables(dataset_id)
        table_ids: List[str] = []

        for table in tables:
            table_ids.append(str(table.table_id))

        return table_ids

    def get_table_info(self, table_id: str):
        """
        Return all metadata from given table.

        Parameters:
        - table_id (str): Full BigQuery dataset ID (project.dataset.table).
        - Should receive user_id/client_id for access control

        Returns:
        - Dict{}: dict of table infos.
        """
        client = self.client

        # table_id = project + '.' + dataset + '.' + table_name
        table = client.get_table(table_id)

        result_dict = {
            "created": str(table.modified),
            "description": str(table.description),
            "modified": str(table.modified),
            "num_bytes": table.num_bytes,
            "num_rows": table.num_rows
        }

        return result_dict

    def delete_table(self, table_id: str):
        client = self.client
        client.delete_table(table_id, not_found_ok=True)

    def create_bsg_datasets(self, organization_id: str) -> str:
        client = self.client

        project_id = client.project

        for layer in ["bronze_", "silver_", "gold_"]:
            dataset_id = layer + self.parse_organization_id(organization_id)
            full_dataset_id = project_id + "." + dataset_id

            # Construct a full Dataset object to send to the API.
            dataset = bigquery.Dataset(full_dataset_id)
            dataset.location = "US"

            # Send the dataset to the API for creation, with an explicit timeout.
            # Raises google.api_core.exceptions.Conflict if the Dataset already
            # exists within the project.
            dataset = client.create_dataset(dataset, timeout=30)

        return "Datasets created."
