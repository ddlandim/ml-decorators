from datetime import datetime
from pydantic import BaseModel, Field, validator
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from typing import List, Dict, Any, Tuple
from credentials_utils import Credentials_broker
from storage_utils import GCStorage_broker
from bigquery_utils import Bq_broker
from google.api_core.exceptions import GoogleAPICallError
from typing import List, Dict, Any
from fastapi import HTTPException
from google.api_core.exceptions import GoogleAPICallError

### Return types, Errors and Exceptions #


class SchemaItem(BaseModel):
    name: str
    type: str


class HttpResponseError(BaseModel):
    data: str = ""
    error: bool = True
    message: str = ""
    statusCode: int = 500


class ApiIOHandler:
    __doc__ = """
        200 OK: The request has succeeded.
        201 Created: The request has been fulfilled and has resulted in one or more new resources being created.
        400 Bad Request: The server could not understand the request due to invalid syntax.
        401 Unauthorized: The request requires user authentication.
        403 Forbidden: The server understood the request, but it refuses to authorize it.
        404 Not Found: The server can't find the requested resource.
        500 Internal Server Error: The server encountered an unexpected condition which prevented it from fulfilling the request.
        502 Bad Gateway: The server was acting as a gateway or proxy and received an invalid response from the upstream server. This is used when there is a GoogleAPICallError.
    """

    def __init__(self, creds=None):
        if creds:
            self.creds = creds
        else:
            creds_obj = Credentials_broker()
            self.creds = creds_obj.get_credentials()
        self.bq = Bq_broker(self.creds)
        self.gcs = GCStorage_broker(self.creds)

    @staticmethod
    def handle_exceptions(e):
        if isinstance(e, GoogleAPICallError):
            errors = str(e.errors)
            return HttpResponseError(
                error=True, message=errors, statusCode=e.code)
        elif isinstance(e, HTTPException):
            return HttpResponseError(
                error=True,
                message=e.detail,
                statusCode=e.status_code)
        else:
            message = str(e)
            return HttpResponseError(
                error=True, message=message, statusCode=500)

    @staticmethod
    def mock_foundation_validate_operation(operation, user_id) -> bool:
        return True

    @staticmethod
    def mock_foundation_validate_user(user_id) -> bool:
        return True

    def validate_write_disposition(self, write_disposition: str) -> str:
        if write_disposition not in [
            'WRITE_TRUNCATE',
            'WRITE_APPEND',
                'WRITE_EMPTY']:
            raise HTTPException(
                status_code=400,
                detail=f"write_disposition must be one of WRITE_TRUNCATE, WRITE_APPEND or WRITE_EMPTY")
        return write_disposition

    def validate_source_format(self, source_format: str) -> str:
        if source_format not in [
            'PARQUET',
            'CSV',
            'JSON',
            'AVRO',
            'ORC',
                'DATASTORE_BACKUP']:
            raise HTTPException(
                status_code=400,
                detail=f"source_format must be one of CSV, JSON or AVRO")
        return source_format

    @staticmethod
    def validate_table_schema(
            schema: List[SchemaItem] = None) -> List[Tuple[str, str]]:
        schema_tuples = [(item.name, item.type)
                         for item in schema] if schema else None
        return schema_tuples

    def validate_table_id(self, table_id, write_disposition=None) -> str:
        if write_disposition:
            if write_disposition != 'WRITE_EMPTY' and not self.bq.check_table(
                    table_id):
                raise HTTPException(
                    status_code=404,
                    detail=f" Impossible to append or overwirte, table {table_id} does not exist")
            if write_disposition == 'WRITE_EMPTY' and self.bq.check_table(
                    table_id):
                raise HTTPException(
                    status_code=400,
                    detail=f" Impossible to WRITE_EMPTY, table {table_id} already exists")
        else:
            if not self.bq.check_table(table_id):
                raise HTTPException(
                    status_code=404,
                    detail=f"Validate Table ID : Table {table_id} does not exist")
        return table_id

    def validate_dataset_id(self, dataset_id, preview_row: int = None) -> str:
        if not self.bq.check_dataset(dataset_id):
            raise HTTPException(status_code=404,
                                detail=f"Dataset {dataset_id} does not exist")
        if preview_row and preview_row < 1:
            raise HTTPException(status_code=400,
                                detail=f"preview_row must be greater than 0")
        return dataset_id

    def validate_query(self, query: str) -> str:
        table_id = query.split('FROM ')[1].split(' ')[0]
        self.validate_table_id(table_id)
        return query

    def validate_blob_uri(self, uri: str, source_format: str = None) -> str:
        if not self.gcs.check_blob(uri):
            raise HTTPException(
                status_code=404,
                detail=f"Blob {uri} does not exist")
        if source_format:
            if not uri.__contains__('.'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Validate {uri}, {source_format} : Blob uri does not have an extension")
            if uri.endswith('/') and source_format != 'folder':
                raise HTTPException(
                    status_code=400,
                    detail=f"Blob {uri} is a folder, not a file")
            elif uri.split('.')[-1] != source_format.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Blob {uri} is not of type {source_format}")
        return uri

    def validate_operation(self, operation, user_id):
        # Check if operation is supported with foundation
        if not self.mock_foundation_validate_operation(operation, user_id):
            raise HTTPException(status_code=403,
                                detail=f"Operation {operation} not supported")
        return operation

    def authorize_user(self, user_id):
        if not self.mock_foundation_validate_user(user_id):
            raise HTTPException(
                status_code=401,
                detail=f"User {user_id} not authorized")
        return user_id

# End Return types, Errors and Exceptions ###


### API DTOs#
# Input DTOs #
apiiohandler = ApiIOHandler()


class TableParam(BaseModel):
    table_id: str

    @validator('table_id')
    def validate_table_id(cls, table_id):
        return apiiohandler.validate_table_id(table_id)

# Output DTOs #


class HttpResponseBool(BaseModel):
    data: bool = None
    error: bool = False
    message: str = ""
    statusCode: int = 200


class HttpResponseInt(BaseModel):
    data: int = None
    error: bool = False
    message: str = ""
    statusCode: int = 200


class HttpResponseStr(BaseModel):
    data: str = None
    error: bool = False
    message: str = ""
    statusCode: int = 200


class HttpResponseLStr(BaseModel):
    data: List[str] = None
    error: bool = False
    message: str = ""
    statusCode: int = 200


class BlobDTO(BaseModel):
    name: str = "some_path/blob_name.csv"
    content_type: str = "text/csv"
    uri: str = "gs://bucket_name/some_path/blob_name.csv"
    time_created: str = "2021-01-01T00:00:00.000Z"
    time_updated: str = "2021-01-01T00:00:00.000Z"
    size: int = 60
    md5_hash: str = "UAxYvloWgc39V/0rqxyBjQ=="  # some hash


class HttpResponseLBlob(BaseModel):
    data: List[BlobDTO]
    error: bool = False
    message: str = ""
    statusCode: int = 200

    class Config:
        arbitrary_types_allowed = True


class HttpResponseTable(BaseModel):
    data: List[Dict[
        str | float | int,
        str | float | int | None
    ]] = None
    error: bool = False
    message: str = ""
    statusCode: int = 200


class BigQueryJobDTO(BaseModel):
    jobId: str
    state: str
    sourceUris: List[str]
    destination_dataset: str
    destination_table: str
    writeDisposition: str
    sourceFormat: str
    inputFiles: int
    inputFileBytes: int
    outputRows: int
    outputBytes: int
    badRecords: int
    creationTime: float
    startTime: float
    endTime: float

    @classmethod
    def from_load_job_result(cls, job_result: Dict):
        jobId = job_result.get('_properties', {})\
            .get('jobReference', {})\
            .get('jobId', {})
        state = job_result.get('_properties', {})\
            .get('status', {})\
            .get('state', {})
        # configuration load properties
        configuration_load: dict = job_result.get('_properties', {})\
            .get('configuration', {})\
            .get('load', {})
        sourceUris = configuration_load.get('sourceUris', {})
        destination_dataset = configuration_load.get('destinationTable', {})\
            .get('datasetId', {})
        destination_table = configuration_load.get('destinationTable', {})\
            .get('tableId', {})
        writeDisposition = configuration_load.get('writeDisposition', {})
        sourceFormat = configuration_load.get('sourceFormat', {})

        # statistics properties
        statistics: dict = job_result.get('_properties', {})\
            .get('statistics', {})
        creationTime = statistics.get('creationTime', {})
        startTime = statistics.get('startTime', {})
        endTime = statistics.get('endTime', {})
        statistics_load: dict = statistics.get('load', {})
        inputFiles = statistics_load.get('inputFiles', {})
        inputFileBytes = statistics_load.get('inputFileBytes', {})
        outputRows = statistics_load.get('outputRows', {})
        outputBytes = statistics_load.get('outputBytes', {})
        badRecords = statistics_load.get('badRecords', {})

        return cls(
            jobId=jobId,
            state=state,
            sourceUris=sourceUris,
            destination_dataset=destination_dataset,
            destination_table=destination_table,
            writeDisposition=writeDisposition,
            sourceFormat=sourceFormat,
            inputFiles=inputFiles,
            inputFileBytes=inputFileBytes,
            outputRows=outputRows,
            outputBytes=outputBytes,
            badRecords=badRecords,
            creationTime=creationTime,
            startTime=startTime,
            endTime=endTime
        )


class HttpResponseBqJob(BaseModel):
    data: BigQueryJobDTO
    error: bool = False
    message: str = ""
    statusCode: int = 200

    class Config:
        arbitrary_types_allowed = True


class RecursiveBqJobItem(BaseModel):
    fileUri: str
    jobId: str


class HttpResponseBqJobRecursive(BaseModel):
    data: List[RecursiveBqJobItem] = []
    error: bool = False
    message: str = ""
    statusCode: int = 200

    class Config:
        arbitrary_types_allowed = True


class TableInfo(BaseModel):
    created: str
    description: str
    modified: str
    num_bytes: int
    num_rows: int


class HttpResponseTableInfo(BaseModel):

    data: TableInfo
    error: bool = False
    message: str = ""
    statusCode: int = 200

    class Config:
        arbitrary_types_allowed = True


class FolderItem(BaseModel):
    Folder: str
    path: str
    subFolders: List = []

# FolderItem.__pydantic_extra__.model  __pydantic_model__.model_rebuild()


class HttpResponseFolderTree(BaseModel):
    data: Any
    error: bool = False
    message: str = ""
    statusCode: int = 200

    class Config:
        arbitrary_types_allowed = True
