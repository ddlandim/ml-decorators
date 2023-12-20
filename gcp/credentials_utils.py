from google.oauth2 import credentials, service_account
import subprocess
import google.auth.transport.requests as gauth_requests
import requests as http_requests
import os
import json
from typing import Dict, List

# Base class Singleton para classe Credentials ser acessada em qualquer
# lugar do projeto


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def clear_instances(cls):
        cls._instances = {}

# Classe Credentials para obter as credenciais de acesso ao Google Cloud
# Platform


class Credentials_broker(metaclass=Singleton):
    creds_map: Dict[str, service_account.Credentials] = {}
    service_account_type: bool = True

    def __init__(self, token: str = None,
                 filename: str = None,
                 info: dict = None,
                 escopes: List[str] = None,
                 service_account_type=True):
        filename = filename or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        info_env = os.getenv("googleapplicationcredentials")
        if info is None and info_env is not None:
            info = json.loads(info_env)
        self.service_account_type = service_account_type
        if token or filename or info:
            self.add_credentials(
                'default',
                token,
                filename,
                info,
                escopes,
                service_account_type)
        else:
            return None

    @staticmethod
    def creds_init(token: str = None,
                   filename: str = None,
                   info: dict = None,
                   escopes: List[str] = None,
                   service_account_type=True):

        if service_account_type:
            if filename:
                if os.path.exists(filename):
                    creds = service_account.Credentials.from_service_account_file(
                        filename=filename, scopes=escopes)
                    token = creds.token
                else:
                    raise Exception(
                        f"NOT FOUND: Credentials {filename} specified, but file not found")

            elif info:
                creds = service_account.Credentials.from_service_account_info(
                    info, scopes=escopes)
                token = creds.token

            else:
                creds = None
                token = None
        else:
            if token:
                token = token
                creds = credentials.Credentials(token=token)

            elif filename and os.path.exists(filename):
                creds = credentials.Credentials.from_authorized_user_file(
                    filename=filename)
                token = creds.token

            elif info:
                creds = credentials.Credentials.from_authorized_user_info(
                    info=info)
                token = creds.token

            else:
                creds = None
                token = None

        return creds, token

    def add_credentials(
            self,
            account_name: str,
            token: str = None,
            filename: str = None,
            info: dict = None,
            escopes: List[str] = None,
            service_account_type=True) -> bool:
        if self.creds_map.get(account_name):
            print("account_name already exists, use another name")
            return False
        else:
            creds, token = Credentials_broker.creds_init(
                token, filename, info, escopes, service_account_type)
            self.creds_map[account_name] = creds
            return True

    def get_credentials(
            self,
            account_name: str = 'default') -> service_account.Credentials | credentials.Credentials:
        return self.creds_map.get(account_name)

    # método get_info para obter as informações do usuário
    # TODO: tratar os erros de refresh token
    def get_info(self, account_name: str = 'default') -> dict:
        creds = self.get_credentials(account_name)
        # creds.refresh(gauth_requests.Request())
        token = creds.token
        info = {}
        if self.service_account_type:
            print("service account type")
            info["account_info"] = creds.service_account_email
        elif token:
            info["account_info"] = Credentials_broker.get_token_info(token)
        return info

    @staticmethod
    def get_gcloud_access_token() -> str:
        # Executa o comando bash para obter o access token
        result = subprocess.run(['gcloud',
                                 'auth',
                                 'application-default',
                                 'print-access-token'],
                                stdout=subprocess.PIPE,
                                text=True)
        # Obtém o access token a partir da saída do comando
        return result.stdout.strip()

    @staticmethod
    def get_token_info(access_token: str) -> dict:
        # URL da API de informações do perfil do Google
        profile_api_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = http_requests.get(profile_api_url, headers=headers)
            if response.status_code == 200:
                user_info = response.json()
                return user_info
            else:
                return None  # Falha ao obter informações do usuário

        except Exception as e:
            print(f"Erro ao obter informações do usuário: {str(e)}")
            return None
