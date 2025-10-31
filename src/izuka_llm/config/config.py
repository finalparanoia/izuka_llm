from pydantic import AnyHttpUrl, AnyUrl
from pydantic_settings import BaseSettings


class SettingsEntity(BaseSettings):
    llm_endpoint: AnyHttpUrl
    llm_token: str
    sql_db_url: AnyUrl
    mongo_url: AnyUrl
    vector_url: AnyUrl
    s3_endpoint: AnyHttpUrl
    S3_access_key: str
    s3_secret_key: str
