import os
from urllib.parse import quote_plus

import boto3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

class DatabaseManager:
    def __init__(self):

        # Fabric Lakehouse_from_DWH_New (SQL Server) ---------------------
        fabric_host = os.getenv("FABRIC_SQL_ENDPOINT")
        fabric_db = os.getenv("FABRIC_DATABASE")
        fabric_user = os.getenv("FABRIC_USERNAME")
        fabric_password = os.getenv("FABRIC_PASSWORD")

        fabric_connection_string = (
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server={fabric_host},1433;"
            f"Database={fabric_db};"
            "Encrypt=Yes;"
            "TrustServerCertificate=Yes;"
            "Authentication=ActiveDirectoryPassword;"
            f"UID={fabric_user};"
            f"PWD={fabric_password};"
        )

        fabric_params = quote_plus(fabric_connection_string)
        self.sqlserver_url = f"mssql+pyodbc:///?odbc_connect={fabric_params}"
        self.sqlserver_engine = create_engine(self.sqlserver_url, pool_size=5)

        # Supabase (PostgreSQL) ---------------------
        pg_user = os.getenv("user")
        pg_password = os.getenv("password")
        pg_host = os.getenv("host")
        pg_port = os.getenv("port", "5432")
        pg_dbname = os.getenv("dbname", "postgres")

        self.supabase_url = (
            f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_dbname}"
        )
        self.supabase_engine = create_engine(self.supabase_url, pool_size=5)

        # Supabase Bucket (S3) ---------------------
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("SUPABASE_AWS_KEY_ID"),
            aws_secret_access_key=os.getenv("SUPABASE_AWS_SECRET_KEY"),
            region_name=os.getenv("SUPABASE_REGION"),
            endpoint_url=os.getenv("SUPABASE_URL"),
        )

    def get_sqlserver_session(self):
        Session = sessionmaker(bind=self.sqlserver_engine)
        return Session()

    def get_supabase_session(self):
        Session = sessionmaker(bind=self.supabase_engine)
        return Session()

    # Backward-compatible alias.
    def get_postgres_session(self):
        return self.get_supabase_session()

    def get_s3(self):
        return self.s3_client


# สร้างเป็น Global Instance เพื่อให้ Reuse connection pool ได้จริง
db_manager = DatabaseManager()

if __name__ == "__main__":
    # สำหรับ test connection

    print("Testing SQL Server connection...")
    try:
        with db_manager.sqlserver_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("SQL Server connection OK, SELECT 1 result:", result.scalar())
    except Exception as e:
        print("SQL Server connection failed:", e)

    print("Testing PostgreSQL (Supabase) connection...")
    try:
        with db_manager.supabase_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("PostgreSQL connection OK, SELECT 1 result:", result.scalar())
    except Exception as e:
        print("PostgreSQL connection failed:", e)

    print("Testing S3 Client connection (Supabase Bucket)...")
    try:
        response = db_manager.s3_client.list_buckets()
        print("S3 Client connection OK, Buckets:", response.get('Buckets', []))
    except Exception as e:
        print("S3 Client connection failed:", e)