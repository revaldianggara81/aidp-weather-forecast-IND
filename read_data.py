import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_WALLET_LOCATION = os.getenv("DB_WALLET_LOCATION")
DB_WALLET_PASSWORD = os.getenv("DB_WALLET_PASSWORD")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME")

connection = oracledb.connect(
    user=DB_USER,
    password=DB_PASS,
    dsn=CONNECTION_STRING,
    config_dir=DB_WALLET_LOCATION,
    wallet_location=DB_WALLET_LOCATION,
    wallet_password=DB_WALLET_PASSWORD,
)

print(f"Connected to Oracle ADB. Reading table: {TABLE_NAME}\n")

with connection.cursor() as cursor:
    cursor.execute(f"SELECT * FROM {TABLE_NAME} FETCH FIRST 10 ROWS ONLY")

    columns = [col[0] for col in cursor.description]
    print("\t".join(columns))
    print("-" * 80)

    for row in cursor:
        print("\t".join(str(val) for val in row))

connection.close()
print("\nDone.")
