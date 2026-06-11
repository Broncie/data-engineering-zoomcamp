#!/usr/bin/env python
# coding: utf-8

import click
import pandas as pd
from sqlalchemy import create_engine
from tqdm.auto import tqdm
import pyarrow.parquet as pq
from pathlib import Path
import math

dtype = {
    "VendorID": "Int64",
    "store_and_fwd_flag": "string",
    "RatecodeID": "Int64",
    "PULocationID": "Int64",
    "DOLocationID": "Int64",
    "passenger_count": "Int64",
    "trip_distance": "float64",
    "fare_amount": "float64",
    "extra": "float64",
    "mta_tax": "float64",
    "tip_amount": "float64",
    "tolls_amount": "float64",
    "ehail_fee": "float64",
    "improvement_surcharge": "float64",
    "total_amount": "float64",
    "payment_type": "Int64",
    "trip_type": "float64",
    "congestion_surcharge": "float64",
    "cbd_congestion_fee": "float64"
}

zones_dtype = {
    "LocationID": "Int64",
    "Borough": "string",
    "Zone": "string",
    "service_zone": "string",
}

datetime_cols = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
]

def apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Apply schema only to the Parquet trip data."""

    for col, col_type in dtype.items():
        if col in df.columns:
            if col_type in ["Int64", "float64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(col_type)
            else:
                df[col] = df[col].astype(col_type)

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df

@click.command()
@click.option('--pg-user', default='root', help='PostgreSQL user')
@click.option('--pg-pass', default='root', help='PostgreSQL password')
@click.option('--pg-host', default='localhost', help='PostgreSQL host')
@click.option('--pg-port', default=5432, type=int, help='PostgreSQL port')
@click.option('--pg-db', default='ny_taxi', help='PostgreSQL database name')
@click.option('--year', default=2025, type=int, help='Year of the data')
@click.option('--month', default=11, type=int, help='Month of the data')
@click.option('--target-table', default='green_taxi_data', help='Target table name')
@click.option('--zones-table', default='zones', help='Target table for taxi zone lookup data')
@click.option('--chunksize', default=100000, type=int, help='Chunk size for reading CSV')
def run(pg_user, pg_pass, pg_host, pg_port, pg_db, year, month, target_table, zones_table, chunksize):
    """Ingest NYC taxi data into PostgreSQL database."""
    
    url1 = (
        "https://github.com/DataTalksClub/nyc-tlc-data/"
        "releases/download/misc/taxi_zone_lookup.csv"
    )

    engine = create_engine(
        f'postgresql+psycopg://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}'
        )
    
    # Insert Parquet data in chunks after applying schema

    local_parquet_file = Path(
    f"/workspaces/data-engineering-zoomcamp/pipeline/"
    f"green_tripdata_{year}-{month:02d}.parquet"
    )

    print(f"Reading Parquet file from: {local_parquet_file}")
    print(f"File exists: {local_parquet_file.exists()}")

    if not local_parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {local_parquet_file}")

    print(f"File size: {local_parquet_file.stat().st_size} bytes")

    parquet_file = pq.ParquetFile(local_parquet_file)

    print(f"Parquet rows: {parquet_file.metadata.num_rows}")
    print(f"Parquet columns: {parquet_file.schema.names}")

    if parquet_file.metadata.num_rows == 0:
        raise ValueError(f"Parquet file has 0 rows: {local_parquet_file}")
    
    total_rows = parquet_file.metadata.num_rows
    total_batches = math.ceil(total_rows / chunksize)

    # Insert CSV zone data in chunks
    first_zones_chunk = True

    df_iter = pd.read_csv(url1, dtype=zones_dtype, iterator = True, chunksize=chunksize)

    for df_chunk in tqdm(df_iter):

        df_chunk.to_sql(
            name=zones_table,
            con=engine,
            if_exists="replace" if first_zones_chunk else "append",
            index=False,
        )

        first_zones_chunk = False
    

    # Insert Parquet trip data in chunks
    first_trips_chunk = True

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=chunksize),
        total=total_batches,
        desc="Inserting Parquet data",
    ):
        df_chunk = batch.to_pandas()
        df_chunk = apply_schema(df_chunk)

        df_chunk.to_sql(
        name=target_table,
        con=engine,
        if_exists="replace" if first_trips_chunk else "append",
        index=False,
        chunksize=5000,
    )

        first_trips_chunk = False


if __name__ == '__main__':
    run()