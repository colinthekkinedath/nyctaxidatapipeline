#!/usr/bin/env python3

import argparse
import time
from contextlib import contextmanager

from pyspark.sql import SparkSession, functions as F


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"{label} took {time.perf_counter() - t0:,.2f} s")


def _write_mysql(df, url, table, user, pw, mode):
    (df.write.format("jdbc")
       .option("url", url)
       .option("dbtable", table)
       .option("user", user)
       .option("password", pw)
       .option("driver", "com.mysql.cj.jdbc.Driver")
       .mode(mode)
       .save())


def main(parquet_path, mysql_url, mysql_user, mysql_pw, overwrite_outputs, skip_mysql):
    mode = "overwrite" if overwrite_outputs else "append"
    spark = SparkSession.builder.appName("YellowTaxiAnalytics").getOrCreate()

    with timed("Load + feature-engineering"):
        df = (
            spark.read.parquet(parquet_path)
            .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
            .withColumn("pickup_dow", F.dayofweek("tpep_pickup_datetime"))
            .withColumn(
                "trip_minutes",
                (
                    F.unix_timestamp("tpep_dropoff_datetime")
                    - F.unix_timestamp("tpep_pickup_datetime")
                )
                / 60.0,
            )
            .withColumn(
                "tip_pct",
                F.when(
                    F.col("fare_amount") > 0,
                    F.col("tip_amount") / F.col("fare_amount"),
                ),
            )
            .cache()
        )

    with timed("Demand heat-map"):
        demand_heatmap = (
            df.groupBy("PULocationID", "pickup_hour")
            .count()
            .withColumnRenamed("count", "n_trips")
        )
        demand_heatmap.count()

    with timed("Tip trends"):
        tip_trends = (
            df.groupBy("PULocationID", "payment_type")
            .agg(
                F.avg("tip_pct").alias("avg_tip_pct"),
                F.count("*").alias("n_trips"),
            )
        )
        tip_trends.count()

    with timed("Fare anomalies"):
        threshold = df.approxQuantile("fare_amount", [0.99], 0.01)[0]
        fare_anomalies = (
            df.filter(F.col("fare_amount") >= threshold)
            .select(
                "VendorID",
                "tpep_pickup_datetime",
                "PULocationID",
                "DOLocationID",
                "fare_amount",
                "tip_amount",
                "trip_distance",
            )
        )
        fare_anomalies.count()

    if not skip_mysql:
        if not mysql_url:
            raise ValueError("MySQL connection details missing")
        _write_mysql(demand_heatmap, mysql_url, "demand_heatmap", mysql_user, mysql_pw, mode)
        _write_mysql(tip_trends, mysql_url, "tip_trends", mysql_user, mysql_pw, mode)
        _write_mysql(fare_anomalies, mysql_url, "fare_anomalies", mysql_user, mysql_pw, "overwrite")

    spark.stop()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--parquet", required=True)
    cli.add_argument("--mysql-url", default="")
    cli.add_argument("--mysql-user", default="")
    cli.add_argument("--mysql-password", default="")
    cli.add_argument("--append", action="store_true")
    cli.add_argument("--no-mysql", action="store_true")
    args = cli.parse_args()

    main(
        args.parquet,
        args.mysql_url,
        args.mysql_user,
        args.mysql_password,
        overwrite_outputs=not args.append,
        skip_mysql=args.no_mysql,
    )