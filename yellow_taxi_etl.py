#!/usr/bin/env python3
"""
NYC Yellow-Taxi Analytics Pipeline 
=====================================================


1. **Identify high-demand pickup zones and peak hours** (heat-map table).
2. **Analyze tip trends** across payment methods and zones.
3. **Detect fare anomalies** based on simple pricing outliers.

Outputs land in MySQL tables:
  • `demand_heatmap` (zone × hour counts)
  • `tip_trends`     (zone × payment avg tip%)
  • `fare_anomalies` (trips with very high fare amounts)

Typical usage:

    spark-submit \
      --packages mysql:mysql-connector-java:8.4.0 \
      yellow_taxi_etl.py \
        --parquet /data/yellow_tripdata_2024_cleaned.parquet \
        --mysql-url jdbc:mysql://mysql.host:3306/nyctaxi \
        --mysql-user nyc_user \
        --mysql-password secret \
        --append   # (optional) append instead of overwrite for demand & tips
"""

import argparse
import datetime as _dt

from pyspark.sql import SparkSession, functions as F

###############################################################################
# JDBC helper
###############################################################################

def _write_mysql(df, url: str, table: str, user: str, pw: str, mode: str):
    (
        df.write
          .format("jdbc")
          .option("url", url)
          .option("dbtable", table)
          .option("user", user)
          .option("password", pw)
          .option("driver", "com.mysql.cj.jdbc.Driver")
          .mode(mode)
          .save()
    )

###############################################################################
# Main logic
###############################################################################

def main(*, parquet_path: str, mysql_url: str, mysql_user: str, mysql_pw: str,
         overwrite_outputs: bool):
    mode = "overwrite" if overwrite_outputs else "append"
    spark = (
        SparkSession.builder
        .appName("YellowTaxiAnalytics")
        .getOrCreate()
    )

    # ------------------------------------------------------------------
    # 1 ▸ Load cleaned parquet and feature engineering
    # ------------------------------------------------------------------
    df = (
        spark.read.parquet(parquet_path)
             .withColumn("pickup_hour",  F.hour("tpep_pickup_datetime"))
             .withColumn("pickup_dow",   F.dayofweek("tpep_pickup_datetime"))
             .withColumn(
                 "trip_minutes",
                 (F.unix_timestamp("tpep_dropoff_datetime") -
                  F.unix_timestamp("tpep_pickup_datetime")) / 60.0,
             )
             .withColumn(
                 "tip_pct",
                 F.when(F.col("fare_amount") > 0, F.col("tip_amount") / F.col("fare_amount")),
             )
             .withColumn(
                 "avg_speed_mph",
                 F.when(F.col("trip_minutes") > 0, F.col("trip_distance") / (F.col("trip_minutes") / 60.0))
             )
             .withColumn(
                 "revenue_per_mile",
                 F.when(F.col("trip_distance") > 0, F.col("fare_amount") / F.col("trip_distance"))
             )
             .withColumn(
                 "distance_category",
                 F.when(F.col("trip_distance") < 1, "Very Short")
                  .when(F.col("trip_distance") < 3, "Short")
                  .when(F.col("trip_distance") < 5, "Medium")
                  .when(F.col("trip_distance") < 10, "Long")
                  .otherwise("Very Long")
             )
             .cache()
    )

    # ------------------------------------------------------------------
    # 2 ▸ High-demand zones × hours
    # ------------------------------------------------------------------
    demand_heatmap = (
        df.groupBy("PULocationID", "pickup_hour")
          .count()
          .withColumnRenamed("count", "n_trips")
    )

    # ------------------------------------------------------------------
    # 3 ▸ Tip trends by payment method & zone
    # ------------------------------------------------------------------
    tip_trends = (
        df.groupBy("PULocationID", "payment_type")
          .agg(
              F.avg("tip_pct").alias("avg_tip_pct"),
              F.count("*").alias("n_trips"),
          )
    )

    # ------------------------------------------------------------------
    # 4 ▸ Fare anomalies – flag unusually expensive trips (top 1%)
    # ------------------------------------------------------------------
    threshold = df.approxQuantile("fare_amount", [0.99], 0.01)[0]

    fare_anomalies = (
        df.filter(F.col("fare_amount") >= threshold)
          .select(
              "VendorID", "tpep_pickup_datetime", "PULocationID", "DOLocationID",
              "fare_amount", "tip_amount", "trip_distance"
          )
    )

    # ------------------------------------------------------------------
    # 6 ▸ Trip Performance Analytics
    # ------------------------------------------------------------------
    trip_performance = (
        df.groupBy("PULocationID", "pickup_hour", "pickup_dow")
          .agg(
              F.avg("trip_minutes").alias("avg_trip_duration"),
              F.avg("avg_speed_mph").alias("avg_speed"),
              F.avg("revenue_per_mile").alias("avg_revenue_per_mile"),
              F.avg("fare_amount").alias("avg_fare"),
              F.sum("fare_amount").alias("total_revenue"),
              F.count("*").alias("n_trips"),
              F.avg("trip_distance").alias("avg_trip_distance"),
              F.avg("tip_amount").alias("avg_tip"),
              F.avg("tip_pct").alias("avg_tip_percentage")
          )
          .withColumn("is_weekend", F.when(F.col("pickup_dow").isin(1, 7), True).otherwise(False))
    )

    # ------------------------------------------------------------------
    # 8 ▸ Popular Routes Analysis
    # ------------------------------------------------------------------
    popular_routes = (
        df.groupBy("PULocationID", "DOLocationID", "pickup_hour")
          .agg(
              F.count("*").alias("n_trips"),
              F.avg("trip_minutes").alias("avg_duration"),
              F.avg("fare_amount").alias("avg_fare"),
              F.avg("trip_distance").alias("avg_distance"),
              F.avg("tip_amount").alias("avg_tip")
          )
          .filter(F.col("n_trips") >= 10)  # Only include routes with at least 10 trips
          .orderBy(F.col("n_trips").desc())
    )

    # ------------------------------------------------------------------
    # 9 ▸ Payment Method Analysis
    # ------------------------------------------------------------------
    payment_analysis = (
        df.groupBy("PULocationID", "pickup_hour", "payment_type")
          .agg(
              F.count("*").alias("n_trips"),
              F.avg("fare_amount").alias("avg_fare"),
              F.avg("tip_amount").alias("avg_tip"),
              F.avg("tip_pct").alias("avg_tip_percentage"),
              F.sum("fare_amount").alias("total_revenue")
          )
          .withColumn(
              "payment_method",
              F.when(F.col("payment_type") == 1, "Credit Card")
               .when(F.col("payment_type") == 2, "Cash")
               .when(F.col("payment_type") == 3, "No Charge")
               .when(F.col("payment_type") == 4, "Dispute")
               .otherwise("Unknown")
          )
    )

    # ------------------------------------------------------------------
    # 10 ▸ Distance Distribution Analysis
    # ------------------------------------------------------------------
    distance_distribution = (
        df.groupBy("PULocationID", "pickup_hour", "distance_category")
          .agg(
              F.count("*").alias("n_trips"),
              F.avg("fare_amount").alias("avg_fare"),
              F.avg("trip_minutes").alias("avg_duration"),
              F.avg("tip_amount").alias("avg_tip"),
              F.sum("fare_amount").alias("total_revenue")
          )
    )

    # ------------------------------------------------------------------
    # 11 ▸ Persist to MySQL
    # ------------------------------------------------------------------
    _write_mysql(demand_heatmap,  mysql_url, "demand_heatmap",  mysql_user, mysql_pw, mode)
    _write_mysql(tip_trends,      mysql_url, "tip_trends",      mysql_user, mysql_pw, mode)
    _write_mysql(fare_anomalies,  mysql_url, "fare_anomalies",  mysql_user, mysql_pw, "overwrite")
    _write_mysql(trip_performance, mysql_url, "trip_performance", mysql_user, mysql_pw, mode)
    _write_mysql(popular_routes, mysql_url, "popular_routes", mysql_user, mysql_pw, mode)
    _write_mysql(payment_analysis, mysql_url, "payment_analysis", mysql_user, mysql_pw, mode)
    _write_mysql(distance_distribution, mysql_url, "distance_distribution", mysql_user, mysql_pw, mode)

    spark.stop()

###############################################################################
# CLI entry-point
###############################################################################

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Yellow-Taxi Spark analytics → MySQL")
    cli.add_argument("--parquet",        required=True, help="Path to cleaned parquet")
    cli.add_argument("--mysql-url",      required=True)
    cli.add_argument("--mysql-user",     required=True)
    cli.add_argument("--mysql-password", required=True)
    cli.add_argument("--append", action="store_true",
                     help="Append instead of overwrite for demand & tip tables")

    args = cli.parse_args()

    main(
        parquet_path=args.parquet,
        mysql_url=args.mysql_url,
        mysql_user=args.mysql_user,
        mysql_pw=args.mysql_password,
        overwrite_outputs=not args.append,
    )
