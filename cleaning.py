from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder.appName("Clean Yellow Taxi Data").getOrCreate()

# Load all parquet files matching the pattern
df = spark.read.parquet("./*.parquet")

# Define the list of columns to check for nulls
columns_to_check = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID",
    "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge", "Airport_fee"
]

# Drop rows with nulls in any of those columns
df_cleaned = df.dropna(subset=columns_to_check)

df_cleaned.write.mode("overwrite").parquet("path/to/cleaned/yellow_tripdata_2024_cleaned.parquet")

# Optional: show a few rows
df_cleaned.show()