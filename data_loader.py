import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, avg, count as _count
import config

def get_spark_session():
    """Initializes and returns a SparkSession."""
    [cite_start]os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64" [cite: 37]
    
    spark = SparkSession.builder \
        .appName("User-Based Collaborative Filtering") \
        [cite_start].getOrCreate() [cite: 45, 47, 49]
    [cite_start]print(f"Spark Session Initialized (Version {spark.version})") [cite: 53]
    return spark

def clone_dataset():
    """Clones the dataset from the GitHub repository."""
    print("\nCloning dataset...")
    [cite_start]os.system(f"git clone -q {config.DATASET_REPO}") [cite: 54]

def prepare_data(spark):
    """
    Loads, cleans, and preprocesses the ratings and movies data.
    Returns cleaned ratings, movies df, user mean ratings, and n_movies.
    """
    print("\n" + "=" * 70)
    [cite_start]print("STEP 1: DATA PREPARATION") [cite: 58]
    print("=" * 70)

    # Load datasets
    [cite_start]ratings_df = spark.read.csv(config.RATINGS_CSV_PATH, header=True, inferSchema=True) [cite: 63, 64]
    [cite_start]movies_df = spark.read.csv(config.MOVIES_CSV_PATH, header=True, inferSchema=True) [cite: 65, 66]

    [cite_start]print("\n1.1 Dataset Schema (Ratings):") [cite: 67]
    [cite_start]ratings_df.printSchema() [cite: 69-72]

    [cite_start]print("\n1.2 Sample Data (first 10 rows):") [cite: 73]
    [cite_start]ratings_df.show(10) [cite: 74]

    # Data Cleaning
    [cite_start]print("\n1.3 Basic Statistics (Cleaned Data):") [cite: 97]
    ratings_clean = ratings_df.na.drop(subset=["userId", "movieId", "rating"]) \
                                .dropDuplicates(["userId", "movieId"]) \
                                [cite_start].filter((col("rating") >= config.MIN_RATING) & (col("rating") <= config.MAX_RATING)) [cite: 80, 81, 82, 252, 253]
    
    [cite_start]ratings_clean.cache() [cite: 83]

    [cite_start]n_users = ratings_clean.select("userId").distinct().count() [cite: 91]
    [cite_start]n_movies = ratings_clean.select("movieId").distinct().count() [cite: 92]
    [cite_start]n_ratings = ratings_clean.count() [cite: 90]
    
    # Avoid division by zero if dataset is empty
    if (n_users * n_movies) == 0:
        sparsity = 100.0
    else:
        [cite_start]sparsity = (1.0 - (n_ratings / (n_users * n_movies))) * 100 [cite: 94, 95]

    [cite_start]print(f"    Total Ratings: {n_ratings:,}") [cite: 99]
    [cite_start]print(f"    Total Users:   {n_users:,}") [cite: 101]
    [cite_start]print(f"    Total Movies:  {n_movies:,}") [cite: 103]
    [cite_start]print(f"    Sparsity:      {sparsity:.2f}%") [cite: 105]

    [cite_start]print("\n1.4 Rating Distribution:") [cite: 107]
    [cite_start]ratings_clean.groupBy("rating").count().orderBy("rating").show() [cite: 108]

    [cite_start]print("\n1.5 User Activity Statistics:") [cite: 111]
    user_activity_df = ratings_clean.groupBy("userId").agg(
        _count("rating").alias("num_ratings"),
        avg("rating").alias("avg_rating")
    [cite_start]) [cite: 112, 114, 116]
    [cite_start]user_activity_df.cache() [cite: 118]
    
    user_activity_df.select(col("userId").cast("double"), "num_ratings", "avg_rating") \
                    [cite_start].describe().show() [cite: 120, 122]

    [cite_start]print("\n1.6 Pre-calculating user mean ratings...") [cite: 123]
    [cite_start]user_mean_df = user_activity_df.select("userId", col("avg_rating").alias("mean_rating")) [cite: 125]
    [cite_start]user_mean_df.cache() [cite: 125]
    [cite_start]print(f"    ✓ Average ratings computed for {user_mean_df.count()} users") [cite: 126]
    
    [cite_start]movies_count = movies_df.count() [cite: 128]
    [cite_start]print(f"\n    ✓ Loaded metadata for {movies_count:,} movies") [cite: 130]
    
    return ratings_clean, movies_df, user_mean_df, n_movies
