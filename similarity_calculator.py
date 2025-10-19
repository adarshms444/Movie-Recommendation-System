from pyspark.sql import functions as F
from pyspark.sql.functions import col, sqrt, sum as _sum, countDistinct, broadcast, abs as _abs
import config

def compute_similarity(ratings_clean, user_mean_df):
    """
    Computes the pairwise Pearson correlation similarity matrix.
    Returns a DataFrame of similarities for the target user.
    """
    print("\n" + "=" * 70)
    [cite_start]print("STEP 2: SIMILARITY COMPUTATION") [cite: 134]
    print("=" * 70)

    [cite_start]r1 = ratings_clean.alias("r1") [cite: 142]
    [cite_start]r2 = ratings_clean.alias("r2") [cite: 142]

    # Find user pairs who rated the same movie
    pairs = r1.join(r2, (col("r1.movieId") == col("r2.movieId")) & (col("r1.userId") < col("r2.userId"))) \
            .select(
                col("r1.userId").alias("userA"),
                col("r2.userId").alias("userB"),
                col("r1.movieId").alias("movieId"),
                col("r1.rating").alias("ratingA"),
                col("r2.rating").alias("ratingB")
            [cite_start]) [cite: 144-146, 148-152, 158, 160]

    pairs_with_mean = pairs.join(broadcast(user_mean_df.withColumnRenamed("userId","userA").withColumnRenamed("mean_rating","meanA")), "userA") \
                           [cite_start].join(broadcast(user_mean_df.withColumnRenamed("userId","userB").withColumnRenamed("mean_rating","meanB")), "userB") [cite: 162, 165-168]

    # Calculate Pearson components
    centered = pairs_with_mean.withColumn("devA", col("ratingA") - col("meanA")) \
                              .withColumn("devB", col("ratingB") - col("meanB")) \
                              .withColumn("devAB", col("devA") * col("devB")) \
                              .withColumn("devA2", col("devA") * col("devA")) \
                              [cite_start].withColumn("devB2", col("devB") * col("devB")) [cite: 171]

    [cite_start]epsilon = 1e-6 [cite: 177]
    similarity_calc_df = centered.groupBy("userA","userB") \
        .agg(
            _sum("devAB").alias("num"),
            _sum("devA2").alias("denA"),
            _sum("devB2").alias("denB"),
            countDistinct("movieId").alias("n_common")
        ) \
        [cite_start].withColumn("similarity", col("num") / ( sqrt(col("denA")) * sqrt(col("denB")) + epsilon ) ) [cite: 185, 189-194, 197, 198]

    [cite_start]similarity_calc_df = similarity_calc_df.na.drop(subset=["similarity"]) [cite: 200]

    # Create full symmetric matrix
    [cite_start]sim_uv = similarity_calc_df.select("userA","userB","similarity","n_common") [cite: 203]
    [cite_start]sim_vu = sim_uv.select(col("userB").alias("userA"), col("userA").alias("userB"), col("similarity"), col("n_common")) [cite: 204, 206, 207]
    [cite_start]sim_all = sim_uv.union(sim_vu).cache() [cite: 208]

    print("\n" + "=" * 30)
    [cite_start]print("User-User Similarity Matrix (Sample)") [cite: 211]
    print("=" * 30)
    [cite_start]sim_all.filter(col("n_common") >= 2).orderBy(col("similarity").desc()).show(20) [cite: 213]

    # --- Target User Similarity ---
    [cite_start]print(f"\n2.1 Computing PEARSON similarity for user {config.TARGET_USER_ID}...") [cite: 214]
    [cite_start]print(f"    Minimum common items threshold: {config.MIN_COMMON_USERS}") [cite: 216]

    sim_with_target = sim_all.filter((col("userA") == config.TARGET_USER_ID) & (col("n_common") >= config.MIN_COMMON_USERS)) \
                             [cite_start].select(col("userB").alias("userId"), col("similarity").alias("sim"), col("n_common")) [cite: 217, 220, 218, 222]

    [cite_start]sim_with_target.cache() [cite: 219]
    [cite_start]sim_count = sim_with_target.count() [cite: 225]
    [cite_start]print(f"    ✓ Computed similarities with {sim_count} users") [cite: 230]

    [cite_start]print("\n2.2 Top 10 Most Similar Users:") [cite: 231]
    [cite_start]print("    User ID    Similarity Score       Common Items   ") [cite: 232, 236]
    [cite_start]print("    " + "-" * 45) [cite: 233, 235]

    [cite_start]top_10_similar = sim_with_target.orderBy(col("sim").desc()).limit(10).collect() [cite: 237]
    for row in top_10_similar:
        [cite_start]print(f"    {row['userId']:<11}{row['sim']:<21.4f}{row['n_common']:<14}") [cite: 239]
        
    return sim_with_target
