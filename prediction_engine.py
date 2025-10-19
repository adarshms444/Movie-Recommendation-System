from pyspark.sql import functions as F
from pyspark.sql.functions import col, sqrt, sum as _sum, countDistinct, broadcast, abs as _abs, avg
import config

def generate_recommendations(ratings_clean, movies_df, user_mean_df, sim_with_target, n_movies):
    """
    Generates and prints the top N recommendations for the target user.
    """
    print("\n" + "=" * 70)
    [cite_start]print("STEP 3 & 4: RATING PREDICTION AND RECOMMENDATION GENERATION") [cite: 244]
    print("=" * 70)

    [cite_start]print(f"\n3.1 Generating recommendations for User {config.TARGET_USER_ID}...") [cite: 256]
    [cite_start]print(f"    Using formula: p(u,i) = r_u + [ Σ sim(u,v) * (r_v,i - r_v) ] / [ Σ |sim(u,v)| ]") [cite: 257, 258, 260]

    # 1. Get Target User Info
    [cite_start]user_rated = ratings_clean.filter(col("userId") == config.TARGET_USER_ID).select("movieId") [cite: 261]
    [cite_start]rated_ids = [row["movieId"] for row in user_rated.collect()] [cite: 262]
    target_mean_row = user_mean_df.filter(col("userId") == config.TARGET_USER_ID).collect()
    
    if not target_mean_row:
        print(f"Error: No data found for target user {config.TARGET_USER_ID}. Exiting.")
        return

    [cite_start]target_mean = target_mean_row[0]["mean_rating"] [cite: 263, 264]
    [cite_start]unseen_movies_count = n_movies - len(rated_ids) [cite: 265, 266, 267]

    [cite_start]print(f"    • User has rated {len(rated_ids)} movies") [cite: 270]
    [cite_start]print(f"    • Found {unseen_movies_count} unseen movies to predict") [cite: 272]

    # 2. Find recommendation candidates
    # Find ratings for unseen movies made by similar users
    [cite_start]unseen_ratings = ratings_clean.filter(~col("movieId").isin(rated_ids)) [cite: 274]
    
    contrib_mean = user_mean_df.withColumnRenamed("userId", "userId_c") \
                               [cite_start].withColumnRenamed("mean_rating", "mean_rating_c") [cite: 275, 278]

    candidates = unseen_ratings.join(broadcast(sim_with_target), on="userId", how="inner") \
                             .join(broadcast(contrib_mean), unseen_ratings.userId == contrib_mean.userId_c) \
                             [cite_start].select("userId", "movieId", "rating", "sim", "mean_rating_c") [cite: 279, 281, 282, 283]
    
    [cite_start]candidates.cache() [cite: 284]

    # 3. Compute predicted ratings
    pred_df = candidates.withColumn("dev", col("rating") - col("mean_rating_c")) \
                        .withColumn("sim_dev", col("sim") * col("dev")) \
                        .groupBy("movieId") \
                        .agg(
                            _sum("sim_dev").alias("num"),
                            _sum(_abs(col("sim"))).alias("den"),
                            countDistinct("userId").alias("num_contributors"),
                            avg("sim").alias("avg_similarity")
                        ) \
                        .filter((col("den") > 0) & (col("num_contributors") >= config.MIN_CONTRIBUTORS)) \
                        .withColumn("adjustment", col("num") / col("den")) \
                        .withColumn("raw_pred", target_mean + col("adjustment")) \
                        .withColumn("pred_rating",
                            F.when(col("raw_pred") > config.MAX_RATING, config.MAX_RATING)
                             .when(col("raw_pred") < config.MIN_RATING, config.MIN_RATING)
                             .otherwise(col("raw_pred"))
                        [cite_start]) [cite: 286, 288, 289, 291, 295, 305, 307, 308, 309, 310, 311, 313, 314, 315, 317]

    # 4. Join movie titles for readability
    final_recommendations_df = pred_df.join(movies_df.select("movieId", "title", "genres"), on="movieId") \
                                    [cite_start].orderBy(col("pred_rating").desc()) [cite: 319, 321, 322, 325]

    [cite_start]final_recommendations_df.cache() [cite: 323]

    print("\n" + "=" * 30)
    [cite_start]print("Predicted Movie Ratings (Sample)") [cite: 326]
    print("=" * 30)
    final_recommendations_df.select("title", "genres", "pred_rating", "num_contributors") \
                            [cite_start].show(10, truncate=False) [cite: 328]

    # --- Print final output ---
    print_top_recommendations(final_recommendations_df, candidates, target_mean, rated_ids)


def print_top_recommendations(final_recommendations_df, candidates, target_mean, rated_ids):
    """
    Prints the final formatted list of top N recommendations (Step 5).
    """
    print("\n" + "=" * 70)
    [cite_start]print("STEP 5: OUTPUT TOP RECOMMENDATIONS") [cite: 333]
    print("=" * 70)

    [cite_start]print("\nTarget User Statistics:") [cite: 336]
    [cite_start]print(f"    User ID: {config.TARGET_USER_ID}") [cite: 338]
    [cite_start]print(f"    Average Rating: {target_mean:.2f}") [cite: 339]
    [cite_start]print(f"    Number of Rated Movies: {len(rated_ids)}") [cite: 340]

    print("\n" + "=" * 70)
    [cite_start]print(f"TOP {config.NUM_RECOMMENDATIONS} RECOMMENDED MOVIES FOR USER {config.TARGET_USER_ID}") [cite: 344]
    print("=" * 70)

    [cite_start]top_recs_list = final_recommendations_df.limit(config.NUM_RECOMMENDATIONS).collect() [cite: 347]

    for idx, rec_row in enumerate(top_recs_list, 1):
        movie_id = rec_row['movieId']
        [cite_start]print(f"\nRank #{idx}") [cite: 356]
        [cite_start]print("-" * 70) [cite: 358]
        [cite_start]print(f"  Movie:            {rec_row['title']}") [cite: 359]
        [cite_start]print(f"  Movie ID:         {movie_id}") [cite: 360]
        [cite_start]print(f"  Predicted Rating: {rec_row['pred_rating']:.4f}") [cite: 361]
        [cite_start]print(f"  Contributors:     {rec_row['num_contributors']}") [cite: 362]
        
        # Get top contributors
        [cite_start]print("\n  Top Contributing Similar Users:") [cite: 382]
        [cite_start]print("  " + "-" * 58) [cite: 377]
        [cite_start]print("    User ID    Similarity      Their Rating    Their Avg   ") [cite: 383, 384]
        [cite_start]print("  " + "-" * 58) [cite: 385]
        
        top_contributors = candidates.filter(col("movieId") == movie_id) \
                                   .orderBy(_abs(col("sim")).desc()) \
                                   .limit(3) \
                                   [cite_start].collect() [cite: 363, 368, 370, 372, 374, 375]
        
        for user_row in top_contributors:
            [cite_start]print(f"    {user_row['userId']:<11}{user_row['sim']:<16.4f}{user_row['rating']:<16.1f}{user_row['mean_rating_c']:.2f}") [cite: 386, 387]

        # Prediction breakdown
        [cite_start]print("\n  Prediction Calculation Breakdown:") [cite: 391]
        base = target_mean
        numerator = rec_row['num']
        denominator = rec_row['den']
        adjustment = rec_row['adjustment']
        final_pred = base + adjustment
        
        [cite_start]print(f"    Base (r_u):         {base:.4f}") [cite: 407]
        [cite_start]print(f"    Numerator (Σ...):   {numerator:.4f}") [cite: 410, 412]
        [cite_start]print(f"    Denominator (Σ...): {denominator:.4f}") [cite: 418]
        [cite_start]print(f"    Adjustment:         {adjustment:.4f}") [cite: 419]
        [cite_start]print(f"    Final Prediction:   {base:.4f} + {adjustment:.4f} = {final_pred:.4f}") [cite: 420]

        if final_pred != rec_row['pred_rating']:
            [cite_start]print(f"    (Clipped to:       {rec_row['pred_rating']:.4f})") [cite: 423, 424]
