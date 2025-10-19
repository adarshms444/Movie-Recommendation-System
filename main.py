import data_loader
import similarity_calculator
import prediction_engine

def main():
    print("=" * 70)
    [cite_start]print("USER-BASED COLLABORATIVE FILTERING RECOMMENDATION SYSTEM") [cite: 51]
    print("=" * 70)
    
    spark = None
    try:
        spark = data_loader.get_spark_session()
        data_loader.clone_dataset()

        # Step 1: Data Preparation
        ratings_clean, movies_df, user_mean_df, n_movies = data_loader.prepare_data(spark)

        # Step 2: Similarity Computation
        sim_with_target = similarity_calculator.compute_similarity(ratings_clean, user_mean_df)

        # Step 3, 4 & 5: Rating Prediction, Generation, and Output
        prediction_engine.generate_recommendations(
            ratings_clean, 
            movies_df, 
            user_mean_df, 
            sim_with_target, 
            n_movies
        )

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if spark:
            [cite_start]print("\n" + "=" * 70) [cite: 426]
            [cite_start]print("RECOMMENDATION SYSTEM COMPLETED SUCCESSFULLY") [cite: 428]
            [cite_start]print("=" * 70) [cite: 429]
            print("Stopping Spark Session...")
            [cite_start]spark.stop() [cite: 431]
            print("Spark Session stopped.")

if __name__ == "__main__":
    main()
