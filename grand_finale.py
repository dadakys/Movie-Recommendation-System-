import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    confusion_matrix
)


# Load dataset from specified path
file_path = "Desktop/ratings.csv"
ratings_df = pd.read_csv(file_path)
print("Dataset successfully loaded!")
print("Dataset shape:", ratings_df.shape)

# Display first few rows of the dataset
print("\nPreview of the dataset:")
print(ratings_df.head())

# Dataset information and summary statistics
print("\nDataset Overview:")
print(ratings_df.info())
print("\nStatistical Summary:")
print(ratings_df.describe())
# Check for missing values
if ratings_df.isnull().sum().any():
    print("\nWarning: Missing values detected. Consider handling them appropriately.")
else:
    print("\nNo missing values found.")

# Display number of unique users and movies
print("\nTotal Unique Users:", ratings_df['userId'].nunique())
print("Total Unique Movies:", ratings_df['movieId'].nunique())

# Analyze Ratings Per Movie
movie_ratings = ratings_df.groupby('movieId')['rating'].count()
print("\nStatistics for ratings per movie:")
print(movie_ratings.describe())

# Visualizing distribution of ratings per movie
plt.figure(figsize=(10, 6))
movie_ratings.hist(bins=50)
plt.title("Distribution of Ratings Per Movie")
plt.xlabel("Number of Ratings")
plt.ylabel("Frequency")
plt.show()

# Analyze Ratings Per User
user_ratings = ratings_df.groupby('userId')['rating'].count()
print("\nStatistics for ratings per user:")
print(user_ratings.describe())

# Visualizing distribution of ratings per user
plt.figure(figsize=(10, 6))
user_ratings.hist(bins=50)
plt.title("Distribution of Ratings Per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Frequency")
plt.show()

#from here

# Test different M_users values
M_users_values = range(20, 401, 20)  # Test thresholds from 20 to 200 in steps of 20
remaining_users = []
remaining_ratings = []

for M_users in M_users_values:
    filtered_users = ratings_df.groupby('userId').filter(lambda x: len(x) >= M_users)
    remaining_users.append(filtered_users['userId'].nunique())
    remaining_ratings.append(len(filtered_users))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(M_users_values, remaining_users, marker='o', label='Remaining Users')
plt.plot(M_users_values, remaining_ratings, marker='s', label='Remaining Ratings')
plt.title("Effect of M_users on Dataset Size")
plt.xlabel("M_users (Minimum Ratings per User)")
plt.ylabel("Count")
plt.legend()
plt.grid()
plt.show()

# Test different M values
M_values = range(1, 100, 10)  # Test thresholds from 1 to 20 in steps of 2
remaining_movies = []
remaining_ratings_movies = []

for M in M_values:
    # Filter movies based on M
    filtered_movies = ratings_df.groupby('movieId').filter(lambda x: len(x) >= M)
    remaining_movies.append(filtered_movies['movieId'].nunique())  # Count remaining movies
    remaining_ratings_movies.append(len(filtered_movies))  # Count remaining ratings

# Plot the results for M
plt.figure(figsize=(10, 6))
plt.plot(M_values, remaining_movies, marker='o', label='Remaining Movies')
plt.plot(M_values, remaining_ratings_movies, marker='s', label='Remaining Ratings')
plt.title("Effect of M on Dataset Size")
plt.xlabel("M (Minimum Ratings per Movie)")
plt.ylabel("Count")
plt.legend()
plt.grid()
plt.show()

#to here

# Creating User-Item Matrix
print("\nGenerating User-Item Matrix...")
user_item_matrix = ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Compute Cosine Similarity Matrix
print("Calculating Cosine Similarity...")
cosine_sim_matrix = cosine_similarity(user_item_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Compute Pearson Correlation Matrix
print("Computing Pearson Correlation...")
pearson_corr_matrix = user_item_matrix.T.corr()
pearson_corr_df = pd.DataFrame(pearson_corr_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

print("Similarity matrices computed successfully!")

def predict_rating(user_id, movie_id, similarity_matrix, user_item_matrix, N, weighting='basic'):
    """
    Predict rating using item-item collaborative filtering and top-N neighbors.
    Implements different weighting strategies, including an extra penalty for frequently rated movies.
    """
    # Ensure the movie exists in the similarity matrix
    if movie_id not in similarity_matrix.index:
        return np.nan  # Return NaN if movie is not in the training data

    # Ensure the user exists in the user-item matrix
    if user_id not in user_item_matrix.columns:
        return np.nan  # Return NaN if user is not in the training data

    # Get all movies rated by the user
    user_ratings = user_item_matrix.loc[:, user_id]
    rated_movies = user_ratings[user_ratings > 0].index

    # Ensure rated movies exist in both the similarity matrix and the user-item matrix
    rated_movies = [m for m in rated_movies if m in similarity_matrix.index]

    if not rated_movies:
        return np.nan  # Return NaN if no rated movies exist in both matrices

    # Select the top-N similar movies that the user has rated
    similar_movies = similarity_matrix.loc[movie_id, rated_movies].sort_values(ascending=False).head(N)

    if similar_movies.empty:
        return np.nan  # Return NaN if there are no similar movies rated by the user

    # Compute movie popularity (how many users rated each movie)
    popularity = user_item_matrix.loc[similar_movies.index].count(axis=1)

    # Apply weighting strategy
    if weighting == 'favor_popular':
        weights = similar_movies * popularity  # More weight to popular movies
    elif weighting == 'penalize_popular':
        weights = similar_movies / (1 + popularity)  # Reduce influence of popular movies
    elif weighting == 'extra_penalty':  
        weights = similar_movies / (1 + 2 * popularity)  # **Extra penalty for very frequently rated movies**
    else:
        weights = similar_movies  # Default basic weighting (no change)

    # Compute weighted average prediction
    numerator = np.dot(weights, user_ratings[similar_movies.index])
    denominator = np.sum(abs(weights))

    return numerator / denominator if denominator != 0 else np.nan

# Experiment Setup
N_values = [5, 15, 30, 75, 200]
num_iterations = 5
results = []

for N in N_values:
    print(f"\nRunning experiments for N={N}...")
    for similarity_name, similarity_matrix in {'Cosine': cosine_sim_df, 'Pearson': pearson_corr_df}.items():
        for weight_type in ['basic', 'favor_popular', 'penalize_popular']:
            mae_scores, precision_scores, recall_scores, confusion_matrices = [], [], [], []

            for iteration in range(num_iterations):
                print(f"  Iteration {iteration + 1}/{num_iterations}...")
                train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
                train_matrix = train_data.pivot(index='movieId', columns='userId', values='rating').fillna(0)

                # Evaluate directly in the loop
                y_actual, y_predicted = [], []

                user_mean_ratings = train_matrix.mean(axis=0)
                global_mean = train_matrix.mean().mean()

                for _, row in test_data.iterrows():
                    user_id, movie_id, actual_rating = row['userId'], row['movieId'], row['rating']
                    predicted_rating = predict_rating(user_id, movie_id, similarity_matrix, train_matrix, N, weight_type)

                    if not np.isnan(predicted_rating):
                        y_actual.append(actual_rating)
                        y_predicted.append(predicted_rating)

                mae = mean_absolute_error(y_actual, y_predicted) if y_actual else float('nan')

                y_actual_binary = [1 if rating >= user_mean_ratings.get(user_id, global_mean) else 0 for user_id, rating in zip(test_data['userId'], y_actual)]
                y_predicted_binary = [1 if rating >= user_mean_ratings.get(user_id, global_mean) else 0 for user_id, rating in zip(test_data['userId'], y_predicted)]

                precision = precision_score(y_actual_binary, y_predicted_binary, average='macro', zero_division=0) if y_actual_binary else float('nan')
                recall = recall_score(y_actual_binary, y_predicted_binary, average='macro', zero_division=0) if y_actual_binary else float('nan')
                cm = confusion_matrix(y_actual_binary, y_predicted_binary, labels=[0, 1]) if y_actual_binary else np.array([[0, 0], [0, 0]])

                mae_scores.append(mae)
                precision_scores.append(precision)
                recall_scores.append(recall)
                confusion_matrices.append(cm)

            avg_mae = np.nanmean(mae_scores)
            avg_precision = np.nanmean(precision_scores)
            avg_recall = np.nanmean(recall_scores)
            avg_cm = np.nansum(confusion_matrices, axis=0)

            print(f"\nResults for N={N}, Similarity={similarity_name}, Weight={weight_type}:")
            print(f"  Avg MAE: {avg_mae:.4f}")
            print(f"  Avg Precision: {avg_precision:.4f}")
            print(f"  Avg Recall: {avg_recall:.4f}")
            print(f"  Avg Confusion Matrix:\n{avg_cm}")

            results.append({
                'N': N,
                'Similarity': similarity_name,
                'Weight': weight_type,
                'Avg MAE': avg_mae,
                'Avg Precision': avg_precision,
                'Avg Recall': avg_recall,
                'Avg Confusion Matrix': avg_cm.tolist()
            })

results_df = pd.DataFrame(results)
results_df.to_csv('D:/experiment_results.csv', index=False)
print("\nExperiment results saved to 'D:/experiment_results.csv'")

# Select best N based on maximum Precision
best_N = results_df.loc[results_df['Avg Precision'].idxmax(), 'N']


# Experiment 2: Different training set percentages
T_values = [50, 70, 90]
experiment_2_results = []

for T in T_values:
    print(f"\nRunning experiments for T={T}%...")
    for similarity_name, similarity_matrix in {'Cosine': cosine_sim_df, 'Pearson': pearson_corr_df}.items():
        for weight_type in ['basic', 'favor_popular', 'penalize_popular']:
            mae_scores, precision_scores, recall_scores, confusion_matrices = [], [], [], []

            for iteration in range(num_iterations):
                train_data, test_data = train_test_split(ratings_df, test_size=(1 - T / 100), random_state=42)
                train_matrix = train_data.pivot(index='movieId', columns='userId', values='rating').fillna(0)

                y_actual, y_predicted = [], []
                user_mean_ratings = train_matrix.mean(axis=0)
                global_mean = train_matrix.mean().mean()

                for _, row in test_data.iterrows():
                    user_id, movie_id, actual_rating = row['userId'], row['movieId'], row['rating']
                    predicted_rating = predict_rating(user_id, movie_id, similarity_matrix, train_matrix, best_N, weight_type)

                    if not np.isnan(predicted_rating):
                        y_actual.append(actual_rating)
                        y_predicted.append(predicted_rating)

                mae = mean_absolute_error(y_actual, y_predicted) if y_actual else float('nan')

                # Convert ratings to binary labels
                y_actual_binary = [1 if rating >= user_mean_ratings.get(user_id, global_mean) else 0 for user_id, rating in zip(test_data['userId'], y_actual)]
                y_predicted_binary = [1 if rating >= user_mean_ratings.get(user_id, global_mean) else 0 for user_id, rating in zip(test_data['userId'], y_predicted)]

                precision = precision_score(y_actual_binary, y_predicted_binary, average='macro', zero_division=0) if y_actual_binary else float('nan')
                recall = recall_score(y_actual_binary, y_predicted_binary, average='macro', zero_division=0) if y_actual_binary else float('nan')
                cm = confusion_matrix(y_actual_binary, y_predicted_binary, labels=[0, 1]) if y_actual_binary else np.array([[0, 0], [0, 0]])

                mae_scores.append(mae)
                precision_scores.append(precision)
                recall_scores.append(recall)
                confusion_matrices.append(cm)

            experiment_2_results.append({
                'T': T,
                'Similarity': similarity_name,
                'Weight': weight_type,
                'Avg MAE': np.nanmean(mae_scores),
                'Avg Precision': np.nanmean(precision_scores),
                'Avg Recall': np.nanmean(recall_scores),
                'Avg Confusion Matrix': np.nansum(confusion_matrices, axis=0).tolist()
            })

experiment_2_df = pd.DataFrame(experiment_2_results)
experiment_2_df.to_csv('D:/experiment_2_results.csv', index=False)
print("\nExperiment 2 results saved to 'D:/experiment_2_results.csv'")


# Experiment 3: Filtering based on movie/user activity with multiple M, M' values
M_values = [10, 3]
M_prime_values = [168, 70]
experiment_3_results = []

for M in M_values:
    for M_prime in M_prime_values:
        print(f"\nRunning experiments for M={M}, M'={M_prime}...")

        # Create a copy of the dataset to avoid modifying the original
        filtered_data = ratings_df.copy()

        # Apply filtering: Keep movies with at least M ratings
        movie_counts = filtered_data['movieId'].value_counts()
        filtered_data = filtered_data[filtered_data['movieId'].isin(movie_counts[movie_counts >= M].index)]

        # Apply filtering: Keep users with at least M' ratings
        user_counts = filtered_data['userId'].value_counts()
        filtered_data = filtered_data[filtered_data['userId'].isin(user_counts[user_counts >= M_prime].index)]

        # Check if filtered data is empty before splitting
        if filtered_data.empty:
            print(f"Skipping M={M}, M'={M_prime} because filtered data is empty.")
            continue  # Skip this iteration

        # Split into train and test sets
        train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

        # Create User-Item matrix
        train_matrix = train_data.pivot(index='movieId', columns='userId', values='rating').fillna(0)

        # Predict ratings for test set
        y_actual, y_predicted = [], []
        user_mean_ratings = train_matrix.mean(axis=0)
        global_mean = train_matrix.mean().mean()

        for _, row in test_data.iterrows():
            user_id, movie_id, actual_rating = row['userId'], row['movieId'], row['rating']
            predicted_rating = predict_rating(user_id, movie_id, cosine_sim_df, train_matrix, best_N, 'basic')

            if not np.isnan(predicted_rating):
                y_actual.append(actual_rating)
                y_predicted.append(predicted_rating)

        # Compute Metrics only if valid predictions exist
        if y_actual and y_predicted:
            mae = mean_absolute_error(y_actual, y_predicted)
            
            # Convert ratings to binary labels
            y_actual_binary = [1 if rating >= user_mean_ratings.get(user_id, global_mean) else 0 for user_id, rating in zip(test_data['userId'], y_actual)]
            y_predicted_binary = [1 if rating >= user_mean_ratings.get(user_id, global_mean) else 0 for user_id, rating in zip(test_data['userId'], y_predicted)]

            precision = precision_score(y_actual_binary, y_predicted_binary, average='macro', zero_division=0)
            recall = recall_score(y_actual_binary, y_predicted_binary, average='macro', zero_division=0)
            cm = confusion_matrix(y_actual_binary, y_predicted_binary, labels=[0, 1])
        else:
            mae, precision, recall, cm = float('nan'), float('nan'), float('nan'), np.array([[0, 0], [0, 0]])

        print(f"\nResults for M={M}, M'={M_prime}:")
        print(f"  Avg MAE: {mae:.4f}")
        print(f"  Avg Precision: {precision:.4f}")
        print(f"  Avg Recall: {recall:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        # Store results
        experiment_3_results.append({
            'M': M,
            'M_prime': M_prime,
            'Avg MAE': mae,
            'Avg Precision': precision,
            'Avg Recall': recall,
            'Confusion Matrix': cm.tolist()
        })

# Save Experiment 3 results to CSV
experiment_3_df = pd.DataFrame(experiment_3_results)
experiment_3_df.to_csv('D:/experiment_3_results.csv', index=False)
print("\nExperiment 3 results saved to 'D:/experiment_3_results.csv'")
