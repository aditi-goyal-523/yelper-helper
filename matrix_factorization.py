import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_squared_error
import random
import math

TRAIN_FILE = ''
TEST_FILE = ''
num_iterations = 869
# Set the number of latent features for SVD
latent_features = 20
# Set a threshold for similarity score
similarity_threshold = 0.90

def get_unique_user(data):
    # Create a dictionary to store the list of objects for each user_id
    user_id_objects = {}
    # Iterate through the JSON objects and group them by user_id
    for obj in data:
        user_id = obj.get('user_id')
        if user_id not in user_id_objects:
            user_id_objects[user_id] = []
        user_id_objects[user_id].append(obj)
    return user_id_objects

def find_user_by_id(user_id, user_id_objects):
    if user_id in user_id_objects:
        return user_id_objects[user_id]
    else:
        return None


print("loading the training set")
# Load and process the training data
with open(TRAIN_FILE, 'r') as train_file:
    train_data = json.load(train_file)

# Get all unique user_ids and business_ids
user_ids = list(set(obj['user_id'] for obj in train_data))
business_ids = list(set(obj['business_id'] for obj in train_data))

# Sort the user_ids and business_ids
user_ids.sort()
business_ids.sort()

# Create an empty DataFrame with user_ids as rows and business_ids as columns
og_dataframe = pd.DataFrame(index=user_ids, columns=business_ids)

# Fill the DataFrame with star ratings from the train_data
for obj in train_data:
    user_id = obj['user_id']
    business_id = obj['business_id']
    stars = obj['stars']

    og_dataframe.loc[user_id, business_id] = stars

# Fill missing values with 0
og_dataframe.fillna(0, inplace=True)

df = pd.DataFrame(train_data)
reader = Reader(rating_scale=(1, 5))


# Load the train data into the Surprise Dataset
data = Dataset.load_from_df(df[['user_id', 'business_id', 'stars']], reader)
trainset = data.build_full_trainset()

print("Loading the validation set")
# Load and process the validation data
with open(TEST_FILE, 'r') as validation_file:
    validation_data = json.load(validation_file)

validation_user_ids_dict = get_unique_user(validation_data)


# Define the SVD model with SGD
print("Training started")
model = SVD(n_factors=latent_features, n_epochs=50, lr_all=0.01,  biased=False, reg_all=0)
# Train the model on the trainset
model.fit(trainset)
joblib.dump(model, 'svd_SB_lf20_ep50_lr01_bF_reg_0.joblib')
print("Training Complete")

# model = joblib.load('svd_SB_lf20_ep10_lr01_bF_reg_0.joblib')


# Retrieve the latent factors
user_latent_factors = model.pu
item_latent_factors = model.qi

# Calculate user similarity using cosine similarity
user_similarities = cosine_similarity(user_latent_factors)

count_list_train = []
count_list_test = []
actual_train_stars = []
predicted_train_stars = []
actual_test_stars = []
predicted_test_stars = []
counter = 1
chosen_indices = []

for _ in range(num_iterations):
    if counter >= num_iterations:
        break

    print(counter)
    user_index = random.randint(0, len(og_dataframe) - 1)

    # Check if user_index has already been chosen
    while user_index in chosen_indices:
        user_index = random.randint(0, len(og_dataframe) - 1)

    chosen_indices.append(user_index)
    similar_users = [j for j, score in enumerate(user_similarities[user_index]) if j != user_index and score > similarity_threshold]

    # Get the user ID of the first user in the dataframe
    first_user_id = user_ids[user_index]
    predicted_count_train = 0
    all_recommendations = []
    print("User: ", first_user_id, " has similar_users count: ", len(similar_users))
    for i in similar_users:
        user_id = user_ids[i]
        # Get the ratings for the current user
        user_ratings = og_dataframe.loc[user_id]
        # Filter out businesses with non-zero ratings
        rated_businesses = user_ratings[user_ratings != 0]
        # Get the business recommendations for a particular user
        # Recommendations will have all the businesses that the OG user has not been to yet
        recommendations = []
        for business_id, rating in rated_businesses.items():
            if rating >= 4:
                if og_dataframe.loc[first_user_id, business_id] == 0:
                    recommendations.append((business_id, rating))
                else:
                    actual_train_stars.append(og_dataframe.loc[first_user_id, business_id])
                    predicted_train_stars.append(rating)
                    predicted_count_train += 1
        for recommendation in recommendations:
                all_recommendations.append(recommendation)
    count_list_train.append(predicted_count_train)

    og_user = find_user_by_id(first_user_id, validation_user_ids_dict)

    prediction_count = 0
    if og_user is not None:
        for obj in og_user:
            for i in range(len(all_recommendations)):
                recommendation = all_recommendations[i]
                business_id = recommendation[0]  # Access business_id at index 0
                rating = recommendation[1]  # Access rating at index 1
                if obj['business_id'] == business_id:
                    actual_test_stars.append(obj['stars'])
                    predicted_test_stars.append(rating)
                    prediction_count += 1
    else:
        print("user_id not found in test data: ", first_user_id)
    counter += 1
    count_list_test.append(prediction_count)
    print("TEST: Out of ", len(all_recommendations), "recommendations. User: ", first_user_id, " has matches count: ", prediction_count)


# print("Count list train: ", count_list_train)
# print("Count list test: ", count_list_test)


if actual_train_stars and predicted_train_stars:
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(actual_train_stars, predicted_train_stars))
    # Print RMSE
    print("Train RMSE:", rmse)
else:
    print("No recommendations can be confirmed as wrong or right, users have not reviewed the recommendations in the train dataset")

if actual_test_stars and predicted_test_stars:
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(actual_test_stars, predicted_test_stars))
    # Print RMSE
    print("Test RMSE:", rmse)
else:
    print("No recommendations can be confirmed as wrong or right, users have not reviewed the recommendations in the test dataset")
