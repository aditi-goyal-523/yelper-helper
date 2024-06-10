import json
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import random
import joblib


TRAIN_FILE = ''
TEST_FILE = ''
num_iterations = 100

def get_recommendation(user1, user2, df, actual_train_stars, predicted_train_stars, predicted_count_train):
    counter = 0
    recommendation_list = []

    for column in df.columns:
        if df.loc[user2, column] == 5:
            if df.loc[user1, column] == 0:
                business_id = column
                stars = df.loc[user2, column]
                recommendation_list.append({'business_id': business_id, 'stars': stars, 'user_id': user2})
            else:
                actual_train_stars.append(df.loc[user1, column])
                predicted_train_stars.append(df.loc[user2, column])
                predicted_count_train += 1
    return recommendation_list, predicted_count_train

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



# Load and process the training data
print("loading the training set")
with open(TRAIN_FILE, 'r') as train_file:
    train_data = json.load(train_file)


# Get all unique user_ids and business_ids
user_ids = list(set(obj['user_id'] for obj in train_data))
business_ids = list(set(obj['business_id'] for obj in train_data))

# Sort the user_ids and business_ids
user_ids.sort()
business_ids.sort()

# Create an empty DataFrame with user_ids as rows and business_ids as columns
df = pd.DataFrame(index=user_ids, columns=business_ids)

# Fill the DataFrame with star ratings from the train_data
for obj in train_data:
    user_id = obj['user_id']
    business_id = obj['business_id']
    stars = obj['stars']

    df.loc[user_id, business_id] = stars

# Fill missing values with 0
df.fillna(0, inplace=True)

print(df)

print("Loading the validation set")
# Load and process the validation data
with open(TEST_FILE, 'r') as validation_file:
    validation_data = json.load(validation_file)

validation_user_ids_dict = get_unique_user(validation_data)

print("Training the model")
# Train the KNN model
k = 20 # Number of neighbors to consider
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
knn_model.fit(df)
print("Done training")

joblib.dump(knn_model, 'knn_SB_k20.joblib')
# knn_model = joblib.load('knn_SB_k5.joblib')

# Get a list of indices from the dataframe
indices_list = df.index.tolist()
random.shuffle(indices_list)

count_list_train = []
count_list_test = []
actual_train_stars = []
predicted_train_stars = []
actual_test_stars = []
predicted_test_stars = []
counter = 1
# Iterate over each row in the dataframe
for index in indices_list:
    print(counter)
    if counter >= num_iterations:
        break
    row = df.loc[index]
    distances, indices = knn_model.kneighbors(row.values.reshape(1, -1))
    similar_user_indices = indices[0][1:]  # Exclude the first index
    similar_user_distances = distances[0][1:]  # Exclude the first distance

    print("Distances: ", distances)
    # Get the similar users' details
    similar_users = []
    for i in similar_user_indices:
        similar_users.append(df.iloc[i])

    # Get the business recommendations for a particular user
    # Recommendations will have all the businesses that the OG user has not been to yet
    all_recommendations = []
    predicted_count_train = 0
    for i in range(len(similar_users)):
        recommendations, predicted_count_train = get_recommendation(row.name, similar_users[i].name, df, actual_train_stars, predicted_train_stars, predicted_count_train)
        for recommendation in recommendations:
            all_recommendations.append(recommendation)
    count_list_train.append(predicted_count_train)
    print("TRAIN: ", "User: ", row.name, " has matches count: ", predicted_count_train)
    og_user = find_user_by_id(row.name, validation_user_ids_dict)

    prediction_count = 0
    if og_user is not None:
        for obj in og_user:
            for i in range(len(all_recommendations)):
                if obj['business_id'] == all_recommendations[i]['business_id']:
                    actual_test_stars.append(obj['stars'])
                    predicted_test_stars.append(all_recommendations[i]['stars'])

                    prediction_count += 1
    else:
        print("user_id not found in test data: ", row.name)
    counter += 1
    count_list_test.append(prediction_count)
    print("TEST: ", "User: ", row.name, " has matches count: ", prediction_count)
    if counter >= num_iterations:
        break

# print("Count list train: ", count_list_train)
# print("Count list test: ", count_list_test)

if actual_train_stars and predicted_train_stars:
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(actual_train_stars, predicted_train_stars))
    # Print RMSE
    print("RMSE:", rmse)
else:
    print("No recommendations can be confirmed as wrong or right, users have not reviewed the recommendations in the train dataset")

if actual_test_stars and predicted_test_stars:
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(actual_test_stars, predicted_test_stars))
    # Print RMSE
    print("RMSE:", rmse)
else:
    print("No recommendations can be confirmed as wrong or right, users have not reviewed the recommendations in the test dataset")




