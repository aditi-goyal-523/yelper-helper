
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter
import math



# makes the original datasets a valid json file
def make_valid_json(input_file, output_file):
    # Read input JSON file
    with open(input_file, 'r') as f:
        data = f.readlines()

    # Create output JSON file
    with open(output_file, 'w') as f:
        # Start of JSON array
        f.write('[')

        # Loop through input JSON objects
        count = 0
        for line in data:
            try:
                # Load JSON object
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip non-JSON lines
                continue

            # Write JSON object to output file
            if count > 0:
                # Add comma separator between objects
                f.write(',')
            json.dump(obj, f)
            count += 1

        # End of JSON array
        f.write(']')


# make_valid_json('yelp_academic_dataset_business.json', 'full_business.json')
# make_valid_json('yelp_academic_dataset_tip.json', 'full_tip.json')
# make_valid_json('yelp_academic_dataset_checkin.json', 'full_checkin.json')
# make_valid_json('yelp_academic_dataset_review.json', 'full_review.json')
# make_valid_json('yelp_academic_dataset_user.json', 'full_user.json')


# gets all the food businesses from the input file and returns them into the output file
def filter_restaurants(input_file, output_file):
    # Open the input and output files
    with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
        # Load the input JSON data
        data = json.load(input_file)

        # Create a list to store the filtered objects
        filtered_objects = []
        positive_count = 0
        negative_count = 0
        # Loop through each object in the JSON data
        for obj in data:
            # Check if "Food" is in the category array
            if obj.get("categories") and ("food" in obj["categories"].lower() or "restaurant" in obj["categories"].lower()):
                # Add the object to the filtered list
                filtered_objects.append(obj)
                positive_count += 1
            else:
                print(obj)
                negative_count += 1

        # Write the filtered objects to the output file
        json.dump(filtered_objects, output_file)

# filter_restaurants('full_business.json', 'food_businesses.json')


# prints the amount of json objects in a file
def print_length_json(input_file):
    with open(input_file, 'r') as file:
        # Load the JSON data
        data = json.load(file)
        # Get the number of objects
        count = len(data)
        # Print the count
        print(f"The JSON file contains {count} objects.")

#print_length_json('full_business.json')
#print_length_json('food_businesses.json')


# returns an array of all the elements from the input_file
# ex: gets the business_id's from the food_businesses.json file
def get_element_from_json(input_file, element):
    with open(input_file, 'r') as file:
        data = json.load(file)
        # Initialize an empty list to store the business IDs
        elements = []
        # Loop through each object in the JSON data
        for obj in data:
            # Get the business ID from the current object
            target_element = obj[element]
            # Add the business ID to the list
            elements.append(target_element)
        return elements


# # gets unique elements
# def get_unique_element_from_json(input_file, element):
#     with open(input_file, 'r') as file:
#         # Load the JSON data
#         data = json.load(file)
#         # Initialize an empty set to store the unique elements
#         elements = set()
#         # Loop through each object in the JSON data
#         for obj in data:
#             # Get the element value from the current object
#             target_element = obj[element]
#             # Add the element value to the set
#             elements.add(target_element)
#         return list(elements)
# # looks through input_file for element matching in the element_array
# # ex: filter the review file based off the business_id element using the food_business_ids array
# def filter_by_element(input_file, output_file, element_array, element_name):
#     with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
#         # Load the JSON data
#         data = json.load(input_file)
#         # Initialize an empty list to store the filtered objects
#         filtered_objects = []
#         positive_count = 0
#         negative_count = 0
#         counter = 0;
#         # Loop through each object in the JSON data
#         for obj in data:
#             # Get the element from the current object
#             target_element = obj.get(element_name)
#             # Check if the target_element is in the element_array
#             if target_element in element_array:
#                 # Add the object to the filtered list
#                 filtered_objects.append(obj)
#                 positive_count += 1
#             else:
#                 negative_count += 1
#             counter += 1
#             if counter % 1000 == 0:
#                 print("counter:", counter)
#         # Write the filtered objects to the output file
#         print("Positive count:", positive_count)
#         print("Negative count:", negative_count)
#         json.dump(filtered_objects, output_file)



# uses a dataframe to get all the unique elements for a given name
def get_unique_element_from_json(input_file, element):
    # Read the JSON file into a DataFrame
    df = pd.read_json(input_file)
    # Get unique values of the specified element
    unique_elements = df[element].unique().tolist()
    return unique_elements

# uses a dataframe to filter by a given element name
def filter_by_element(input_file, output_file, element_array, element_name):
    # Read the JSON file into a DataFrame
    df = pd.read_json(input_file)
    # Filter the DataFrame based on the element array
    filtered_df = df[df[element_name].isin(element_array)]
    # Write the filtered DataFrame to the output file
    filtered_df.to_json(output_file, orient='records')


# food_business_ids = get_element_from_json('food_businesses.json', 'business_id')
# filter_by_element('full_review.json', "restaurant_reviews.json", food_business_ids, 'business_id')
# filter_by_element('full_checkin.json', "restaurant_checkin.json", food_business_ids, 'business_id')
# filter_by_element('full_tip.json', "restaurant_tip.json", food_business_ids, 'business_id')

# restaurant_user_ids = get_unique_element_from_json('restaurant_reviews.json', 'user_id')
# print(len(restaurant_user_ids))
# print(restaurant_user_ids[0])
# filter_by_element('filtered_user.json', "restaurant_users.json", restaurant_user_ids, 'user_id')
# print_length_json('restaurant_users.json')


# old code for splitting data
def split_file_80_20(original_file, file_for_20, file_for_80):
    random.seed(42)

    # Load the restaurant_reviews.json file
    with open(original_file, 'r') as file:
        data = json.load(file)

    # Shuffle the data randomly
    random.shuffle(data)

    # Calculate the split index
    split_index = int(0.2 * len(data))

    # Split the data
    test_data = data[:split_index]
    train_data = data[split_index:]

    # Write the test data to restaurant_reviews_test.json
    with open(file_for_20, 'w') as file:
        json.dump(test_data, file)

    # Write the train data to restaurant_reviews_80.json
    with open(file_for_80, 'w') as file:
        json.dump(train_data, file)
# split_file_80_20('restaurant_reviews.json', 'restaurant_reviews_test.json', 'restaurant_reviews_80.json')
# split_file_80_20('restaurant_reviews_80.json', 'restaurant_reviews_valid.json', 'restaurant_reviews_train.json')


# old code for splitting data
def split_X_Y_sets(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    x_train = []
    y_train = []

    for obj in data:
        stars = obj['stars']
        # Create a new dictionary excluding 'review_id', 'stars', and 'date' fields
        x = {key: value for key, value in obj.items() if key not in ['review_id', 'stars', 'date']}

        # Convert 'date' field to datetime
        # x['date'] = datetime.strptime(obj['date'], '%Y-%m-%d %H:%M:%S')

        x_train.append(x)
        y_train.append(stars)

        return x_train, y_train

#X_train, Y_train = split_X_Y_sets('restaurant_reviews_train.json')
#X_valid, Y_valid = split_X_Y_sets('restaurant_reviews_valid.json')





# # plots the distribution of data
# with open('food_businesses.json') as file:
#     data = json.load(file)
#
# # Extract "review_count" values
# review_counts = [obj['review_count'] for obj in data]
#
# # Calculate the mean and standard deviation
# mean_review_count = np.mean(review_counts)
# std_review_count = np.std(review_counts)
#
# # Print the results
# print("Mean review count:", mean_review_count)
# print("Standard deviation of review count:", std_review_count)
#
# # count_0 = review_counts.count(0)
# # count_1 = review_counts.count(1)
# # count_2 = review_counts.count(2)
# # count_3 = review_counts.count(3)
# # count_4 = review_counts.count(4)
# # count_5 = review_counts.count(5)
# #
# # # Step 3: Print the results
# # print("Number of users with review count 0:", count_0)
# # print("Number of users with review count 1:", count_1)
# # print("Number of users with review count 2:", count_2)
# # print("Number of users with review count 3:", count_3)
# # print("Number of users with review count 4:", count_4)
# # print("Number of users with review count 5:", count_5)
#
# # print_length_json('restaurant_users.json')
#
# #  Plot all users
# plt.figure(1)
# plt.plot(review_counts)
# plt.xlabel('Business')
# plt.ylabel('Review Count')
# plt.title('Review Count Distribution')
#
# # Save the plot to a file
# plt.savefig('business_distribution.png')
#
# # Count the number of users with each review count
# counted_review_counts = dict()
# for count in review_counts:
#     if count in counted_review_counts:
#         counted_review_counts[count] += 1
#     else:
#         counted_review_counts[count] = 1
#
# # Create lists for x-axis (review count values) and y-axis (number of users)
# x_values = list(range(0,200))  # review count values from 0 to 50
# y_values = [counted_review_counts.get(count, 0) for count in x_values]
#
# # Step 5: Plot the distribution
# plt.bar(x_values, y_values)
# plt.xlabel('Review Count')
# plt.ylabel('Number of Businesses')
# plt.title('Business Review Count Distribution')
# plt.xlim(0, 200)  # Set the x-axis limits
# plt.savefig('business_distribution_0-200.png')





#Utility
# # Read the full_user.json file
# with open('full_user.json', 'r') as file:
#     data = json.load(file)
#
# # Filter out entries with review_count equal to 0 or 1
# filtered_data = [entry for entry in data if entry.get('review_count', 0) not in (0, 1)]
#
# # Write the filtered data to a new JSON file
# with open('filtered_user.json', 'w') as file:
#     json.dump(filtered_data, file)
#
# print_length_json('full_user.json')
# print_length_json('filtered_user.json')




#puts the restaurant reviews into train/validation/test splits
def filter_restaurant_reviews(input_file, train, validation, test):
    # Read the JSON file and load its contents
    with open(input_file) as f:
        data = json.load(f)

    # Create a dictionary to store the list of objects for each user_id
    user_id_objects = {}

    # Iterate through the JSON objects and group them by user_id
    for obj in data:
        user_id = obj.get('user_id')
        if user_id:
            if user_id not in user_id_objects:
                user_id_objects[user_id] = []
            user_id_objects[user_id].append(obj)

    # Create a list to store the train/validation/test objects
    test_data = []
    validation_data = []
    train_data = []

    # Iterate through the grouped objects and select a random subset for each user_id
    for user_id, objects in user_id_objects.items():
        count = len(objects)

        # Special cases for counts 0, 1, 2, 3, 4
        if count == 0 or count == 1:
            print("error, count for this user should not be 0 or 1. User_id:", user_id)
            # train_data.extend(objects)
        elif count == 2:
            random_object = random.choice(objects)
            test_data.append(random_object)
            objects.remove(random_object)
            train_data.extend(objects)
        elif count == 3:
            random_object = random.choice(objects)
            test_data.append(random_object)
            objects.remove(random_object)
            train_data.extend(objects)
        elif count == 4:
            random_object = random.choice(objects)
            test_data.append(random_object)
            objects.remove(random_object)
            train_data.extend(objects)
        elif count >= 5:
            num_to_select_20_percent = int(count * 0.2)
            random_objects_20_percent = random.sample(objects, num_to_select_20_percent)
            test_data.extend(random_objects_20_percent)
            for obj in random_objects_20_percent:
                objects.remove(obj)

            num_to_select_16_percent = int(count * 0.16)
            random_objects_16_percent = random.sample(objects, num_to_select_16_percent)
            validation_data.extend(random_objects_16_percent)

            for obj in random_objects_16_percent:
                objects.remove(obj)

            train_data.extend(objects)

    # 20% of the unique user_id count store in Test
    with open(test, 'w') as f:
        json.dump(test_data, f)

    # 16% of the unique user_id count store in Validation
    with open(validation, 'w') as f:
        json.dump(validation_data, f)

    # 64% of the unique user_id count store in Train
    with open(train, 'w') as f:
        json.dump(train_data, f)


# filter_restaurant_reviews('target_reviews.json', 'reviews_test.json','reviews_validation.json', 'reviews_train.json')
# print_length_json('reviews_train.json')
# print_length_json('reviews_validation.json')
# print_length_json('reviews_test.json')


# filter_restaurant_reviews('SB_u10.json', 'SB_train.json', 'SB_validation.json', 'SB_test.json')
# print("Train:")
# print_length_json('SB_train.json')
# print("Validation:")
# print_length_json('SB_validation.json')
# print("Test:")
# print_length_json('SB_test.json')

# gets the reviews if the user who reviewed the restaurant had [atleast_this_many_reviews] or [atmost_this_many_reviews] in their history
def get_target_reviews(input_file, output_file, atleast_this_many_reviews, atmost_this_many_reviews, element_name):
    target_reviews = []
    not_added_counter = 0
    with open(input_file) as f:
        data = json.load(f)
        # Create a dictionary to store the list of objects for each user_id
        name_objects = {}

        # Iterate through the JSON objects and group them by user_id
        for obj in data:
            name = obj.get(element_name)
            if name:
                if name not in name_objects:
                    name_objects[name] = []
                name_objects[name].append(obj)

        # Iterate through the grouped objects and select a random subset for each user_id
        for name, objects in name_objects.items():
            count = len(objects)

            # get rid of reviews that user_id had too few or too high
            if atleast_this_many_reviews <= count <= atmost_this_many_reviews:
                target_reviews.extend(objects)
            else:
                not_added_counter = not_added_counter + count

        print("not added total:", not_added_counter)

    with open(output_file, 'w') as f:
        json.dump(target_reviews, f)

# get_target_reviews('restaurant_reviews.json', 'target_reviews.json', 2, 10000000, 'user_id')
# print_length_json('target_reviews.json')
# print_length_json('restaurant_reviews.json')

# get_target_reviews("reviews_train.json", "small_train.json", 20, 20, 'user_id')
# print_length_json("reviews_train.json")
# print_length_json("small_train.json")


# get_target_reviews('reviews_train.json', 'train_50.json', 50, 100000000, 'business_id')
# get_target_reviews('combined_test.json', 'test_50.json', 50, 100000000, 'business_id')
# print_length_json('train_50.json')
# print_length_json('test_50.json')

# get_target_reviews('reviews_train.json', 'train_100.json', 100, 100000000, 'business_id')
# get_target_reviews('combined_test.json', 'test_100.json', 100, 100000000, 'business_id')
# print_length_json('train_100.json')
# print_length_json('test_100.json')

# get_target_reviews('train_100.json', 'train_greater_10.json', 10, 100000000, 'user_id')
# get_target_reviews('test_100.json', 'test.json', 25, 100000000, 'user_id')
# print_length_json('train_greater_10.json')
# print_length_json('test.json')

# get_target_reviews('santa_barbara_businesses.json', 'SB_u10.json', 10, 100000000, 'user_id')
# print_length_json('SB_u10.json')




# only get businesses near San Fran
def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 6371 * c  # Radius of the Earth in kilometers

    return distance

def find_restaurants(city_lat, city_lon, max_distance, restaurant_data):
    nearby_restaurants = []

    for restaurant in restaurant_data:
        restaurant_lat = restaurant['latitude']
        restaurant_lon = restaurant['longitude']
        distance = calculate_distance(city_lat, city_lon, restaurant_lat, restaurant_lon)

        if distance <= max_distance:
            nearby_restaurants.append(restaurant)

    return nearby_restaurants

# with open('food_businesses.json', 'r') as file:
#     restaurant_data = json.load(file)
#
#
# city_lat = 37.7749  # Latitude of city center (San Francisco)
# city_lon = -122.4194  # Longitude of city center (San Francisco)
# max_distance = 50  # Maximum distance in kilometers
#
# nearby_restaurants = find_restaurants(city_lat, city_lon, max_distance, restaurant_data)
#
# print(len(nearby_restaurants))
#
# filter_by_element('reviews_train.json', 'train.json', nearby_restaurants, 'business_id')
# filter_by_element('reviews_validation.json', 'validation.json', nearby_restaurants, 'business_id')
# filter_by_element('reviews_test.json', 'test.json', nearby_restaurants, 'business_id')
#
# print_length_json('reviews_train.json')
# print_length_json('reviews_validation.json')
# print_length_json('reviews_test.json')
#
# print_length_json('train.json')
# print_length_json('validation.json')
# print_length_json('test.json')



def combine_json_files(input_file1, input_file2, output_file):
    # Read the contents of the first JSON file
    with open(input_file1, 'r') as file1:
        data1 = json.load(file1)

    # Read the contents of the second JSON file
    with open(input_file2, 'r') as file2:
        data2 = json.load(file2)

    # Merge the two JSON objects into one
    merged_data = data1 + data2

    # Convert the merged object back into JSON format
    merged_json = json.dumps(merged_data)

    # Write the merged JSON data to a new file
    with open(output_file, 'w') as merged_file:
        merged_file.write(merged_json)

#
# combine_json_files('reviews_validation.json', 'reviews_test.json','combined_test.json')
# print_length_json('combined_test.json')

# combine_json_files('SB_validation.json', 'SB_test.json', 'SB_combined_test.json')
# print("Combined:")
# print_length_json('SB_combined_test.json')

def filter_postal_code(input_file1, input_file2, output_file, postal_code):
    with open(input_file1, 'r') as file1:
        data1 = json.load(file1)
    # Initialize an empty list to store the business IDs
    elements = []
    # Loop through each object in the JSON data
    for obj in data1:

        if obj['postal_code'] == postal_code:
            target_element = obj['business_id']
            # Add the business ID to the list
            elements.append(target_element)
            print("MATCH")
            print(target_element)
    print(len(elements))
    filter_by_element(input_file2, output_file, elements, 'business_id')

# filter_postal_code('food_businesses.json', 'restaurant_reviews.json','santa_barbara_businesses.json', '93101')
# print_length_json('santa_barbara_businesses.json')


