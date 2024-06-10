import json
import numpy as np



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

def get_one_users_elements(data, entry_index):
    # Create a dictionary to store the list of objects for the given user
    user_0_elements = {}
    i = 0
    user_0_elements[i] = []
    user_0_elements[i].append(data[entry_index])
    i = 1

    for i, obj in enumerate(data):
        user_id = obj.get('user_id')
        review_id = obj.get('review_id')
        if user_id == data[entry_index]['user_id']:
            if review_id not in user_0_elements:
                user_0_elements[i] = []
                user_0_elements[i].append(obj)
    return user_0_elements


def find_matches(user, data, user_matches_dict):
    match_count = 0
    # Add a new item (userId) to the user_matches_dict
    user_id = user[0]['user_id']
    if user_id not in user_matches_dict:
        user_matches_dict[user_id] = []

    user_business_ids = {obj["business_id"] for obj in user_objects}
    for obj in data:
        if obj["business_id"] in user_business_ids:
            if obj["user_id"] != user_id:
                match_count += 1
                # Add the userId from obj to the current item (current user) in the user_matches_list
                user_matches_dict[user_id].append(obj['user_id'])
                # Start by just adding every userId.  Later we will add code to count duplicates
    return match_count



#MAIN
with open('small_train.json', 'r') as f:
        data = json.load(f)

unique_users = get_unique_user(data)
print(len(unique_users))
matches_list = []
user_matches_dict = {}   # Contains for each userId, a dictionary of the userIds that rated the same businesses
for user_id, user_objects in unique_users.items():
    matches = find_matches(user_objects, data, user_matches_dict)
    matches_list.append(matches)
    # print("User: ", user_id)
    # print("Matches: ", matches)

for user_id, user_objects in user_matches_dict.items():
    print("User ID:", user_id)
    print("User Objects:", len(user_objects) )
    for user in user_objects:
        print(user, end=', ')
    print("\n")

