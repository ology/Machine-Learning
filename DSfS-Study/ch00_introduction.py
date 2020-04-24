# https://github.com/joelgrus/data-science-from-scratch

from collections import defaultdict, Counter

users = [
    { "id": 0, "name": "Hero"},
    { "id": 1, "name": "Dunn"},
    { "id": 2, "name": "Sue"},
    { "id": 3, "name": "Chi"},
    { "id": 4, "name": "Thor"},
    { "id": 5, "name": "Clive"},
    { "id": 6, "name": "Hicks"},
    { "id": 7, "name": "Devin"},
    { "id": 8, "name": "Kate"},
    { "id": 9, "name": "Klein"},
]

friendship_pairs = [
    (0,1),
    (0,2),
    (1,2),
    (1,3),
    (2,3),
    (3,4),
    (4,5),
    (5,6),
    (5,7),
    (6,8),
    (7,8),
    (8,9),
]

friendships = { user["id"]: [] for user in users }
for i,j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)
print('friendships:', friendships)

def number_of_friends(user):
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(number_of_friends(user) for user in users)
print('total_connections:', total_connections)

num_users = len(users)
print('num_users:', num_users)

avg_connections = total_connections / num_users
print('avg_connections:', avg_connections)

num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]
num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1], reverse=True)
print('num_friends_by_id:', num_friends_by_id)

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]
        for foaf_id in friendships[friend_id]
        if foaf_id != user_id
        and foaf_id not in friendships[user_id]
    )

i = 3
print('number of friends in common for', i, friends_of_friends(users[i]))

######################################################################

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"), (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"), (1, "Postgres"),
    (2, "Python"), (2, "scikit-learn"), (2, "scipy"), (2, "numpy"), (2, "statsmodels"), (2, "pandas"),
    (3, "R"), (3, "Python"), (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"), (4, "libsvm"),
    (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"), (5, "Haskell"), (5, "programming languages"),
    (6, "statistics"), (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"), (7, "neural networks"),
    (8, "neural networks"), (8, "deep learning"), (8, "Big Data"), (8, "artificial intelligence"),
    (9, "Hadoop"), (9, "Java"), (9, "MapReduce"), (9, "Big Data"),
]

user_ids_by_interest = defaultdict(list)
for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)
#print('user_ids_by_interest:', user_ids_by_interest)

interest_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interest_by_user_id[user_id].append(interest)
#print('interest_by_user_id:', interest_by_user_id)

def most_common_interests_with(user):
    return Counter(
        interested_user_id
        for interest in interest_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user["id"]
    )
print('most_common_interests_with', i, most_common_interests_with(users[i]))

######################################################################

salaries_and_tenures = [
    (83000, 8.7),
    (88000, 8.1),
    (48000, 0.7),
    (76000, 6),
    (69000, 6.5),
    (76000, 7.5),
    (60000, 2.5),
    (83000, 10),
    (48000, 1.9),
    (63000, 4.2),
]

def tenure_bucket(tenure):
    if tenure < 2:
        return 'less than two'
    elif tenure < 5:
        return 'between two and five'
    else:
        return 'more than five'

salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

average_salary_by_bucket = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}
print('average_salary_by_bucket:', average_salary_by_bucket)

######################################################################

words_and_counts = Counter(
    word
    for user, interest in interests
    for word in interest.lower().split()
)
print('words_and_counts:')
for word, count in words_and_counts.most_common():
    if count > 1:
        print('\t', word, count)
