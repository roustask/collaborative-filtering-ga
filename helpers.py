import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# rescale pearson coefficients from (-1,1) to (0,1)
# NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
def rescaler(value):
    value = (value + 1) / 2
    return value


def prepare_df():
    df = pd.read_csv("ml-100k/u.data", delimiter='\t')
    df.columns = ['user', 'movies', 'ratings', 'timestamp']
    df.drop(columns=['timestamp'], inplace=True)
    df.sort_values(by=['user', 'movies'], inplace=True)
    df = df.pivot_table(index=['user', ], columns=['movies'],
                        values='ratings').reset_index(drop=True)
    return df


# fill with random values for the population generator
def fill_with_random(user):
    user = np.copy(user)
    nan_number = np.isnan(user).sum()
    filler = np.random.randint(1, 6, nan_number, dtype='int')
    mask = np.isnan(user)
    user[mask] = filler
    return user


# fill with average per user
def fill_with_average(df):
    for i in range(len(df.index)):
        avg = df.iloc[i].mean()
        df.iloc[i].fillna(avg, inplace=True)
    matrix = np.array(df)
    return matrix


# find 10 nearest neighbors of user, based on pearson correlation coefficient
def find_neighbors():
    pearson_values = [pearsonr(users[user_selected], users[i]) for i in range(len(users))]
    pearson_values = np.array(pearson_values)[:, 0]
    top10 = sorted(range(len(pearson_values)), key=lambda i: pearson_values[i], reverse=True)[1:11]
    return users[top10]


# evaluation function based on the average pearson correlation between the user that was selected and his neighbors
def evaluation_function(individual):
    average_pearson = [pearsonr(individual, neighbor)[0] for neighbor in neighbors]
    average_pearson = [rescaler(value) for value in average_pearson]
    average_pearson = np.mean(average_pearson)
    return average_pearson,


# repair function used to re-enter movie that user has already rated, and should remain static after each generation run of the GA algorithm
def repair_function(ind, user):
    ind = np.copy(ind)
    user = np.copy(user)
    mask = ~np.isnan(user)
    ind[mask] = user[mask]


# select a random user
# user_selected = np.random.randint(1, 944, dtype='int')

# or a fixed one
user_selected = 10
print("User selected is #", user_selected)

df = prepare_df()

user = np.array(df.iloc[user_selected]).tolist()

users = fill_with_average(df)

neighbors = find_neighbors()