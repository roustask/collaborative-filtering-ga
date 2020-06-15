import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from scipy.stats import pearsonr

df_whole = pd.read_csv("../ml-100k/u.data", delimiter='\t')
df_train = pd.read_csv("../ml-100k/ua.base", delimiter='\t')
df_test = pd.read_csv("../ml-100k/ua.test", delimiter='\t')


# prepare u.data dataframe
def prepare_df(df):
    df.columns = ['user', 'movies', 'ratings', 'timestamp']
    df.sort_values(by=['user', 'movies'], inplace=True)
    df = df.pivot_table(index=['user', ], columns=['movies'],
                        values='ratings').reset_index(drop=True)
    return df


# prepare ua.base and ua.test dataframes, used for running the algorithm and testing it, respectively
def prepare_train_test(df):
    to_merge = prepare_df(df)
    empty_df = pd.DataFrame(np.nan, index=prepare_df(df_whole).index, columns=prepare_df(df_whole).columns)
    merged = empty_df.merge(right=to_merge, how='right')
    return merged


# fill user with random
def fill_with_random(matrix):
    matrix = np.copy(matrix)
    nan_number = np.isnan(matrix).sum()
    filler = np.random.randint(1, 6, nan_number, dtype='int')
    mask = np.isnan(matrix)
    matrix[mask] = filler
    return matrix


# fill with average per user
def fill_with_average(df):
    for i in range(len(df.index)):
        avg = df.iloc[i].mean()
        df.iloc[i].fillna(avg, inplace=True)
    matrix = np.round(np.array(df))
    return matrix


def find_neighbors():
    pearson_values = [[pearsonr(users[i], users[j])[0] for j in range(len(users))] for i in range(len(users))]
    pearson_values = np.array(pearson_values)
    indices = [np.argpartition(i, -10)[-10:] for i in pearson_values]
    top10 = [users[i].tolist() for i in indices]
    return top10


# evaluation function based on the average pearson correlation between the user that was selected and his neighbors
def evaluation_function(individual):
    average_pearson = [[pearsonr(individual[i], neigh)[0] for neigh in neighbors[i]] for i in range(len(users))]
    average_pearson = np.mean(average_pearson)
    return average_pearson,


def repair_function(ind):
    ind = np.copy(ind)
    df = np.copy(prepare_train_test(df_train))
    mask = ~np.isnan(df)
    ind[mask] = df[mask]


users = fill_with_average(prepare_train_test(df_train))

neighbors = find_neighbors()

individual_gen = partial(fill_with_random, prepare_train_test(df_train))


def main():
    POP_SIZE = 20
    P_CROSSOVER = 0.9
    P_MUTATION = 0.05
    MAX_GENERATIONS = 30
    GENERATIONS_EXIT = 10  # state the number of generations in which maxFitness remains the same,  for stopping condition

    toolbox = base.Toolbox()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox.register("individualCreator", lambda individual, individual_gen: individual(individual_gen().tolist()),
                     creator.Individual,
                     individual_gen)
    toolbox.register("populationCreator", tools.initRepeat,
                     list, toolbox.individualCreator)

    toolbox.register("evaluate", evaluation_function)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.8)

    population = toolbox.populationCreator(n=POP_SIZE)
    generationCounter = 0

    fitnessValues = list(map(toolbox.evaluate, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    fitnessValues = [individual.fitness.values[0] for individual in population]

    maxFitnessValues = []
    meanFitnessValues = []

    while max(fitnessValues) < 1 and generationCounter < MAX_GENERATIONS:
        # Stopping condition #3: If maxFitness remains the same for VALUE_GEN generations, stop algorithm
        if generationCounter > GENERATIONS_EXIT and (
                all(v == maxFitnessValues[len(maxFitnessValues) - 1] for v in maxFitnessValues[
                                                                              (len(
                                                                                  maxFitnessValues) - GENERATIONS_EXIT):len(
                                                                                  maxFitnessValues)])):
            break

        generationCounter = generationCounter + 1
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        # elite individuals don't get mutated
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
            del mutant.fitness.values

        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        # repair individuals with the right static ratings
        for child in offspring:
            repair_function(child)

        population[:] = offspring
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}"
              .format(generationCounter, maxFitness, meanFitness))

        best_index = fitnessValues.index(max(fitnessValues))
        # print("Best Individual = ", *population[best_index], "\n")

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()
