import random

import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools

from src.helpers import *

POP_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.01
MAX_GENERATIONS = 200
GENERATIONS_EXIT = 20  # state the number of generations in which maxFitness remains the same,  for stopping condition


def main():
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
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.8)

    maxFitValues = []
    maxGenValues = []

    # due to stochastic nature of the Genetic Algorithms, run GA 10 times
    for i in range(10):
        population = toolbox.populationCreator(POP_SIZE)
        generationCounter = 0

        fitnessValues = list(map(toolbox.evaluate, population))

        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        fitnessValues = [individual.fitness.values[0] for individual in population]

        maxFitnessValues = []
        meanFitnessValues = []

        rmseList = []
        maeList = []

        while max(fitnessValues) < 1 and generationCounter < MAX_GENERATIONS:
            # Stopping condition #3: If maxFitness remains the same for VALUE_GEN generations, stop algorithm
            if generationCounter > GENERATIONS_EXIT and (
                    all(v == maxFitnessValues[len(maxFitnessValues) - 1] for v in maxFitnessValues[
                                                                                  (len(
                                                                                      maxFitnessValues) - GENERATIONS_EXIT):len(
                                                                                      maxFitnessValues)])):
                print("Stopping conditions occured. Genetic Algorithm now stops.")
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
                if mutant != tools.selBest(offspring, 3, fit_attr="fitness"):
                    if random.random() < P_MUTATION:
                        toolbox.mutate(mutant)
                    del mutant.fitness.values

            freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            # repair individuals with the right static ratings
            for child in offspring:
                repair_function(child, user)

            population[:] = offspring
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)
            print("- Generation {}: Max Fitness = {}, Avg Fitness = {}"
                  .format(generationCounter, maxFitness, meanFitness))

            best_index = fitnessValues.index(max(fitnessValues))
            RMSE = count_error(population[best_index])[0]
            MAE = count_error(population[best_index])[1]

            rmseList.append(RMSE)
            maeList.append(MAE)
            # print("Best Individual = ", *population[best_index], "\n")

        maxGenValues.append(generationCounter)
        maxFitValues.append(max(maxFitnessValues))

    print(count_error(population[best_index]), user_selected)
    print("Average Fitness of best individual is:", np.mean(maxFitValues))
    print("Average generation when stopping conditions are met:", np.mean(maxGenValues))
    plt.figure(0)
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()

    plt.figure(1)
    plt.plot(rmseList, color='blue')
    plt.plot(maeList, color='green')
    plt.xlabel('Generation')
    plt.ylabel('RMSE / MAE')
    plt.title('RMSE and MAE over Generations')
    plt.show()


if __name__ == '__main__':
    main()
