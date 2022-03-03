"""
Created on Feb 2022
@author: benedetta gambosi benedetta.gambosi@polimi.it

"""



#if __name__ == '__main__':
import nest
from deap import creator, base, tools, algorithms
from simulate_controller import simulate_controller
#import sim_param
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import os
import math

def remove_files():
    for f in os.listdir(data_folder):
        if '.gdf' in f or '.dat' in f:
            os.remove(data_folder+f)

# Reconfigure scaffold
filename_h5 = "300x_200z_claudia_dcn_test_3.hdf5"
filename_config = 'mouse_cerebellum_cortex_update_dcn_copy_post_stepwise_colonna_X.json'
data_folder = "./"


NUM_IND = int(sys.argv[1])      #10
NUM_GEN = int(sys.argv[2])      #5

CX_PB =
MUT_PB = 

NUM_PARAM = 9

low = np.array([800, 0.01,8, 0.05, 0.1, 0.1, 0.001, 0.001, 0.0001])
up = np.array([1500, 2.0, 12, 0.5, 3, 3, 0.05, 0.05, 0.05])

# Variables to save generation, individuals and corresponding fitness values
log_variable = []

def evalFitness(individual, name_folder):
    print("Evaluating fitness for individual ", name_folder, " on rank ")

    values = np.asarray(individual)
    individual_in_range = values*(up-low) + low
    path_saving = os.getcwd() +"/Data/"+ name_folder
    print(path_saving)

    error_x, error_y, error_all = simulate_controller(individual_in_range, path_saving)

    return error_x, error_y, error_all

# Create fitness and individual classes
creator.create("FitnessController" ,base.Fitness, weights=(-1.0,-1.0, -3.0))
creator.create("Individual", list, fitness = creator.FitnessController)

# Initialize DEAP toolbox (--> individual, population and GA operators)
toolbox = base.Toolbox()
toolbox.register("num", random.uniform, 0, 1)
toolbox.register("individual",tools.initRepeat, creator.Individual, toolbox.num, n = NUM_PARAM)
toolbox.register("population",tools.initRepeat, list, toolbox.individual, n = NUM_IND)
toolbox.register("evaluate", evalFitness)

# cxUniform: Executes a uniform crossover that modify in place the two sequence
#individuals. The attributes are swapped according to the indpb probability.
toolbox.register("mate", tools.cxUniform, indpb=0.1)

# mutGaussian: This function applies a gaussian mutation of mean mu and standard deviation
# sigma on the input individual. This mutation expects a sequence individual
# composed of real valued attributes. The indpb argument is the probability of each attribute to be mutated.
toolbox.register("mutate",tools.mutGaussian, mu=0.1, sigma=0.05, indpb=0.05)

# selNSGA2: Apply NSGA-II selection operator on the individuals. Usually,
# the size of individuals will be larger than k because any individual present
# in individuals will appear in the returned list at most once. Having the size
# of individuals equals to k will have no effect other than sorting the population
# according to their front rank. The list returned contains references to the input
# individuals. For more details on the NSGA-II operator see [Deb2002].
toolbox.register("select",tools.selNSGA2)


#with Pool() as pool:
#   pool.workers_exit()

#toolbox.register("map", pool.map)


# Logging info and statistics tools
logbook=tools.Logbook()
stats_fits = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_fits.register("min", np.min, axis=0)
stats_fits.register("avg", np.mean, axis=0)
stats_fits.register("max", np.max, axis=0)
logbook.header = "gen", "evals", "min", "avg", "max"


# Fitness value file
fitfile = open("fitness.dat", 'a')


# Allocation of individuals, folders creation to save data
population = toolbox.population()
hof = tools.HallOfFame(1)
folders = []
for i in range(len(population)):
    folders.append("/gen" +str(0)+"_ind"+str(i))
# print(folders)

# Evaluation of FIRST generation
fits = toolbox.map(toolbox.evaluate, population, folders)


for fit, ind in zip(fits, population):
  
    ind.fitness.values = fit
    
    # Save fitness value in the file
    fitfile.write(str(0)+'\t'+str(ind)+'\t'+"[{:.2f} {:.2f} {:.2f}]".format(*fit)+'\n')
    
    # Save optimization data (gen ind fitness)
    log_variable.append([0, ind, fit])

# Save data of FIRST generation in the logbook
record = stats_fits.compile(population)
logbook.record(gen=0, nevals=len(population), **record)

print(logbook)

# Run optimization
if NUM_GEN>1:
    for gen in range(1,NUM_GEN):
        folders = []
        for i in range(len(population)):
            folders.append("/gen" +str(gen)+"_ind"+str(i))
        #print(folders)
        
        offspring = algorithms.varOr(population, toolbox, lambda_ = NUM_IND, cxpb=CX_PB, mutpb=MUT_PB)
        #offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

        fits = toolbox.map(toolbox.evaluate, offspring, folders)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            # Save fitness value in the file and in a variable
            fitfile.write(str(gen)+'\t'+str(ind)+'\t'+"[{:.2f} {:.2f} {:.2f}]".format(*fit)+'\n')
            # Save optimization data (gen ind fitness)
            log_variable.append([gen, ind, fit])


        population = toolbox.select(offspring + population, k=NUM_IND)
        hof.update(population)
        record = stats_fits.compile(population)
        logbook.record(gen=gen, nevals=len(population), **record)



print(logbook)
print("Best individual is ", hof[0], hof[0].fitness.values)

# Close the file of fitnesses
fitfile.close()


# Extracting data from logbook
gen = logbook.select("gen")
nevals = logbook.select("nevals")

fit_mins = logbook.select("min")
fit_avgs = logbook.select("avg")
fit_maxs = logbook.select("max")

# Writing optimization statistics to file
with open('optimization_stats.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(gen, nevals, fit_mins, fit_avgs, fit_maxs))


# Plotting optimization
fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, fit_maxs, "r-", label="Maximum Fitness")
ax2.set_ylabel("Fitness", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")


lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()


# Saving log_variable to file
import pickle
with open('log_variable.dat', 'wb') as f:
    pickle.dump(log_variable, f)
