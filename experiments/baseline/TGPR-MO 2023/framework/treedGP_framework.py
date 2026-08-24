import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/TreedGP_MOEA/')

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.surrogate_treedGP import treeGP as treedGP
from desdeo_problem.testproblems.TestProblems import test_problem_builder
import numpy as np
import pandas as pd
import time


def build_surrogates(
    nobjs,
    nvars,
    x_data,
    y_data,
    x_low,
    x_high,
    robust_small_data=False,
    random_state=None,
):
    x_names = [f'x{i}' for i in range(1,nvars+1)]
    y_names = [f'f{i}' for i in range(1,nobjs+1)]
    row_names = ['lower_bound','upper_bound']
    data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    # The original 10*nvars setting cannot split the tree for the N=100
    # small-data protocol.
    # Cap it at half the available fitting rows so at least one statistically
    # meaningful split remains possible, while preserving the original value
    # whenever the dataset is large enough.
    min_samples_leaf = max(2, min(10 * nvars, len(x_data) // 2))
    problem.train(
        treedGP,
        model_parameters={
            'min_samples_leaf': min_samples_leaf,
            'robust_small_data': robust_small_data,
            'random_state': random_state,
        },
    )
    return problem


def run_treed_GP(
    x_data,
    y_data,
    x_low,
    x_high,
    I_max=None,
    G_max=50,
    framework=False,
    robust_small_data=False,
    random_state=None,
):
    nvars = np.shape(x_data)[1]
    nobjs = np.shape(y_data)[1]
    nsamples = np.shape(x_data)[0]
    if framework is False:
        if robust_small_data:
            I_max = max(1, int(np.ceil(nsamples / (10 * nvars))))
        else:
            I_max= nsamples/(10*nvars)
    if robust_small_data:
        I_max = max(1, int(I_max))
        G_max = max(1, int(G_max))

    start = time.time()
    print("Building trees...")
    surrogate_problem = build_surrogates(
        nobjs,
        nvars,
        x_data,
        y_data,
        x_low,
        x_high,
        robust_small_data=robust_small_data,
        random_state=random_state,
    )
    print("Building leaf node GPs...")
    total_points_all = 0
    total_points_all_sequence = []
    total_points_per_model = np.zeros(nobjs)
    total_points_per_model_sequence = None
    delta_total_point = 1
    evolver_opt_tree = RVEA(surrogate_problem, use_surrogates=True, n_iterations=I_max, n_gen_per_iter=G_max)
    while evolver_opt_tree.continue_evolution() and delta_total_point > 0:
        evolver_opt_tree.iterate()    
        population_opt_tree = evolver_opt_tree.population
        X_solutions = population_opt_tree.individuals
        # Add a GP in every tree
        added_gps = 0
        for i in range(nobjs):
            added = surrogate_problem.objectives[i]._model.addGPs(X_solutions)
            if robust_small_data:
                added_gps += int(bool(added))
            else:
                total_points_all += surrogate_problem.objectives[i]._model.total_point_gps
            total_points_per_model[i] = surrogate_problem.objectives[i]._model.total_point
        if robust_small_data:
            total_points_all = int(np.sum(total_points_per_model))
        total_points_all_sequence = np.append(total_points_all_sequence,total_points_all)
        if total_points_per_model_sequence is None:
            total_points_per_model_sequence = total_points_per_model.copy()
        else:
            total_points_per_model_sequence = np.vstack((total_points_per_model_sequence, total_points_per_model))
        # Check whether building has converged
        if robust_small_data:
            delta_total_point = added_gps
        elif evolver_opt_tree._iteration_counter > 5:
            delta_total_point = total_points_all - total_points_all_sequence[evolver_opt_tree._iteration_counter-3]    
        if robust_small_data:
            if added_gps > 0 and evolver_opt_tree.continue_evolution():
                evolver_opt_tree._refresh_population(reevaluate_only=True)
        else:
            evolver_opt_tree._refresh_population()

    end = time.time()
    print("Building finished...")
    time_taken = end - start
    if framework is True:
        while evolver_opt_tree.continue_evolution():
            evolver_opt_tree.iterate()
        population = evolver_opt_tree.population
        results_dict = {
                'individual_archive': population.individuals_archive,
                'objectives_archive': population.objectives_archive,
                'uncertainty_archive': population.uncertainty_archive,
                'individuals_solutions': population.individuals,
                'obj_solutions': population.objectives,
                'uncertainty_solutions': population.uncertainity,
                'time_taken': time_taken,
                'total_points':  total_points_per_model,
                'total_points_per_model_sequence': total_points_per_model_sequence
            }
        return results_dict
    else:
        return (surrogate_problem, total_points_per_model, total_points_per_model_sequence)
