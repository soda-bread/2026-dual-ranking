import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_emo.othertools.ProbabilityWrong import Probability_wrong


class ProbMOEAD_select_v3(SelectionBase):
    """The MOEAD selection operator. 

    Parameters
    ----------
    pop : Population
        The population of individuals
    SF_type : str
        The scalarizing function employed to evaluate the solutions

    """
    def __init__(
        self, pop: Population, SF_type: str
    ):
	 # initialize
        self.SF_type = SF_type
        # Kept false for the legacy runner.  The official small-data adapter
        # enables it because uncertain samples can fall below the current ideal;
        # PBI uses a non-negative distance along the reference direction.
        self.use_absolute_pbi_projection = False

    def do(self, pop: Population, vectors: ReferenceVectors, ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> List[int]:
        """Select the individuals that are kept in the neighborhood.

        Parameters
        ----------
        pop : Population
            The current population.
        vectors : ReferenceVectors
            Class instance containing reference vectors.
        ideal_point
            Ideal vector found so far
        current_neighborhood
            Neighborhood to be updated
        offspring_fx
            Offspring solution to be compared with the rest of the neighborhood

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        # Compute the value of the SF for each neighbor
        num_neighbors               = len(current_neighborhood)
        current_population          = pop.objectives[current_neighborhood,:]
        current_uncertainty          = pop.uncertainity[current_neighborhood,:]
        current_reference_vectors   = vectors.values[current_neighborhood,:]
        offspring_population = np.repeat(
            np.asarray(offspring_fx, dtype=float).reshape(1, -1),
            num_neighbors,
            axis=0,
        )
        offspring_uncertainty = np.repeat(
            np.asarray(offspring_unc, dtype=float).reshape(1, -1),
            num_neighbors,
            axis=0,
        )
        ideal_point_matrix          = np.array([ideal_point]*num_neighbors)
        theta_adaptive_matrix       = np.array([theta_adaptive]*num_neighbors)
        n_samples = 1000
        pwrong_current = Probability_wrong(mean_values=current_population, stddev_values=current_uncertainty, n_samples=n_samples)
        pwrong_current.vect_sample_f()

        pwrong_offspring = Probability_wrong(mean_values=offspring_population, stddev_values=offspring_uncertainty, n_samples=n_samples)
        pwrong_offspring.vect_sample_f()

        values_SF_current = self._evaluate_SF(current_population, current_reference_vectors, ideal_point_matrix, pwrong_current, theta_adaptive_matrix)
        values_SF_offspring = self._evaluate_SF(offspring_population, current_reference_vectors, ideal_point_matrix, pwrong_offspring, theta_adaptive_matrix)

        """
        ##### KDE here and then compute probability
        pwrong_current.pdf_list = {}
        pwrong_current.ecdf_list = {}
        pwrong_offspring.pdf_list = {}
        pwrong_offspring.ecdf_list = {}
        values_SF_offspring_temp = np.asarray([values_SF_offspring])
        values_SF_current_temp = np.asarray([values_SF_current])
        pwrong_offspring.compute_pdf(values_SF_offspring_temp.reshape(num_neighbors,1,n_samples))
        pwrong_current.compute_pdf(values_SF_current_temp.reshape(num_neighbors,1,n_samples))
        #pwrong_offspring.plt_density(values_SF_offspring.reshape(20,1,1000))
        probabilities = np.zeros(num_neighbors)
        for i in range(num_neighbors):
            probabilities[i]=pwrong_current.compute_probability_wrong_PBI(pwrong_offspring, index=i)
        # Compare the offspring with the individuals in the neighborhood 
        # and replace the ones which are outperformed by it if P_{wrong}>0.5
        selection = np.where(probabilities>0.5)[0]
        """
        # Considering mean
        mean_current = (
            values_SF_current
            if np.ndim(values_SF_current) == 1
            else np.mean(values_SF_current, axis=1)
        )
        mean_offspring = (
            values_SF_offspring
            if np.ndim(values_SF_offspring) == 1
            else np.mean(values_SF_offspring, axis=1)
        )
        if not np.all(np.isfinite(mean_current)) or not np.all(
            np.isfinite(mean_offspring)
        ):
            raise ValueError("Prob-MOEA/D scalarization produced non-finite values.")
        selection = np.where(mean_offspring < mean_current)[0]
        #print(selection)

        return current_neighborhood[selection]


    def tchebycheff(self, objective_values:np.ndarray, weights:np.ndarray, ideal_point:np.ndarray):
        feval   = np.abs(objective_values - ideal_point) * weights
        max_fun = np.max(feval)
        return max_fun

    def weighted_sum(self, objective_values, weights):
        feval   = np.sum(objective_values * weights)
        return feval

    def pbi(self, objective_values, weights, ideal_point, pwrong_f_samples, theta):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        ideal_point = np.asarray(ideal_point, dtype=float).reshape(-1)
        samples = np.asarray(pwrong_f_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[0] != len(weights):
            raise ValueError("Prob-MOEA/D PBI samples have an invalid shape.")
        norm_weights = np.linalg.norm(weights)
        if not np.isfinite(norm_weights) or norm_weights <= np.finfo(float).eps:
            raise ValueError("Prob-MOEA/D received a zero or invalid reference vector.")
        weights = weights / norm_weights

        centered = samples.T - ideal_point
        d1 = centered @ weights
        if self.use_absolute_pbi_projection:
            d1 = np.abs(d1)
        residual = centered - d1[:, None] * weights[None, :]
        d2 = np.linalg.norm(residual, axis=1)
        return d1 + float(np.asarray(theta).reshape(-1)[0]) * d2


    def _evaluate_SF(self, neighborhood, weights, ideal_point, pwrong, theta_adaptive):
        if self.SF_type == "TCH":
            SF_values = np.array(list(map(self.tchebycheff, neighborhood, weights, ideal_point)))
            return SF_values
        elif self.SF_type == "PBI":
            SF_values = np.array(list(map(self.pbi, neighborhood, weights, ideal_point, pwrong.f_samples, theta_adaptive)))
            return SF_values
        elif self.SF_type == "WS":
            SF_values = np.array(list(map(self.weighted_sum, neighborhood, weights)))
            return SF_values
        else:
            raise ValueError(f"Unknown Prob-MOEA/D scalarization: {self.SF_type}")



    

    

    
    
