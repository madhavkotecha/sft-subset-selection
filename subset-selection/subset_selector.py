import os
import pickle
import numpy as np
import submodlib as sb
from config import CACHE_PATH

class SubsetSelector:
    """
    A class to handle subset selection using facility location optimization.
    Provides functionality for creating and retrieving subsets with caching.
    """
    
    def __init__(self, cache_path=CACHE_PATH):
        self.cache_path = cache_path
        
    def create_subset(self, data_sijs: np.ndarray, utility_name: str, k: float = 0.3):
        subset_name = f'{utility_name}_{k}'
        cache_file = os.path.join(self.cache_path, f"{subset_name}.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        if os.path.exists(cache_file):
            print(f'Subset: {subset_name} found in cache')
            with open(cache_file, 'rb') as f:
                return pickle.load(f), subset_name
                
        print(f'Subset: {subset_name} not found in cache')
        
        data_sijs = self._normalize_matrix(data_sijs)
        n_samples = data_sijs.shape[0]
        subset = self._optimize_facility_location(data_sijs, n_samples, k)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(subset, f)
        print(f'Subset: {subset_name} computed and saved')
        
        return subset, subset_name
    
    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        return (matrix - matrix.min()) / (matrix.max() - matrix.min())
    
    @staticmethod
    def _optimize_facility_location(sijs: np.ndarray, n_samples: int, k: float):
        fl = sb.functions.facilityLocation.FacilityLocationFunction(
            n_samples,
            mode='dense',
            sijs=sijs,
            separate_rep=False
        )
        
        return fl.maximize(
            budget=int(k * n_samples),
            optimizer='LazyGreedy',
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
            show_progress=True
        )
    
    @staticmethod
    def get_subset_data(subset, prompts: list, references: list):
        subset_indices = np.array(subset)[:, 0].astype(int)
        return (
            [prompts[i] for i in subset_indices],
            [references[i] for i in subset_indices]
        )

def create_subset(data_sijs, utility_name, k=0.3):
    selector = SubsetSelector()
    return selector.create_subset(data_sijs, utility_name, k)

def get_subset(subset, prompts, references):
    return SubsetSelector.get_subset_data(subset, prompts, references)