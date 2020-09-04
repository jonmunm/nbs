from sklearn.preprocessing import FunctionTransformer as FT
from typing import List, Dict, Tuple
import numpy as np

Feature = Dict[str, list]
FeaturesList = List[Feature]


class FeatureExtractor:
    def __init__(self, tms_fn:callable):
        self.tms_fn = tms_fn
        
    def transform(self, features:np.ndarray=list()):
        return self.tms_fn(features)