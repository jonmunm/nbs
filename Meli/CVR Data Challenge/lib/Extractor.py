import numpy as np

class FeatureExtractor:
    def __init__(self, tms_fn:callable):
        self.tms_fn = tms_fn
        
    def transform(self, features:np.ndarray) -> np.ndarray:
        return self.tms_fn(features)