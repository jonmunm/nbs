import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from typing import List, Dict, Tuple, Union
import numpy as np

class NumericalTransformer(TransformerMixin):
    def __init__(self, feature:str, pre_tms:List[FT]=list(), imputer:TransformerMixin=None, scaler:TransformerMixin=None, post_tms:List[FT]=list()):
        self.feature = feature
        self.pre_tms = pre_tms
        
        if imputer != False:
            if imputer is None:
                self.imputer = SimpleImputer()
            else:
                self.imputer = imputer
        else:
            self.imputer = None
            
        if scaler != False:
            if scaler is None:
                self.scaler = StandardScaler()
            else:
                self.scaler = scaler
        else:
            self.scaler = None
            
        self.post_tms = post_tms
        
    def fit_transform(self, data:np.ndarray) -> np.ndarray:        
        for tms in self.pre_tms:
            data = tms.transform(data)
        
        data = self.imputer.fit_transform(np.array(data).reshape(-1,1)).squeeze() if self.imputer is not None else data
        data = self.scaler.fit_transform(np.array(data).reshape(-1,1)).squeeze() if self.scaler is not None else data
        
        for tms in self.post_tms:
            data = tms.transform(data)
            
        return data
    
    def transform(self, data:np.ndarray) -> np.ndarray:       
        for tms in self.pre_tms:
            data = tms.transform(data)
            
        data = self.imputer.transform(np.array(data).reshape(-1,1)).squeeze() if self.imputer is not None else data
        data = self.scaler.transform(np.array(data).reshape(-1,1)).squeeze() if self.scaler is not None else data
        
        for tms in self.post_tms:
            data = tms.transform(data)        
        
        return data

class CategoricalTransformer(TransformerMixin):
    def __init__(
        self, 
        feature:str, 
        pre_tms:List[FT]=list(), 
        encoder:TransformerMixin=None, 
        post_tms:List[FT]=list()
    ):
        self.feature = feature
        self.pre_tms = pre_tms

        if encoder != False:
            if encoder is None:
                self.encoder = LabelEncoder()
            else:
                self.encoder = encoder
        else:
            self.encoder = None
            
        self.post_tms = post_tms
        
    def fit_transform(self, data:np.ndarray) -> np.ndarray:
        data = np.array([str(item).strip() for item in data], dtype=object)
        
        for tms in self.pre_tms:
            data = tms.transform(data).astype(np.str)
        
        self.encoder.fit(np.append(data, ['UNKNOWN']))
        data = self.encoder.transform(data) if self.encoder is not None else data
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data
    
    def transform(self, data:np.ndarray) -> np.ndarray:      
        data = np.array([str(item).strip() for item in data], dtype=object)
        
        for tms in self.pre_tms:
            data = tms.transform(data).astype(np.str)
            
        data = np.array([item if item in self.encoder.classes_ else 'UNKNOWN' for item in data], dtype=object)
            
        if self.encoder is not None:
            data = self.encoder.transform(data)
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data
    
class CategoricalFeatureExtractor:
    def __init__(self, features:List[str], tms_fn:callable, encoder:TransformerMixin=None):
        self.features = features
        self.tms_fn = tms_fn
        
        if encoder != False:
            if encoder is None:
                self.encoder = LabelEncoder()
            else:
                self.encoder = encoder
        else:
            self.encoder = None
        
    def fit_transform(self, data:Dict[str, str]) -> np.ndarray:
        data = [item.strip() if item.strip() != '' else 'N/A' for item in data] 
        data = self.tms_fn(np.array(data, dtype=object))
        return self.encoder.fit_transform(data)
    
    def transform(self, data:Dict[str, str]) -> np.ndarray:
        data = [item.strip() if item.strip() != '' else 'N/A' for item in data] 
        data = self.tms_fn(np.array(data, dtype=object))
        
        unique_values = np.unique(data)
        diff = np.setdiff1d(unique_values, self.encoder.classes_, True)
        self.encoder.classes_ = np.append(self.encoder.classes_, diff)     
        
        return self.encoder.transform(data)
    
class CategoricalTargetTransformer(TransformerMixin):
    def __init__(self, target:str):
        self.target = target
        self.encoder = LabelEncoder()
        
    def fit_transform(self, data:Union[List[bool], List[str]]) -> np.ndarray:
        data = np.array([str(item).strip() for item in data], dtype=object)
        return self.encoder.fit_transform(data).reshape(-1,1)
    
    def transform(self, data:Union[List[bool], List[str]]) -> np.ndarray:        
        data = np.array([str(item).strip() for item in data], dtype=object)  
        return self.encoder.transform(data).reshape(-1,1)

class DatasetTransformer():
    def __init__(self, features_tms:List[Tuple[str, TransformerMixin]]):
        #self.features_tms = features_tms
        self.numerical_features = []
        self.categorical_features = []
        
        for feature_tms in features_tms:
            feature, tms = feature_tms

            if isinstance(tms, CategoricalTransformer):
                self.categorical_features.append(feature_tms)
                
            if isinstance(tms, NumericalTransformer):
                self.numerical_features.append(feature_tms)

    def fit_transform(self, features:List[Dict[str, any]]) -> np.ndarray:      
        dataset_len = len(features)
        
        numerical_features = np.zeros((dataset_len, len(self.numerical_features)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.categorical_features)), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for feature_tms in self.categorical_features:
            feature, tms = feature_tms
            _features = tms.feature
            data = np.array([record[_features] for record in features]) 
            
            categorical_features[:,categorical_index] = tms.fit_transform(data)
            categorical_index += 1
                
        for feature_tms in self.numerical_features:
            feature, tms = feature_tms
            _features = tms.feature
            data = np.array([record[_features] for record in features]) 
            
            numerical_features[:,numerical_index] = tms.fit_transform(data)
            numerical_index += 1
                
        return numerical_features, categorical_features
    
    def transform(self, features:List[Dict[str, any]]) -> np.ndarray:
        dataset_len = len(features)
        
        numerical_features = np.zeros((dataset_len, len(self.numerical_features)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.categorical_features)), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for feature_tms in self.categorical_features:
            feature, tms = feature_tms
            _features = tms.feature
            data = np.array([record[_features] for record in features]) 
            
            categorical_features[:,categorical_index] = tms.transform(data)
            categorical_index += 1
                
        for feature_tms in self.numerical_features:
            feature, tms = feature_tms
            _features = tms.feature
            data = np.array([record[_features] for record in features]) 
            
            numerical_features[:,numerical_index] = tms.transform(data)
            numerical_index += 1

        return numerical_features, categorical_features
    
    def get_feature(self, feature:str) -> Tuple[str, TransformerMixin]:
        for feature_name, feature_tms in self.features_tms:
            if feature_name == feature:
                return feature_name, feature_tms
            
        raise KeyError(f"Feature {feature} doesn't exist")
        
    def get_embeddings_size(self, embedding_max_size:int=50) -> list:
        embedding_dims = []
        for feature_tms in self.categorical_features:
            _, tms = feature_tms

            q_unique_values = len(tms.encoder.classes_)
            embedding_size = min(q_unique_values//2, embedding_max_size)
            mapping = (q_unique_values, embedding_size)
            embedding_dims.append(mapping)
            
        return embedding_dims
    
    def get_features_quantity(self) -> tuple:
        return len(self.numerical_features), len(self.categorical_features)
        
    def dumps(self, filename:str):
        with open(filename, 'wb') as f:
            return pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(obj):
        return pickle.loads(obj)
    
    def __get_features(self, features:Dict[str, str], keys:Union[str, List[str]], primitive_fill_missing:str=np.nan) -> np.ndarray:
        if isinstance(keys, str):
            data = list(map(lambda record: record[keys], features))
            data = [item.strip() if item.strip() != '' else primitive_fill_missing for item in data]
            return np.array(data, dtype=object)