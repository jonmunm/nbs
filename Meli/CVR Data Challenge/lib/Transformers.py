import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from typing import List, Dict, Tuple, Union
import numpy as np

class NumericalTransformer(TransformerMixin):
    def __init__(self, key=None, handle_unknown=False, pre_tfms=list(), imputer=None, scaler=None, post_tfms=list(), dtype=np.int64):
        self.key = key
        self.pre_tfms = pre_tfms
        self.dtype = dtype
        
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
            
        self.post_tfms = post_tfms
        
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
    def __init__(self, key=None, handle_unknown=False, pre_tfms=list(), encoder=None, post_tfms=list()):
        self.key = key
        print(self.key)
        self.pre_tfms = pre_tfms
        self.handle_unknown = handle_unknown

        if encoder != False:
            if encoder is None:
                self.encoder = LabelEncoder()
            else:
                self.encoder = encoder
        else:
            self.encoder = None
            
        self.post_tfms = post_tfms
        
    def fit_transform(self, data:np.ndarray) -> np.ndarray:
        data = np.array([str(item).strip() for item in data], dtype=object)
        
        for tms in self.pre_tms:
            data = tms.transform(data).astype(np.str)
        
        self.encoder.fit(np.append(data, ['UNKNOWN'] if self.handle_unknown else []))
        data = self.encoder.transform(data) if self.encoder is not None else data
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data
    
    def transform(self, data:np.ndarray) -> np.ndarray:      
        data = np.array([str(item).strip() for item in data], dtype=object)
        
        for tms in self.pre_tms:
            data = tms.transform(data).astype(np.str)
            
        if self.handle_unknown:
            data = np.array([item if item in self.encoder.classes_ else 'UNKNOWN' for item in data], dtype=object)
            
        if self.encoder is not None:
            data = self.encoder.transform(data)
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data
    
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
    def __init__(self, X_tfms):
        self.X_numerical = []
        self.X_categorical = []
        
        for X_tfm in X_tfms:
            _, tfm = X_tfm

            if isinstance(tfm, CategoricalTransformer):
                self.X_categorical.append(tfm)
                
            if isinstance(tfm, NumericalTransformer):
                self.X_numerical.append(tfm)
                
    def __reshape_X(self, X):
        X_transformed = {}

        for tfm in [tfm for tfm in self.X_categorical if tfm.key is not None]:
            X_transformed[tfm.key] = [x[tfm.key] for x in X]
        
        for tfm in [tfm for tfm in self.X_numerical if tfm.key is not None]:
            X_transformed[tfm.key] = [x[tfm.key] for x in X]
            
        return X_transformed

    def fit_transform(self, X, Y=None):      
        dataset_len = len(X)
        X = self.__reshape_X(X)
        
        numerical_features = np.zeros((dataset_len, len(self.X_numerical)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.X_categorical)), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for feature, tms in self.X_categorical:
            _features = tms.feature
            data = np.array([record[_features] for record in features]) 
            
            categorical_features[:,categorical_index] = tms.fit_transform(data)
            categorical_index += 1
                
        for feature, tms in self.X_numerical:
            _features = tms.feature
            
            if isinstance(_features, str):
                data = np.array([record[_features] for record in features])
            else:
                data_list = []
                for __feature in _features:
                    data_list.append(np.array([record[__feature] for record in features]))
                    
                data = np.vstack(tuple(data_list))
            
            numerical_features[:,numerical_index] = tms.fit_transform(data)
            numerical_index += 1
                
        return numerical_features, categorical_features
    
    def transform(self, features:List[Dict[str, any]]) -> np.ndarray:
        dataset_len = len(features)
        
        numerical_features = np.zeros((dataset_len, len(self.numerical_features)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.categorical_features)), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for feature, tms in self.categorical_features:
            _features = tms.feature
            data = np.array([record[_features] for record in features]) 
            
            categorical_features[:,categorical_index] = tms.transform(data)
            categorical_index += 1
                
        for feature, tms in self.numerical_features:
            _features = tms.feature
            
            if isinstance(_features, str):
                data = np.array([record[_features] for record in features])
            else:
                data_list = []
                for __feature in _features:
                    data_list.append(np.array([record[__feature] for record in features]))
                    
                data = np.vstack(tuple(data_list))
            
            numerical_features[:,numerical_index] = tms.transform(data)
            numerical_index += 1

        return numerical_features, categorical_features
    
    def get(self, feature:str) -> Tuple[int, str, TransformerMixin]:
        for i, feature_tms in enumerate(self.categorical_features):
            _feature, tms = feature_tms
            if _feature == feature:
                return i, feature, tms
            
        for i, feature_tms in enumerate(self.numerical_features):
            _feature, tms = feature_tms
            if _feature == feature:
                return i, feature, tms
            
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
    
    @staticmethod
    def delete_features(feature_type:str, features_to_delete:Union[str, List[str]], features:np.ndarray) -> np.ndarray:
        if dataset_type not in ['numerical', 'categorical']:
            raise KeyError(f"FeatureType {feature_type} not allowed")
            
        features_to_delete = [features_to_delete] if isinstance(features_to_delete, str) else features_to_delete
        index_to_delete = []
            
        for i, feature_tms in enumerate(getattr(self, f'{feature_type}_features')):
            feature,_ = feature_tms
            
            if feature in features_to_delete:
                index_to_delete.append(i)
                
        return np.delete(features, index_to_delete, 1)
                
            