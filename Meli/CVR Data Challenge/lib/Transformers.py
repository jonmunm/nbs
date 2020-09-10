import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from typing import List, Dict, Tuple
from .Types import *
import numpy as np

class NumericalTransformer(TransformerMixin):
    def __init__(
        self, 
        pre_tms:FunctionTransformerList=list(), 
        imputer:TransformerMixin=None, 
        scaler:TransformerMixin=None, 
        post_tms:FunctionTransformerList=list()
    ):
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
        pre_tms:FunctionTransformerList=list(),
        encoder:TransformerMixin=None,
        post_tms:FunctionTransformerList=list(),
        force_casting:bool=False
    ):
        self.pre_tms = pre_tms
        self.force_casting = force_casting

        if encoder != False:
            if encoder is None:
                self.encoder = LabelEncoder()
            else:
                self.encoder = encoder
        else:
            self.encoder = None
            
        self.post_tms = post_tms
        
    def fit_transform(self, data:np.ndarray) -> np.ndarray: 
        if self.force_casting:
            data = np.array([str(item) for item in data], dtype=object)
            
        for tms in self.pre_tms:
            data = tms.transform(data)
        
        data = self.encoder.fit_transform(data) if self.encoder is not None else data
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data
    
    def transform(self, data:np.ndarray) -> np.ndarray:
        if self.force_casting:
            data = np.array([str(item) for item in data], dtype=object)
        
        for tms in self.pre_tms:
            data = tms.transform(data)
            
        if self.encoder is not None:
            unique_values = np.unique(data)
            diff = np.setdiff1d(unique_values, self.encoder.classes_, True)
            self.encoder.classes_ = np.append(self.encoder.classes_, diff)
            
            data = self.encoder.transform(data)
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data 

class DatasetTransformer():
    def __init__(self, features_tms:TransformationList, target_tms:TrasformationItem=None):
        self.features_tms = features_tms
        self.numerical_features_names = []
        self.categorical_features_names = []
        
        for i, feature_tms in enumerate(self.features_tms):
            feature, tms = feature_tms

            if isinstance(tms, CategoricalTransformer):
                self.categorical_features_names.append(feature)
                
            if isinstance(tms, NumericalTransformer):
                self.numerical_features_names.append(feature)
                
        if target_tms is not None:
            target_name, _ = target_tms
            self.target_tms = target_tms
            self.target_name = target_name
        else:
            self.target_tms = None
            self.target_name = None

    def fit_transform(self, features:np.ndarray, target:np.ndarray=None):      
        dataset_len = len(features)
        
        numerical_features = np.zeros((dataset_len, len(self.numerical_features_names)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.categorical_features_names)), dtype=np.int64)
        the_target = np.zeros((dataset_len, 1), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for i, feature_tms in enumerate(self.features_tms):
            feature, tms = feature_tms
            print(feature)
            
            feature_transformed = tms.fit_transform(features[:,i].squeeze())
            
            if isinstance(tms, CategoricalTransformer):
                categorical_features[:,categorical_index] = feature_transformed
                categorical_index = categorical_index+1
                
            if isinstance(tms, NumericalTransformer):
                numerical_features[:,numerical_index] = feature_transformed
                numerical_index = numerical_index+1
                   
        if self.target_tms is not None:
            _, tms = self.target_tms
            target_transformed = tms.fit_transform(target.squeeze())
            the_target[:,0] = target_transformed

        return numerical_features, categorical_features, the_target if self.target_tms is not None else None 
    
    def transform(self, features:np.ndarray, target:np.ndarray=None):
        dataset_len = len(features)
        
        numerical_features = np.zeros((dataset_len, len(self.numerical_features_names)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.categorical_features_names)), dtype=np.int64)
        the_target = np.zeros((dataset_len, 1), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for i, feature_tms in enumerate(self.features_tms):
            feature, tms = feature_tms
            feature_transformed = tms.transform(features[:,i].squeeze())
            
            if isinstance(tms, CategoricalTransformer):
                categorical_features[:,categorical_index] = feature_transformed
                categorical_index = categorical_index+1
                
            if isinstance(tms, NumericalTransformer):
                numerical_features[:,numerical_index] = feature_transformed
                numerical_index = numerical_index+1
                   
        if target is not None:
            _, tms = self.target_tms
            target_transformed = tms.transform(target.squeeze())
            the_target[:,0] = target_transformed

        return numerical_features, categorical_features, the_target if self.target_tms is not None else None 
    
    def get_feature_transformation(self, feature:str) -> TrasformationItem:
        for feature_name, feature_tms in self.features_tms:
            if feature_name == feature:
                return feature_tms
            
        raise KeyError(f"Feature {feature} doesn't exist")
        
    def dumps(self, filename:str):
        with open(filename, 'wb') as f:
            return pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(obj):
        return pickle.loads(obj)