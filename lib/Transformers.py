from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from typing import List, Dict, Tuple
import numpy as np

TrasformationItem = Tuple[str, TransformerMixin]
TransformationList = List[TrasformationItem]
FunctionTransformerList = List[FT]

class NumericFeature(TransformerMixin):
    def __init__(
        self, 
        pre_tms:FunctionTransformerList=list(), 
        imputer:None=SimpleImputer(), 
        scaler:None=StandardScaler(), 
        post_tms:FunctionTransformerList=list()
    ):
        self.pre_tms = pre_tms
        self.imputer = imputer
        self.scaler = scaler
        self.post_tms = post_tms
        
    def fit_transform(self, data:list) -> np.ndarray:        
        for tms in self.pre_tms:
            data = tms.transform(data)
        
        data = self.imputer.fit_transform(np.array(data).reshape(-1,1)).squeeze() if self.imputer != False else data
        data = self.scaler.fit_transform(np.array(data).reshape(-1,1)).squeeze() if self.scaler != False else data
        
        for tms in self.post_tms:
            data = tms.transform(data)
            
        return data
    
    def transform(self, data:list) -> np.ndarray:       
        for tms in self.pre_tms:
            data = tms.transform(data)
            
        data = self.imputer.transform(np.array(data).reshape(-1,1)).squeeze() if self.imputer != False else data
        data = self.scaler.transform(np.array(data).reshape(-1,1)).squeeze() if self.scaler != False else data
        
        for tms in self.post_tms:
            data = tms.transform(data)        
        
        return data

class CategoricalFeature(TransformerMixin):
    def __init__(
        self, 
        pre_tms:FunctionTransformerList=list(),
        encoder:None=LabelEncoder(),
        post_tms:FunctionTransformerList=list(),
    ):
        self.pre_tms = pre_tms
        self.encoder = encoder
        self.post_tms = post_tms
        
    def fit_transform(self, data:list) -> np.ndarray:      
        for tms in self.pre_tms:
            data = tms.transform(data)
            
        data = self.encoder.fit_transform(data) if self.encoder != False else data
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data
    
    def transform(self, data:list) -> np.ndarray:      
        for tms in self.pre_tms:
            data = tms.transform(data)
            
        data = self.encoder.transform(data) if self.encoder != False else data
        
        for tms in self.post_tms:
            data = tms.transform(data)     
            
        return data 

class DatasetTransformer():
    def __init__(self, features_tms:TransformationList=list(), target_tms:TrasformationItem=tuple()):
        self.features_tms = features_tms
        self.numerical_features_names = []
        self.categorical_features_names = []
        
        for i, feature_tms in enumerate(self.features_tms):
            feature, tms = feature_tms

            if isinstance(tms, CategoricalFeature):
                self.categorical_features_names.append(feature)
                
            if isinstance(tms, NumericFeature):
                self.numerical_features_names.append(feature)
                
        if len(target_tms) > 0:
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
            feature_transformed = tms.fit_transform(features[:,i].squeeze())
            
            if isinstance(tms, CategoricalFeature):
                categorical_features[:,categorical_index] = feature_transformed
                categorical_index = categorical_index+1
                
            if isinstance(tms, NumericFeature):
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
            
            if isinstance(tms, CategoricalFeature):
                categorical_features[:,categorical_index] = feature_transformed
                categorical_index = categorical_index+1
                
            if isinstance(tms, NumericFeature):
                numerical_features[:,numerical_index] = feature_transformed
                numerical_index = numerical_index+1
                   
        if self.target_tms is not None:
            _, tms = self.target_tms
            target_transformed = tms.transform(target.squeeze())
            the_target[:,0] = target_transformed

        return numerical_features, categorical_features, the_target if self.target_tms is not None else None 