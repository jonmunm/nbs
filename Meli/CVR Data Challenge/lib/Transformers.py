import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
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
        
    def fit_transform(self, X):
        X = np.array(X[self.key], dtype=self.dtype) if self.key is not None else X
        
        for tfm in self.pre_tfms:
            X = np.array(tfm.transform(X))
        
        X = self.imputer.fit_transform(X.reshape(-1,1)).squeeze() if self.imputer is not None else X
        X = self.scaler.fit_transform(X.reshape(-1,1)).squeeze() if self.scaler is not None else X
        
        for tfm in self.post_tfms:
            X = tms.transform(X)
            
        return X
    
    def transform(self, X):     
        X = np.array(X[self.key], dtype=self.dtype) if self.key is not None else X
        
        for tfm in self.pre_tfms:
            X = np.array(tfm.transform(X))
            
        X = self.imputer.transform(X.reshape(-1,1)).squeeze() if self.imputer is not None else X
        X = self.scaler.transform(X.reshape(-1,1)).squeeze() if self.scaler is not None else X
        
        for tfm in self.post_tfms:
            X = tms.transform(X)
            
        return X

class CategoricalTransformer(TransformerMixin):
    def __init__(self, key=None, handle_unknown=False, pre_tfms=list(), encoder=None, post_tfms=list()):
        self.key = key
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
        
    def fit_transform(self, X, Y=None):
        X = np.array(X[self.key], dtype=np.str) if self.key is not None else X
        
        for tfm in self.pre_tfms:
            X = np.array(tfm.transform(X))
        
        X = X.astype(np.str)
        
        self.encoder.fit(np.append(X, ['UNKNOWN'] if self.handle_unknown else []))
        X = self.encoder.transform(X) if self.encoder is not None else X
        
        for tfm in self.post_tfms:
            X = tfm.transform(X)     
            
        return X
    
    def transform(self, X):      
        X = np.array(X[self.key], dtype=np.str) if self.key is not None else X
        
        for tfm in self.pre_tfms:
            X = np.array(tfm.transform(X))
            
        if self.handle_unknown:
            X = np.array([x if x in self.encoder.classes_ else 'UNKNOWN' for x in X])
            
        X = X.astype(np.str)
        if self.encoder is not None:
            X = self.encoder.transform(X)
        
        for tfm in self.post_tfms:
            X = tfm.transform(X)     
            
        return X
    
class CategoricalTargetTransformer(TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.encoder = LabelEncoder()
        
    def fit_transform(self, X):
        X = np.array([str(x).strip() for x in X], dtype=np.str)
        return self.encoder.fit_transform(X).reshape(-1,1)
    
    def transform(self, X):        
        X = np.array([str(x).strip() for x in X], dtype=np.str) 
        return self.encoder.transform(X).reshape(-1,1)

class DatasetTransformer():
    def __init__(self, X_tfms):
        self.X_numerical = []
        self.X_categorical = []
        
        numerical_index = 0
        categorical_index = 0
        
        for X_tfm in X_tfms:
            key, tfm = X_tfm
            entry = {'key' : key, 'tfm' : tfm}

            if isinstance(tfm, CategoricalTransformer):
                entry['index'] = categorical_index
                self.X_categorical.append(entry)
                categorical_index += 1
                
            if isinstance(tfm, NumericalTransformer):
                entry['index'] = numerical_index
                self.X_numerical.append(entry)
                numerical_index += 1
                
    def __reshape_X(self, X):
        X_transformed = {}
        keys = X[0].keys()
        for key in keys:
            X_transformed[key] = [x[key] for x in X]
            
        return X_transformed

    def fit_transform(self, X, Y=None):      
        dataset_len = len(X)
        X = self.__reshape_X(X)
        
        numerical_features = np.zeros((dataset_len, len(self.X_numerical)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.X_categorical)), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for tfm in [entry['tfm'] for entry in self.X_categorical]:
            categorical_features[:,categorical_index] = tfm.fit_transform(X)
            categorical_index += 1
               
        for tfm in [entry['tfm'] for entry in self.X_numerical]:
            numerical_features[:,numerical_index] = tfm.fit_transform(X)
            numerical_index += 1
               
        return numerical_features, categorical_features
    
    def transform(self, X, Y=None):
        dataset_len = len(X)
        X = self.__reshape_X(X)
        
        numerical_features = np.zeros((dataset_len, len(self.X_numerical)), dtype=np.float64)
        categorical_features = np.zeros((dataset_len, len(self.X_categorical)), dtype=np.int64)
        
        numerical_index = 0
        categorical_index = 0
        
        for tfm in [entry['tfm'] for entry in self.X_categorical]:
            categorical_features[:,categorical_index] = tfm.transform(X)
            categorical_index += 1
               
        for tfm in [entry['tfm'] for entry in self.X_numerical]:
            numerical_features[:,numerical_index] = tfm.transform(X)
            numerical_index += 1
               
        return numerical_features, categorical_features
        
    def get_embeddings_size(self, embedding_max_size=50):
        embedding_dims = []
        for tmf in [entry['tfm'] for entry in self.X_categorical]:
            q_unique_values = len(tmf.encoder.classes_)
            embedding_size = min(q_unique_values//2, embedding_max_size)
            mapping = (q_unique_values, embedding_size)
            embedding_dims.append(mapping)
            
        return embedding_dims
    
    def get_features_quantity(self):
        return len(self.X_numerical), len(self.X_categorical)
        
    def dumps(self, filename):
        with open(filename, 'wb') as f:
            return pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(obj):
        return pickle.loads(obj)
                
            