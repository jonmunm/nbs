import csv
from random import Random
import os
import gzip
import shutil

def load_dataset(fname='train.csv', sample=False):
    """
    Use the `sample` parameter to get a small version of the dataset. 
    That is useful to make sure your code works before doing all the heavy lifting.
    
    This function assumes that there's a file called 'train.csv'. 
    You can download it from http://data-challenges.ml.com/static/data/cvr-estimation/train.csv.gz
    """
    
    if not os.path.isfile(fname):
        if os.path.isfile(f'{fname}.gz'):
            with gzip.open(f'{fname}.gz', 'rb') as f_in:
                with open(fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            

    with open(fname) as f:
        data = list(csv.DictReader(f, dialect=csv.excel))
        data = [row for row in data if row['item_id'] is not None]
        
    
    if sample: data = Random(42).sample(data, int(len(data) * 0.1))
        
    X_train = []
    X_test = []
    for row in data:
        if abs(hash(row['item_id'])) % 10 < 2: 
            X_test.append(row)
        else:
            X_train.append(row)

    y_train = [row.pop('conversion') == 'True' for row in X_train]
    y_test = [row.pop('conversion') == 'True' for row in X_test]
    
    return X_train, X_test, y_train, y_test