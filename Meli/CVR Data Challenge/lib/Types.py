from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.base import TransformerMixin
from typing import List, Dict, Tuple

TrasformationItem = Tuple[str, TransformerMixin]
TransformationList = List[TrasformationItem]
FunctionTransformerList = List[FT]
StringList = List[str]