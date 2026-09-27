"""Lightweight pair features for laptop-scale entity resolution."""
from __future__ import annotations
import numpy as np
from rapidfuzz import fuzz
from ..preprocessing import extract_address_hints

FEATURE_NAMES_FAST = [
    "name_ratio","name_partial_ratio","name_token_sort_ratio","name_token_set_ratio",
    "address_ratio","address_token_set_ratio",
    "country_exact_match","name_exact_match","address_exact_match",
    "city_exact_match","postal_exact_match","house_number_exact_match","state_exact_match",
    "name_char_len_diff","address_char_len_diff",
]

def compute_fast_features_batch(n1_col,a1_col,c1_col,n2_col,a2_col,c2_col):
    n=len(n1_col)
    out=np.empty((n,len(FEATURE_NAMES_FAST)),dtype=np.float32)
    for i in range(n):
        n1=n1_col[i] or ""; a1=a1_col[i] or ""; c1=c1_col[i] or ""
        n2=n2_col[i] or ""; a2=a2_col[i] or ""; c2=c2_col[i] or ""
        h1=extract_address_hints(a1); h2=extract_address_hints(a2)
        out[i,0]=fuzz.ratio(n1,n2)/100
        out[i,1]=fuzz.partial_ratio(n1,n2)/100
        out[i,2]=fuzz.token_sort_ratio(n1,n2)/100
        out[i,3]=fuzz.token_set_ratio(n1,n2)/100
        out[i,4]=fuzz.ratio(a1,a2)/100
        out[i,5]=fuzz.token_set_ratio(a1,a2)/100
        out[i,6]=float(bool(c1 and c2 and c1==c2))
        out[i,7]=float(bool(n1 and n2 and n1==n2))
        out[i,8]=float(bool(a1 and a2 and a1==a2))
        out[i,9]=float(bool(h1["city"] and h2["city"] and h1["city"]==h2["city"]))
        out[i,10]=float(bool(h1["postal_code"] and h2["postal_code"] and h1["postal_code"]==h2["postal_code"]))
        out[i,11]=float(bool(h1["house_number"] and h2["house_number"] and h1["house_number"]==h2["house_number"]))
        out[i,12]=float(bool(h1["state"] and h2["state"] and h1["state"]==h2["state"]))
        out[i,13]=abs(len(n1)-len(n2))
        out[i,14]=abs(len(a1)-len(a2))
    return out
