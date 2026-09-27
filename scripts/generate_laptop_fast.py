"""Generate full submission with the laptop-fast pipeline."""
import argparse, os, sys, pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","code")))
from business_entity_resolution.src.blocking.laptop_fast_inference import run_laptop_fast

p=argparse.ArgumentParser()
p.add_argument("--data-dir",default="dataset/test")
p.add_argument("--model-dir",default="models/laptop_fast")
p.add_argument("--output-dir",default="output_laptop_fast")
p.add_argument("--top-k",type=int,default=20)
p.add_argument("--chunk-size",type=int,default=5000)
a=p.parse_args()
s1=pd.read_csv(os.path.join(a.data_dir,"test_source1.tsv"),sep="\t")
s2=pd.read_csv(os.path.join(a.data_dir,"test_source2.tsv"),sep="\t")
s3=pd.read_csv(os.path.join(a.data_dir,"test_source3.tsv"),sep="\t")
print(f"Loaded {len(s1):,} S1 / {len(s2):,} S2 / {len(s3):,} S3")
run_laptop_fast(s1,s2,s3,a.model_dir,a.output_dir,a.top_k,a.chunk_size)
