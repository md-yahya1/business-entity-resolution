import pandas as pd

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10)
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10)

needed_ids = set()
for m in gt['matched_entity_ids'].dropna():
    for x in str(m).split(','):
        if x.strip():
            needed_ids.add(x.strip())

s2_chunks = []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
    hit = chunk[chunk['entity_id'].isin(needed_ids)]
    if len(hit) > 0:
        s2_chunks.append(hit)
s2_matches = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()

s3_chunks = []
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
    hit = chunk[chunk['entity_id'].isin(needed_ids)]
    if len(hit) > 0:
        s3_chunks.append(hit)
s3_matches = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()

s2_lookup = {r.entity_id: r for r in s2_matches.itertuples()}
s3_lookup = {r.entity_id: r for r in s3_matches.itertuples()}

print("Sample ground truth matched entities:")
for r in gt.head(5).itertuples():
    s1_id = r.source1_entity_id
    matches = [m.strip() for m in str(r.matched_entity_ids).split(',') if m.strip()]
    s1_rows = s1[s1['entity_id'] == s1_id]
    if len(s1_rows) > 0:
        s1_r = s1_rows.iloc[0]
        print(f"S1: [{s1_id}] ({s1_r.country}) '{s1_r.business_name}' | '{s1_r.business_address}'")
        for m in matches:
            if m in s2_lookup:
                m_r = s2_lookup[m]
                print(f"  -> S2: [{m}] ({m_r.country}) '{m_r.business_name}' | '{m_r.business_address}'")
            elif m in s3_lookup:
                m_r = s3_lookup[m]
                print(f"  -> S3: [{m}] ({m_r.country}) '{m_r.business_name}' | '{m_r.business_address}'")
            else:
                print(f"  -> [{m}] not found")
        print("-" * 50)
