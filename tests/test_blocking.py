import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from business_entity_resolution.src.blocking.inference_blocking import generate_inference_candidates


def test_inference_blocking_recovers_match_from_address_tokens():
	s1 = pd.DataFrame([{
		"entity_id": "S1-1",
		"business_name": "Acme Holdings",
		"business_address": "15 Harbor Street, New York",
		"country": "US",
	}])
	s2 = pd.DataFrame([
		{
			"entity_id": "S2-decoy",
			"business_name": "Acme Other",
			"business_address": "99 Far Road, New York",
			"country": "US",
		},
		{
			"entity_id": "S2-match",
			"business_name": "Harbor Market",
			"business_address": "15 Harbor Street, New York",
			"country": "US",
		},
	])
	s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

	candidates = generate_inference_candidates(s1, s2, s3, top_k=1)

	assert candidates.loc[0, "candidate_entity_ids"] == ["S2-match"]
