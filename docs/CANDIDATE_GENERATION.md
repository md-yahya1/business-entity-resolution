# Candidate Generation

## Purpose

Candidate generation is the recall ceiling of the final entity-resolution pipeline. It narrows the 10M+ Source-2/Source-3 records to a small deterministic candidate set for each Source-1 record before the trained classifier runs.

The production implementation is:

`code/business_entity_resolution/src/blocking/laptop_final.py`

The final submission CLI is:

`scripts/generate_submission_laptop_final.py`

## Final inference blocking

The blocker builds compact sorted uint64 posting indexes over Source 2 + Source 3. Each index is scoped by normalized country.

The active retrieval passes are:

1. Exact normalized business name.
2. First three characters of normalized business name.
3. First selective non-generic business-name token.
4. Exact postal code.
5. Exact house number.
6. Exact city hint.
7. First selective non-generic address token.

The block results are unioned and deduplicated. If more than `retrieval-limit` candidates remain, a cheap deterministic score ranks them before the ML stage.

Ranking signals are, in order of importance:

- exact normalized business name
- exact normalized address
- exact postal code
- exact house number
- exact city
- exact selective address token
- exact state
- same name prefix
- name/address length differences

No RapidFuzz similarity is calculated during blocking. This is intentional: fuzzy scoring at this stage previously reduced throughput too heavily for the laptop-scale run.

## Why multiple blocks are used

Blocking is a hard filter: a true pair that is never retrieved cannot be recovered by the classifier. Multiple overlapping blocking rules therefore provide different failure paths while keeping the expensive fuzzy features limited to the final shortlist.

The address-token pass is particularly useful when a business name changes substantially but location text remains similar. Exact address is also given a strong cheap ranking weight so an address-supported candidate is not displaced by a weak name-prefix candidate.

## Limits

Default production settings:

- `posting-cap=64`
- `exact-cap=32`
- `retrieval-limit=10`
- `output-candidate-limit=10`

These limits are deliberately small for laptop-scale inference. Increasing them changes both runtime and the downstream candidate population and must be benchmarked before a full run.

## Address hints

`extract_address_hints()` extracts:

- house number
- postal code
- state
- city

Postal codes are removed before state detection. This fixes a previous case where an address ending in `<state> <ZIP>` could fail to expose the state because the ZIP was the final token.

Country values are canonicalized for common aliases such as:

- `US`, `USA`, `United States` -> `us`
- `IN`, `India` -> `india`
- `FR`, `France` -> `france`
- `GB`, `UK`, `United Kingdom` -> `uk`

## Candidate recall evaluation

Candidate recall must be measured independently of classifier precision/recall.

For a Source-1 entity:

`candidate recall = true matches present in candidates / total true matches`

The previous 5,000-entity diagnostic, before the current blocker fixes, measured:

- candidate recall: 32.9110%
- final precision: 48.0508%
- final recall: 28.9656%
- macro F0.5: 39.8709%

Those are historical baseline measurements, not results for the current branch. The corrected pipeline must be re-benchmarked before the full 1.7M test run.

## Tests

The final blocker is covered by `tests/test_blocking.py`, including:

- recovery through an address token when the name is different
- preference for an exact-address candidate over a name-prefix decoy

Run:

```powershell
pytest tests/test_blocking.py tests/test_preprocessing.py -q
```

Then run the full suite:

```powershell
pytest tests/ -q
```
