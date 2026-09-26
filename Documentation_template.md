# Documentation Template

Use this outline when documenting a pipeline stage. Replace every bracketed value with facts verified from the current implementation. State explicitly when a requested capability is not implemented. Do not report unmeasured data counts or scores.

## 1. Purpose

[What this stage does and what it does not do.]

## 2. Position in Pipeline

[Upstream and downstream stages, with a short flow diagram if useful.]

## 3. Inputs

[Exact files or API arguments, formats, required columns, optional columns, and path defaults.]

## 4. Processing

[Implemented transformations or algorithms, deterministic settings, and why each step exists.]

## 5. Outputs

[Exact file names or return values, schemas, retained/added/removed columns, and empty-result behavior.]

## 6. Data Quality and Leakage

[Validation, missing-value behavior, duplicate handling, label separation, and any limitations. Do not imply checks that are not present in code.]

## 7. Evaluation

[Metric definition, evaluation population, split strategy, measured results, and reproducible command. Label proxy metrics and unverified external scores clearly.]

## 8. Execution

[Working command lines using paths accepted by the current scripts. Note legacy or mismatched defaults.]

## 9. Tests and Validation

[Relevant test commands and the behaviors they cover.]

## 10. Limitations

[Known false-negative risks, unsupported input fields, unimplemented features, and required data that is not available.]

## 11. Handoff

[What the next stage receives and any schema or ordering contract it relies on.]
