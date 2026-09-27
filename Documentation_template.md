# Documentation Template

Use this template only for a pipeline stage whose implementation is present in the repository.

## 1. Purpose

What the stage does and what it deliberately does not do.

## 2. Position in Pipeline

Describe the upstream and downstream stages.

## 3. Inputs

Document exact filenames, CLI arguments, formats, required columns, optional columns, and defaults.

## 4. Processing

Document only implemented transformations and algorithms.

For entity resolution, distinguish preprocessing, blocking/candidate generation, pairwise feature extraction, model scoring, and entity-level decision.

Do not describe an older or unused implementation as the production path.

## 5. Outputs

Document exact files, columns, schemas, ordering expectations, and empty-result behavior.

## 6. Correctness

Document missing-value behavior, duplicate handling, required-column validation, deterministic behavior, known hard-filter failure modes, and leakage controls.

## 7. Evaluation

Always distinguish candidate recall, pairwise metrics, and final entity-level macro F0.5.

Never use pairwise model metrics as a substitute for the final entity-level score.

Include the exact population and command used for each measured result.

## 8. Performance

Document dataset size, candidate limit, posting caps, chunk size, measured runtime, and throughput.

Do not claim a runtime that has not actually been measured.

## 9. Tests

List exact test files and commands.

## 10. Limitations

Document known recall risks, unsupported fields, missing data, and assumptions.

## 11. Handoff

State the exact contract the next stage receives.
