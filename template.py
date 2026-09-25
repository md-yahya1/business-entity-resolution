from pathlib import Path

ROOT = Path("business-entity-resolution")

directories = [
    "dataset/train",
    "dataset/test",
    "output",
    "code/business_entity_resolution/src/preprocessing",
    "code/business_entity_resolution/src/blocking",
    "code/business_entity_resolution/src/features",
    "code/business_entity_resolution/src/models",
    "code/business_entity_resolution/src/evaluation",
    "code/business_entity_resolution/src/output",
    "notebooks",
    "tests",
]

files = [
    "README.md",
    "Documentation_template.md",
    "requirements.txt",
    ".gitignore",

    "code/business_entity_resolution/README.md",
    "code/business_entity_resolution/requirements.txt",

    "code/business_entity_resolution/src/__init__.py",
    "code/business_entity_resolution/src/config.py",
    "code/business_entity_resolution/src/pipeline.py",

    "code/business_entity_resolution/src/preprocessing/__init__.py",
    "code/business_entity_resolution/src/preprocessing/loader.py",
    "code/business_entity_resolution/src/preprocessing/normalization.py",

    "code/business_entity_resolution/src/blocking/__init__.py",
    "code/business_entity_resolution/src/blocking/name_blocking.py",
    "code/business_entity_resolution/src/blocking/address_blocking.py",
    "code/business_entity_resolution/src/blocking/candidate_generator.py",

    "code/business_entity_resolution/src/features/__init__.py",
    "code/business_entity_resolution/src/features/pair_features.py",

    "code/business_entity_resolution/src/models/__init__.py",
    "code/business_entity_resolution/src/models/train.py",
    "code/business_entity_resolution/src/models/predict.py",

    "code/business_entity_resolution/src/evaluation/__init__.py",
    "code/business_entity_resolution/src/evaluation/metrics.py",
    "code/business_entity_resolution/src/evaluation/validation.py",

    "code/business_entity_resolution/src/output/__init__.py",
    "code/business_entity_resolution/src/output/writer.py",

    "tests/test_preprocessing.py",
    "tests/test_blocking.py",
    "tests/test_features.py",
    "tests/test_pipeline.py",

    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
]

for directory in directories:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

for file in files:
    path = ROOT / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

print("Project structure created successfully!")