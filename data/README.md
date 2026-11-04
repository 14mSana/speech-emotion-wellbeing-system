# Dataset storage

Do not commit dataset audio or generated features to this repository.

- Put original WAV files in `data/raw/`.
- Put cleaned manifests and derived feature tables in `data/processed/`.
- Keep both directories local; they are ignored by Git.
- Retain speaker identifiers in the dataset manifest. Future evaluation must split by speaker so no speaker appears in more than one of the train, validation, or test partitions.
- Store any dataset license, source URL, label-mapping notes, and preprocessing decisions in a small tracked documentation file, not inside raw data folders.

Supported baseline labels are `neutral`, `calm`, `happy`, `sad`, and `angry`.
