# Data governance

## Release scope

The project owner confirmed on 2026-09-13 that the required public-release rights were settled. The Stage 4 inference-only EMA weights are released under CC BY-NC-SA 4.0; project code is MIT-licensed. The SADA source is the pinned `MohamedRashad/SADA22` mirror at revision `094fe2c0fe4b549a4f34349e6e0622e7c7273c2d`. Preserve attribution and ShareAlike obligations.

## Handling rules

Keep `data/raw/` immutable and retain row-level provenance, source hashes, manifests, ledgers, and asset hashes. Do not publish raw dataset audio, private reference recordings, generated evaluations tied to a private reference, or full resumable checkpoints. Do not infer speaker identity or speaker-disjointness from the mirror: stable actor/show fields are unavailable. Native Saudi review remains necessary for transcript/audio alignment and dialect correctness; ASR QC is diagnostic only.

Public distribution is limited to the normalized Stage 4 EMA SafeTensors artifact and the supporting code/documentation. Inference requires user-supplied reference audio and a matching transcript. Users must have permission to use the reference voice and must not use the model for impersonation, deception, fraud, or misrepresentation. Supplemental datasets listed in research notes are not pipeline inputs and require their own licence review.

See [`docs/project_state.md`](project_state.md), [`reports/data_quality_report.md`](../reports/data_quality_report.md), [`reports/preprocessing_report.md`](../reports/preprocessing_report.md), and [`reports/validation_report.md`](../reports/validation_report.md).
