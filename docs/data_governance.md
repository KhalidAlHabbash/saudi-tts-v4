# Data governance

## Allowed scope

The prepared artifact is for private, local, non-commercial research only. The SADA source is the pinned `MohamedRashad/SADA22` mirror at revision `094fe2c0fe4b549a4f34349e6e0622e7c7273c2d`; project reports record CC BY-NC-SA 4.0 as the reported source term. Preserve attribution and ShareAlike obligations as applicable, but treat this record as provenance—not legal clearance.

## Handling rules

Keep `data/raw/` immutable and retain row-level provenance, source hashes, manifests, ledgers, and asset hashes. Do not publish raw audio or trained weights. Do not infer speaker identity, stable actor/show membership, consent, or speaker-disjointness from this mirror: those fields are unavailable. Native Saudi review is required for transcript/audio alignment and dialect correctness; ASR QC is diagnostic only.

Commercial use, public distribution, hosted inference, voice/identity cloning, or release of derived weights are out of scope pending rightsholder, counsel, consent, and attribution review. Supplemental datasets listed in research notes are not pipeline inputs and require their own licence review.

See [`docs/project_state.md`](project_state.md), [`reports/data_quality_report.md`](../reports/data_quality_report.md), [`reports/preprocessing_report.md`](../reports/preprocessing_report.md), and [`reports/validation_report.md`](../reports/validation_report.md).
