# Third-party notices and provenance

This file records verified provenance pointers for local research use. It does not grant rights or provide legal advice.

- **SADA / SADA22:** SDAIA and Saudi Broadcasting Authority source identified by the project reports; pinned convenience mirror: [MohamedRashad/SADA22](https://huggingface.co/datasets/MohamedRashad/SADA22), revision `094fe2c0fe4b549a4f34349e6e0622e7c7273c2d`. The reports record the source term as CC BY-NC-SA 4.0 and cite the [ICASSP 2024 paper](https://doi.org/10.1109/ICASSP48485.2024.10446243) and original [Kaggle distribution](https://www.kaggle.com/datasets/sdaiancai/sada2022). Verify current terms before any use beyond the private pilot.
- **SILMA TTS v1:** model assets from [silma-ai/silma-tts](https://huggingface.co/silma-ai/silma-tts), immutable revision `ac81834c5ce305504fe4c7602187042fdd6913db`; local file hashes are in `models/base/asset_manifest.json`.
- **F5-TTS:** runtime package version 1.1.7 and pinned source revision `c96c3aeed84f5e02aa54dc42c1193537ead39837`; see [F5-TTS](https://github.com/SWivid/F5-TTS) and the installed package metadata. The environment report records the locally attested wheel hash and MIT metadata; confirm upstream terms for redistribution.
- **Vocos:** local vocoder from [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz), revision `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`; local file hashes are recorded in the asset manifest.
- **Whisper diagnostic:** `openai/whisper-small`, revision `973afd24965f72e36ca33b3055d56a652f456b4d`, used only for bounded diagnostic QC; its cached hash and sample provenance are recorded in `reports/asr_quality_report.md` and `reports/asr_quality_sample.json`.

The repository does not include a complete third-party licence bundle. Keep source attribution and licence records with any private copy, and obtain separate rights review before sharing data, weights, or outputs.
