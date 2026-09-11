# Saudi Arabic / Saudi-accent TTS data research

Research date: 2026-09-11.  “Saudi” below means the source explicitly identifies Saudi regional dialects, Saudi nationality, or a Saudi accent; it is not inferred merely because Arabic is spoken in the Gulf.  Counts and availability are source-reported and must be re-checked against the downloaded release manifest before training.

## Executive ranking

| Rank | Dataset | Why it matters | TTS disposition |
|---:|---|---|---|
| 1 | SADA (Saudi Audio Dataset for Arabic) | The only presently public, large, directly Saudi-labelled source found. | Research-only candidate after stringent clip/speaker/rights filtering; not a clean single-speaker TTS corpus. |
| 2 | KSU Arabic Speech Database / LDC2014S02 | Many explicitly Saudi speakers, controlled prompts plus spontaneous speech, rich channels/environments. | Strong Saudi-accent MSA supplement if an eligible organization licenses it; research-only. |
| 3 | L2-KSU / LDC2024S11 | Recent, 16-kHz read MSA, metadata enables selection of native Saudi speakers. | Small, controlled supplement only; separate Saudi speakers and avoid L2/error-labelled material. |
| 4 | SaudiTalk | Direct Hijazi, Ha'il, and Southern labels with human-verified transcripts. | Pilot/evaluation material only until licensing and source-video rights are cleared. |
| 5 | SAAVB (Saudi Accented Arabic Voice Bank) | Best documented nationwide Saudi-accent MSA collection (1,033 speakers). | High-value acquisition lead, but not an open download; contract is required. |

## CORE SAUDI DATA

### 1. SADA — Saudi Audio Dataset for Arabic (SDAIA + Saudi Broadcasting Authority)

**Sources / access.** The creators' [ICASSP 2024 SADA paper](https://doi.org/10.1109/ICASSP48485.2024.10446243) identifies SDAIA and the Saudi Broadcasting Authority (SBA) as publishers and points to the original [Kaggle `sdaiancai/sada2022`](https://www.kaggle.com/datasets/sdaiancai/sada2022) distribution. The currently visible public [Hugging Face mirror/card](https://huggingface.co/datasets/MohamedRashad/SADA22) has 253,166 rows and 50.4 GB; another mirror reports 77.1 GB. Treat mirrors as convenience copies, not authoritative releases, and record the Kaggle version / file checksums acquired. Card and paper cite **CC BY-NC-SA 4.0**: attribution, non-commercial use, and ShareAlike apply. The card was checked 2026-09-11; its authors say it is community-written, so the paper is the primary authority for methodology.

**Coverage and contents.** SADA has 667.7 actual audio hours from 57+ Saudi TV shows in 11 genres, chiefly Saudi dialects—**Najdi, Hijazi, and Gulf/Khaleeji**—with some non-Saudi Arabic labels (e.g., Egyptian/Levantine). The paper reports 435 h of high-quality labelled audio and 233 h unlabelled; split accounting reports 417.6 h transcribed train, 9.2 h dev, 10.8 h test. Audio is mono **16-kHz, 16-bit PCM WAV**. Metadata/columns exposed by the mirror include raw and cleaned text, age group, gender, dialect, source/show/genre, environment, file/segment times and a local speaker tag. Its sample rows demonstrate useful fields: `speaker_gender`, `speaker_dialect`, `speaker_age`, environment (`Clean`, `Music`, etc.), verbatim and normalized transcript.

**Speaker, annotation, and conditions.** This is broadcast drama/TV rather than prompted studio TTS. Each annotation unit is marked `Speaker1..7`, other/multiple/unclear; the paper describes this as speaker tracking within a split source segment, so do **not** assume `Speaker1` is a corpus-global identity. It provides gender (male/female), age (child/young adult/adult/elderly), dialect (including Najdi/Hijazi/Gulf), and acoustic condition (clean/music/laughter/advert/noisy). More than 150 annotators performed two sequential correction passes over automatic ASR/diarization, then 10–20% senior sampling and 1% independent correction; estimated average annotator WER was under 5%. Transcription is verbatim (fillers, repetitions and dialect phonetic spellings retained), with raw text and a normalized form. Remaining risks include dramatic acting, music, character/scene changes, overlap and a non-global speaker label.

**TTS suitability.** Best current scale for Saudi pronunciation/prosody adaptation, especially after retaining only single-speaker, clean, 3–12 s clips with a chosen dialect/gender and manually auditing transcript alignment. It is unsuitable as-is for a consistent named voice or commercial model; use only under the NC licence and isolate show/character leakage. The paper explicitly lists TTS as a potential task, but its original baselines are ASR—not a guarantee of TTS-grade pairs. Citation: [Alharbi et al., ICASSP 2024](https://doi.org/10.1109/ICASSP48485.2024.10446243).

### 2. King Saud University Arabic Speech Database — LDC2014S02

**Sources / access / licence.** The authoritative [LDC catalogue entry](https://catalog.ldc.upenn.edu/LDC2014S02) offers web download to LDC members/non-members under a signed [KSU agreement](https://catalog.ldc.upenn.edu/license/ksu-arabic-speech-database.pdf), not open redistribution. It permits only linguistic education and research and prohibits commercial exploitation/republication. KSU’s faculty page independently identifies the same LDC release. Checked 2026-09-11.

**Coverage and technical details.** 590 h from **269** male/female speakers; the release includes both Saudis and non-Saudis (the public catalogue does not provide the final Saudi-only count). It has prompted words, sentences, paragraphs and Q&A spontaneous responses. Each person was recorded in a soundproof room, office and cafeteria, via multiple professional/medium microphones and a mobile phone, in three sessions about six weeks apart (roughly 16–19 min/person). Audio is two-channel **48-kHz 16-bit FLAC-compressed PCM WAV**. The catalogue says recordings were verified for missing/problem recordings. The design paper documents Saudi female/male counts during sessions and that speaker nationality/gender were collected; confirm the exact shipped metadata fields and stable speaker IDs after licensing rather than assuming filenames are sufficient.

**Dialect / transcript / TTS assessment.** It is Arabic speech by a selectable Saudi-national subset, including accent-differentiating prompts, but the public documentation does **not** label Najdi/Hijazi by speaker or state a uniform dialect. The texts are designed prompts; not every utterance has an independently verbatim transcript in the public catalogue. This is a valuable controlled **Saudi-accent MSA** and robustness supplement, not a direct Najdi/Hijazi corpus. For TTS, use only the Saudi subset, a single clean channel/environment, and speaker-disjoint splits; do not mix cafeteria/mobile clips into a quality voice set. Its research-only agreement makes product training/deployment out of scope without new permission.

### 3. L2-KSU Native and Non-Native Arabic Speech — LDC2024S11 (and public IWAN package)

**Sources / access / licence.** The official [LDC2024S11 catalogue](https://catalog.ldc.upenn.edu/LDC2024S11) is web download after the LDC agreement/fee process. A public [IWAN Hugging Face package](https://huggingface.co/datasets/IWAN/L2-KSU-Dataset) identifies **CC BY-NC-ND 4.0** and its own archive layout. Do not treat that mirror's terms as a substitute for a licence from the KSU/LDC rights holders; **NoDerivatives and non-commercial** require legal confirmation before model training or release. Checked 2026-09-11.

**Coverage and technical details.** Official release: approximately 6 h / 6 min, **80 subjects** (40 native, 40 non-native), recorded in 2022; native participants are from Saudi Arabia, Egypt and Palestine, so filter using speaker metadata rather than treating all 40 as Saudi. They repeatedly read ten MSA sentences. It provides **16-kHz, 16-bit WAV/PCM**, UTF-8 transcripts, speaker metadata, Arabic sentences plus transliteration, English translation and IPA. The associated peer-reviewed [dataset description](https://link.springer.com/article/10.1007/s10791-024-09489-8) reports 4,086 audio files / 4,077 utterances and equal male/female composition within native and non-native groups, but does not publish the Saudi-only count or its sex balance.

The public package documents 4,094 WAVs, 4,098 Praat TextGrids and 4,174 `.txt` files, with stable path-like `speakerN` IDs and male/female/native/non-native directory partitions; it also has 818 phone-level error annotations. Those small count differences are a revision signal: pin the chosen source and build a manifest rather than mixing releases.

**Dialect / quality / TTS assessment.** The target is read **MSA**, not labelled Najdi/Hijazi; recordings came via a crowdsourcing platform for natives and on-site KSU for L2 participants, so conditions differ. The phone-level annotations and canonical/actual pronunciation are strong for pronunciation research, but the L2 material deliberately contains errors. For TTS, retain only confirmed native-Saudi speakers with clean files and canonical text, preserve `speakerN` isolation, and use it as a small accent/phonetic supplement—not a voice identity dataset.

### 4. SaudiTalk — multi-source Saudi social-media speech

**Source / access.** Public at [SaudiTalk/SaudiTalk on Hugging Face](https://huggingface.co/datasets/SaudiTalk/SaudiTalk), loaded through `datasets` or repository files; card checked 2026-09-11. The card contains no YAML licence and expressly traces content to TikTok, YouTube and Snapchat. Therefore its public repository status is **not** a demonstrated licence to train, redistribute audio, or ship a model; obtain creator/platform/rightsholder clearance first.

**Contents.** 133 WAV samples, 3.86 GB, labelled **Hijazi, Ha'il, Southern**. `dataset.csv` includes `id`, source-video link/title, `speaker_name` (or channel), dialect, content type, speaker type, duration, source, human-verified transcript, audio relative path and filename. Thus it has sample IDs and a speaker/channel string but no documented immutable, de-identified corpus speaker ID; it does not publish gender/age, total duration, sample rate, channel count, or per-dialect counts. Recording conditions are intentionally diverse/uncontrolled social media.

**TTS assessment.** Direct regional labels and human-verified transcripts make it useful for a tiny Saudi dialect evaluation set or manual audit seed. The size, provenance/rights uncertainty, unknown technical profile and variable audio make it inappropriate as a main TTS fine-tune source. Never infer demographic labels from names or video appearances.

### 5. SAAVB — Saudi Accented Arabic Voice Bank (KACST)

**Source / access.** The primary [ISCA Archive paper](https://www.isca-archive.org/exling_2008/alghamdi08_exling.pdf) says SAAVB is available only when a contract with KACST is signed; it had also been licensed to IBM. No current public data download or published open-data licence was found as of 2026-09-11. Contact KACST/rightsholders rather than using third-party claims.

**Contents and quality.** This is a particularly strong Saudi-accent MSA acquisition target: **1,033** speakers drawn from all Saudi cities (523 male, 510 female), 96.37 h, 60,947 audio/text files (59 per speaker; mean 5.7 s), 302,107 transcribed words. File naming encodes region, city, gender, age, telephone type and calling environment, providing practical speaker/condition metadata. Prompts are 49 read MSA items plus 10 elicited spontaneous responses; every speaker had a unique prompt sheet except two common dialect-variation sentences. Ages are 16–30 (512), 31–45 (364), and 46–60 (147); 725 cellular recordings span quiet/noisy/moving-vehicle settings, and 308 landline recordings span quiet/noisy conditions. Internal KACST and external IBM Cairo verification is documented.

**TTS assessment.** Its nationwide Saudi accent coverage, sex balance, short utterances and transcript alignment are excellent for controlled accent research. Telephone/channel variation and MSA—not explicit Najdi/Hijazi—limit natural regional TTS. It is **not actionable** until a contract explicitly permits the intended training, model artefact, inference and commercial/non-commercial use.

## SUPPLEMENTAL DATA (Gulf, not Saudi; never use to claim a Saudi voice)

### A. UAE Arabic Speech Recognition Corpus (Mobile) — ELRA-S0228

The [ELRA catalogue](https://catalogue.elda.org/en-us/repository/browse/ELRA-S0228_130/) describes 168 screened UAE speakers (94 male, 74 female), two channels, quiet home/office recording, with news/daily-dialogue scripts. It is a paid/licensed ELRA resource; exact hours, audio encoding/sample rate, transcript format and speaker-ID schema need confirmation from the delivery documentation. It is controlled Emirati/Gulf Arabic **supplemental acoustic material**, not Saudi dialect evidence; use only if its licence covers model training and only after a dialect-separation experiment.

### B. Casablanca (Emirati subset)

The [Casablanca paper and project](https://www.dlnlp.ai/speech/casablanca) describe 48 h across eight country-level dialects, including Emirati. It provides human transcriptions, country/dialect, gender and code-switch labels; each accepted segment is intended to be one clear speaker, 3–30 s. The Emirati share/hours and global speaker IDs are not reported in the public summary. Owing to YouTube copyright, the release provides URLs, timestamps and annotations—not source audio—and the authors say only a subset is public. Its annotation procedure (two native annotators per dialect, guidelines and review) is credible, but it remains ASR-oriented broadcast material and has availability/copyright drift. Use only as a reproducible evaluation/research supplement, not a distributable TTS training corpus.

### C. Ramsa (Emirati; research watchlist, not an acquisition path yet)

[Ramsa's 2026 preprint](https://arxiv.org/abs/2603.08125) reports a developing 41-h Emirati corpus with 157 speakers (59 female/98 male), interviews and national television, Urban/Bedouin/Mountain-Shihhi labels, and ASR/TTS baseline work. No public download, licence, audio specification, transcript release, speaker-ID policy or access mechanism was located. It is closely related Gulf data only and **not usable now**.

## REJECTED DATASETS / RELEASES

| Dataset/release | Rejection reason for Saudi TTS fine-tuning |
|---|---|
| Saudi ASWAT / KSAA Aswat Corpus | The [LREC 2026 paper](https://aclanthology.org/2026.lrec-1.124/) is scientifically promising: target 2,500 h of natural Saudi conversations across Najdi, Eastern, Hijazi, Northern and Southern varieties, 55+ local varieties, balanced age/gender and structured metadata. The [official KSAA initiative page](https://ksaa.gov.sa/en/initiatives/50869-Aswat-Corpus) confirms multi-region audio/TEI/CODA-oriented work. Neither source supplied a downloadable corpus, licence, released hours/utterance count, audio spec or access procedure. Treat as a watchlist, not data. |
| Haneen Corpus (Medina dialect) | The [2024 peer-reviewed corpus paper](https://doi.org/10.1016/j.jksuci.2023.101864) reports 70,364 Medina-dialect tokens and a 64-phoneme dictionary, but no data repository, licence, audio count/hours, sample rate, speaker demographics/IDs, transcript release, or request path was found. The paper is evidence of a Saudi dialect corpus—not an accessible training set. |
| Aswat (ArabicNLP 2023) | [Aswat](https://aclanthology.org/2023.arabicnlp-1.10/) reports 732 h of clean, multi-genre Arabic for self-supervised ASR pretraining, but the public paper does not establish a Saudi-national/dialect-labelled subset, audio/transcript release mechanism, licence, speaker metadata or TTS-aligned supervision. Do not confuse it with KSAA's similarly named Saudi initiative. |
| SADA-derived “TTS” mirrors as independent datasets | Examples such as [mohamedalgmaal/sada2022-arabic-tts](https://huggingface.co/datasets/mohamedalgmaal/sada2022-arabic-tts) are repackagings of SADA, not new speakers. Their file sizes differ from other mirrors; avoid double-counting, silently changed segmentation and unclear redistribution provenance. Use a single pinned official/cleared SADA release. |
| Synthetic Saudi/Najdi audio datasets | Example: [Rabe3/saudi-tts-higgs-synthetic-200k](https://huggingface.co/datasets/Rabe3/saudi-tts-higgs-synthetic-200k) states its audio is synthetic. It cannot establish native speaker acoustics/prosody and risks self-training artefacts; reject as source speech data. |
| Common Voice Arabic, FLEURS Arabic, MGB-2/QASR and generic Arabic collections | They may help Arabic pretraining/ASR, but available documentation does not give Saudi nationality/dialect labels sufficient to isolate native Saudi/Najdi/Hijazi speech. QASR and MGB-2 also use lightly supervised broadcast transcription. They are not evidence for a Saudi voice and are outside this Saudi fine-tune dataset set. |
| Arabic Speech Corpus / Egyptian, Levantine, Moroccan TTS corpora | Arabic alone is not Saudi. Regional linguistic mismatch makes them unsuitable for a Saudi-accent/dialect fine-tune except possibly broad base-model pretraining, which is outside this dataset decision. |

## Decisions and acquisition gates

1. **Start due diligence with SADA only** for a non-commercial research pilot: pin the original release, preserve original metadata, select a single target dialect, require single-speaker/clean/no-music clips, and have native Saudi reviewers audit audio-text alignment and dialect. Do not collapse SADA's local `Speaker1..7` label into a global speaker ID.
2. **For controlled Saudi-accent MSA, pursue LDC2014S02 first**, and LDC2024S11 second. Before procurement, obtain the catalogue documentation/sample manifest and confirm that nationality, sex and speaker IDs can select the Saudi subset; confirm intended model use under their research-only agreements.
3. **For explicitly regional Saudi dialects, use SaudiTalk only after clearance** and as a small hand-reviewed evaluation/pilot source; no Najdi-labelled open, well-licensed, studio-quality corpus was verified.
4. **Open a rights/access enquiry for SAAVB and Saudi ASWAT/KSAA Aswat.** They may materially improve coverage, but neither may be assumed available or licensed based on papers alone.
5. Preserve per-source licence/provenance in every eventual manifest. Do not publish source audio, inferred identities, or model weights trained on NC/research-only material without a written rights review.

## Sources and revision notes

Primary/authoritative sources prioritised: SDAIA/SBA's SADA paper; LDC catalogues and license PDFs; KACST/ISCA SAAVB paper; KSAA’s official Aswat page; ACL Anthology for Saudi ASWAT; ELRA for UAE corpus; original Casablanca paper/project. Hugging Face and GitHub records were used only to inspect current distribution claims and gaps. All URLs were accessed or search-index cross-checked on **2026-09-11**. No full dataset was downloaded, no audio was inspected, and no licence legality beyond the published terms is asserted here.
