# Sinner–Alcaraz on Bluesky: Social Network & Sentiment Analysis

A reproducible NLP and social-network-analysis pipeline that characterises the online
discourse around the tennis rivalry between **Jannik Sinner** and **Carlos Alcaraz** on
the [Bluesky](https://bsky.app) platform during the **2025 US Open** (21 August – 10
September 2025).

The pipeline crawls posts from the Bluesky `atproto` API, enriches them with sentiment,
emotion and stance signals, builds reply/mention interaction graphs, detects communities,
and renders 20+ visualisations covering network structure, emotional tone, and fanbase
polarisation.

## Key findings

Computed on the crawled dataset of **9,078 posts**:

- **Sparse, modular network.** The interaction graph has density ≈ 0.002 with very high
  community modularity (Louvain ≈ 0.97, Infomap ≈ 0.94) — fans cluster tightly into
  fanbase-aligned groups. The giant connected component covers ~16% of users.
- **Positive emotional tone.** Dominant emotions are positive (trust, joy, anticipation);
  negative emotions (anger, fear, disgust) are comparatively rare.
- **Sentiment tracks match outcomes.** The sentiment time series reacts to key matches,
  with each fanbase diverging after wins and losses (notably the final).
- **Sentiment-aware stance beats frequency baselines.** Fusing RoBERTa sentiment with
  mention frequency reclassifies users by their stance (Sinner/Alcaraz/Neutral) against a simple raw count of mentions-.

> A full write-up with methodology and discussion is in [`report/report.tex`](report/report.tex).

## Pipeline architecture


```
[1] crawl       src/crawler.py (run separately)  ->  data/sinner_alcaraz_posts.csv
[2] preprocess  raw posts                         ->  data/sinner_alcaraz_processed.csv (FROZEN)
[3] network     processed CSV                      ->  network_*.csv, plots/network_*.png
[4] sentiment   processed + network                ->  plots/sentiment_*, community_emotion_*
[5] stance      processed + network                ->  plots/stance_*, fanbase_*
```

| Stage | Module | What it does |
|-------|--------|--------------|
| **Crawl** | `src/crawler.py` | Day-by-day search of the Bluesky `atproto` API with rate-limit handling and URI deduplication. |
| **Preprocess** | `src/preprocessing.py` | Text cleaning/lemmatisation (spaCy), RoBERTa sentiment, dual-backend emotion scoring, NER + entity linking, post- and user-level stance. |
| **Network** | `src/social_network_analysis.py` | Builds undirected (`Gu`) and directed (`Gd`) interaction graphs; computes centralities; runs Louvain + Infomap; renders network plots. |
| **Sentiment** | `src/social_sentiment_analysis.py` | Community emotion profiles, sentiment distribution, sentiment-over-time, emotion-backend comparison. |
| **Stance** | `src/social_stance_analysis.py` | User/community stance aggregation, polarisation metrics, fanbase study, diagnostics. |
| **Shared** | `src/utils.py` | Constants (player keywords/URIs, NRC emotions, US Open events) and shared helpers (RoBERTa loader, plotting). |

## Methodology

**Data collection.** Two keyword queries (Sinner-oriented and Alcaraz-oriented) are issued
per 24-hour window across the tournament, each capped at 5,000 posts/day to respect API
limits. Results are merged and deduplicated by post URI.

**Sentiment & emotion.**
- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` → a `sentiment_category`
  (positive/neutral/negative) and a continuous `sentiment_compound` = P(pos) − P(neg).
- **Emotion (two backends):** the **NRC lexicon** (NRCLex) and **GoEmotions**
  (`SamLowe/roberta-base-go_emotions`), whose 28 fine-grained labels are collapsed onto the
  8 Plutchik/NRC categories for side-by-side comparison between Transformers architecture and a lexicon algorithm.

**Stance.** A stance detection algorithm was implemented: Single-mention posts assign the post's compound score to that player; dual-mention posts are split via sentence-level sentiment. User-level net stance fuses mean sentiment and mention frequency:

```
net_stance = 0.50 · (mean_sinner − mean_alcaraz) + 0.15 · (freq_sinner − freq_alcaraz)
```

(the remaining 0.35 weight is reserved for a future emotion-based component). Users are
labelled `sinner` / `alcaraz` / `neutral` with a ±0.03 threshold (that can be adjusted in the constants section).

**Network.** Nodes are users; edges are replies/mentions. Centrality (degree, closeness,
betweenness, in/out-degree, PageRank) and communities (Louvain on `Gu`, Infomap on `Gd`)
are computed, with filtered subgraphs (components ≥ 10 nodes, top communities by volume)
used for visualisation and stance analysis.

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/brusati04/bluesky-sinner-alcaraz-analysis.git
cd bluesky-sinner-alcaraz-analysis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm   # auto-downloads on first run if missing
```

A CUDA-capable GPU is used automatically for the RoBERTa/GoEmotions models when available,
but the pipeline also runs on CPU.

## Usage

### Run the analysis pipeline

The processed dataset is already committed, so the pipeline runs end-to-end out of the box:

```bash
python main.py
```

On first run, preprocessing detects the cached `data/sinner_alcaraz_processed.csv` and loads
it directly (skipping the expensive NLP enrichment). Stages 3–5 then regenerate all CSVs and
plots in `data/` and `plots/`.

### Re-crawl from Bluesky (optional)

To rebuild the raw dataset you need Bluesky credentials. Create a `.env` file:

```bash
# create one in Bluesky Settings → App Passwords
BSKY_HANDLE=your-handle.bsky.social
BSKY_APP_PASSWORD=your-app-password   
```

Then run the crawler, and delete the processed cache so `main.py` re-enriches from scratch:

```bash
python src/crawler.py
rm data/sinner_alcaraz_processed.csv
python main.py
```

## Outputs

**`data/`**
- `sinner_alcaraz_posts.csv` — raw crawled posts (13 fields).
- `sinner_alcaraz_processed.csv` — **frozen** enriched dataset (44 columns: cleaned text,
  sentiment, NRC/BERT emotions, entities, post/user stance).
- `network_centrality_metrics.csv` — per-user centralities, community IDs and stance.
- `network_global_metrics.csv` — graph-level density, transitivity, clustering, GCC size,
  Louvain/Infomap modularity.

**`plots/`** (mirrored into `report/` for the LaTeX build) — 20+ PNGs including:
- Network graphs coloured by community, dominant emotion, and fanbase (plus `filtered/` views).
- Community emotion profiles (Louvain & Infomap × NRC & BERT).
- Sentiment distribution, sentiment over time, emotion-backend comparison.
- Stance analysis summary, fanbase classification comparison, community word cloud.

**`report/`** — `report.tex` and the figures used to build the PDF report.

## Project structure

```
.
├── main.py                          # Pipeline entrypoint (DAG orchestrator)
├── requirements.txt
├── src/
│   ├── crawler.py                   # [1] Bluesky atproto crawler
│   ├── preprocessing.py             # [2] NLP/stance enrichment (source of truth)
│   ├── social_network_analysis.py   # [3] Graphs, centralities, communities
│   ├── social_sentiment_analysis.py # [4] Sentiment & emotion plots
│   ├── social_stance_analysis.py    # [5] Stance aggregation & polarisation
│   └── utils.py                     # Shared constants & helpers
├── data/                            # Raw, processed & metrics CSVs
├── plots/                           # Generated visualisations
└── report/                          # LaTeX report + figures
```

## Tech stack

`atproto` · `pandas` / `numpy` / `scipy` · `networkx` + `infomap` · `spaCy` · `NLTK` ·
`transformers` / `torch` (RoBERTa, GoEmotions) · `nrclex` · `matplotlib` / `seaborn` /
`wordcloud`.

## Reproducibility

Random seeds are fixed (`random.seed(42)`, `np.random.seed(42)`) and the DAG design with a
frozen intermediate dataset means any downstream stage can be re-run without recomputing the
expensive NLP enrichment.

## License

Released under the [MIT License](LICENSE).
