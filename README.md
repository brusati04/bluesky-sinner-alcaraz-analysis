# Sinner vs Alcaraz: Social Media Analysis on Bluesky (US Open 2025)

Web and Social Media Analysis — University of Pavia / University of Milano-Bicocca  
**Authors:** 
* Lorenzo Brusati (lorenzo.brusati01@universitadipavia.it)
* Lorenzo Cinquemani (lorenzo.cinquemani01@universitadipavia.it)
* Lorenzo Goatelli (lorenzo.goatelli01@universitadipavia.it)

**Course:** WSA — Web and Social Media Search and Analysis

---

## Repository Structure

```
project_root\
├── main.py                  # Root-level entrypoint (hit-and-run execution script)
├── src\
│   ├── crawler.py           # AT Protocol data crawler
│   ├── network_analysis.py  # Network modeling, centralities, and community detection
│   ├── nlp_analysis.py      # VADER sentiment, spaCy NER, DBpedia Spotlight NEL, Kruskal-Wallis testing
│   ├── stance_propagation.py # Iterative Laplacian smoothing Stance Propagation
│   ├── visualization.py     # Matplotlib and Seaborn visualization rendering
│   ├── export_dashboard_data.py # Profiles communities for all algorithms and exports dashboard JSON
│   ├── utils.py             # Shared utilities (Regex fallback entity linking, data loader, etc.)
│   └── emotion_analysis.py  # NRC Emotion Lexicon module
├── data\                    # Dataset files (crawled posts and centrality metrics)
│   ├── sinner_alcaraz_posts.csv
│   └── network_centrality_metrics.csv
├── plots\                   # Generated plot images (community detection comparison, stance propagation, etc.)
├── report\                  # LaTeX report source files and compiled PDF
│   ├── report.tex
│   └── report.pdf
├── .env.example             # Credentials template
├── web\                     # Interactive dashboard assets
│   ├── index.html           # Interactive visual explorer interface (HTML)
│   ├── dashboard.js         # Visual explorer script (dynamically handles vis-network rendering)
│   ├── dashboard.css        # Premium dashboard dark theme styling
│   └── data\
│       └── dashboard_data.json # Profiles communities for all algorithms (JSON payload)
└── requirements.txt         # Python dependencies
```

---

## Setup (first time only)

Open a Command Prompt/PowerShell in this folder and run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
python -m textblob.download_corpora
```

---

## Configure Credentials

Only needed if you want to re-crawl data. Skip this step if you are using the provided `data\sinner_alcaraz_posts.csv`.

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
2. Open `.env` in Notepad and fill in your Bluesky credentials:
   ```env
   BSKY_HANDLE=your-email@domain.com
   BSKY_APP_PASSWORD=your-app-password-here
   ```

Never share or commit the `.env` file.

---

## Run the Pipeline

### Step 1 — Data collection (optional, skip if data already exists)

From Command Prompt:
```bash
python src\crawler.py
```
Output: `data\sinner_alcaraz_posts.csv`

### Step 2 — Run the Analysis & Dashboard Data Export (Hit-and-Run Pipeline)

From Command Prompt:
```bash
python main.py
```
This single entrypoint handles the entire analysis and visualization pipeline end-to-end:
* **Loads and Preprocesses** crawled Bluesky data.
* **Performs Social Network Analysis (SNA)**: builds undirected/directed networks, computes degree, closeness, PageRank, and assortativity metrics.
* **Compares Community Detection Algorithms**: Louvain, Leiden, Infomap, Label Propagation (LPA), and Fluid Communities (on GCC).
* **Enriches Text with NLP**: extracts VADER sentiments, calculates NRC emotion profiles, maps entities via spaCy NER and DBpedia Spotlight NEL.
* **Propagates Stance Attributes**: runs Laplacian smoothing to classify Sinner vs. Alcaraz vs. Neutral fans.
* **Generates Comparative Visualizations**: saves 12 comparative plots to `plots/` and `report/`.
* **Exports Dashboard JSON**: automatically runs `export_dashboard_data.py` to generate `web/data/dashboard_data.json`.


### Step 4 — Launch the Interactive Explorer

To view the premium interactive dashboard with dynamic algorithm switching:
1. Start a local HTTP server targeting the `web` folder:
   ```bash
   python -m http.server 8000 -d web
   ```
2. Open your browser and navigate to **`http://localhost:8000`** (this automatically serves `web/index.html`).
3. Use the **Detection Algorithm** dropdown in the Left Sidebar to switch community partitions dynamically (forces nodes to filter and rearrange dynamically in the graph view).

---

## Troubleshooting

**"Python is not recognized"**  
→ Re-install Python and make sure to check "Add Python to PATH".

**"No module named X"**  
→ Run `setup.bat` again, or manually: `pip install -r requirements.txt`

**"Missing Bluesky credentials"**  
→ You need a `.env` file. See "Configure Credentials" above. If you already have the CSV data, you can skip the crawler entirely.

**"data\sinner_alcaraz_posts.csv not found"**  
→ Either run the crawler (Step 1) or make sure the CSV file is in the `data\` folder.

---

## Dependencies

Key libraries listed in `requirements.txt`:
* `atproto` — Bluesky AT Protocol SDK
* `networkx` — Graph construction and SNA metrics
* `igraph` + `leidenalg` — Optimized Leiden community algorithm
* `infomap` — Infomap community detection
* `vaderSentiment` — Sentiment analysis (VADER)
* `nrclex` — NRC Emotion Lexicon for emotion analysis
* `spacy` + `en_core_web_sm` — Named Entity Recognition
* `requests` — DBpedia Spotlight NEL API calls (with circuit breaker)
* `python-dotenv` — Credential management
* `scipy` — Kruskal-Wallis statistical test
