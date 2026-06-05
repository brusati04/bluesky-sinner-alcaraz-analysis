import os
import ast
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set styling for high-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# US Open 2025 Key Match Dates
US_OPEN_EVENTS = [
    ("2025-08-24", "US Open begins", "grey"),             
    ("2025-08-25", "Alcaraz R1", "blue"),                  
    ("2025-08-26", "Sinner R1", "blue"),                   
    ("2025-08-27", "R2 matches", "blue"),                  
    ("2025-08-30", "R3 matches", "blue"),                  
    ("2025-09-01", "R4 matches", "blue"),                  
    ("2025-09-03", "Quarterfinals", "orange"),             
    ("2025-09-05", "Semifinals", "red"),                   
    ("2025-09-07", "Final\n(Alcaraz wins)", "darkred"),    
]

# US Open 2025 Round Windows (for median sentiment per round plot)
US_OPEN_ROUNDS = [
    {"label": "Pre-Tournament", "start": "2025-08-21", "end": "2025-08-23"},
    {"label": "US Open begins",  "start": "2025-08-24", "end": "2025-08-24"},
    {"label": "R1",              "start": "2025-08-25", "end": "2025-08-26"},
    {"label": "R2",              "start": "2025-08-27", "end": "2025-08-28"},
    {"label": "R3",              "start": "2025-08-29", "end": "2025-08-30"},
    {"label": "R4",              "start": "2025-08-31", "end": "2025-09-01"},
    {"label": "Quarterfinals",   "start": "2025-09-02", "end": "2025-09-03"},
    {"label": "Semifinals",      "start": "2025-09-04", "end": "2025-09-05"},
    {"label": "Final",           "start": "2025-09-06", "end": "2025-09-07"},
    {"label": "Post-Final",      "start": "2025-09-08", "end": "2025-09-09"},
]

def parse_list_col(val):
    """Safely parse columns stored as string-formatted lists."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return json.loads(val)
        except Exception:
            return []

def build_did_to_handle(df):
    """Build a DID -> handle mapping from the posts DataFrame."""
    did_to_handle = {}
    for _, row in df.iterrows():
        did = row['author_did']
        handle = row['author_handle']
        if pd.notna(did) and pd.notna(handle):
            did_to_handle[did] = handle
    return did_to_handle

def save_plot_copies(filename, output_dir="plots"):
    """Save the current figure to output_dir/ and the mirrored report/ directory."""
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    report_dir = output_dir.replace("plots", "report", 1)
    os.makedirs(report_dir, exist_ok=True)
    plt.savefig(os.path.join(report_dir, filename), dpi=300, bbox_inches='tight')
