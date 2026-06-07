import os
import ast
import json
from typing import Any, Union, Optional


workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.abspath(os.path.join(workspace_path, ".huggingface_cache"))

import torch
from transformers import pipeline

SINNER_URI = "http://dbpedia.org/resource/Jannik_Sinner"
ALCARAZ_URI = "http://dbpedia.org/resource/Carlos_Alcaraz"

SINNER_KEYWORDS = {"sinner", "jannik"}
ALCARAZ_KEYWORDS = {"alcaraz", "carlos", "carlitos"}

NRC_EMOTIONS = [
    'fear', 'anger', 'anticipation', 'trust',
    'surprise', 'sadness', 'disgust', 'joy'
]


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def parse_list_col(val: Any) -> list:
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        try:
            return json.loads(val)
        except (ValueError, TypeError):
            return []


def build_did_to_handle(df: pd.DataFrame) -> dict[str, str]:
    did_to_handle: dict[str, str] = {}
    for did, handle in zip(df['author_did'], df['author_handle']):
        if pd.notna(did) and pd.notna(handle):
            did_to_handle[did] = handle
    return did_to_handle


def save_plot_copies(filename: str, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

    report_dir = output_dir.replace("plots", "report", 1)
    os.makedirs(report_dir, exist_ok=True)
    plt.savefig(os.path.join(report_dir, filename), dpi=300, bbox_inches='tight')


def render_flat_chart(
    fig,
    filename: str,
    output_dir: str = "plots",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    legend_title: str = None,
    legend_ncol: int = None,
    tight_layout: bool = True,
    title_weight: str = "normal",
    title_pad: int = 15,
) -> None:
    if title:
        plt.title(title, fontsize=14, pad=title_pad, weight=title_weight)
    if xlabel:
        plt.xlabel(xlabel, fontsize=11, labelpad=10)
    if ylabel:
        plt.ylabel(ylabel, fontsize=11, labelpad=10)
    if legend_title:
        plt.legend(title=legend_title, ncol=legend_ncol or 1)
    if tight_layout:
        plt.tight_layout()
    save_plot_copies(filename, output_dir=output_dir)
    plt.close(fig)


_sentiment_clf_roberta = None


def get_sentiment_clf_roberta():
    global _sentiment_clf_roberta
    if _sentiment_clf_roberta is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[NLP] Initializing CardiffNLP RoBERTa sentiment model on device={device}...")
        _sentiment_clf_roberta = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            top_k=None
        )
    return _sentiment_clf_roberta


def score_roberta_sentiment(text_or_list: Union[str, list[str]], batch_size: int = 128) -> list[dict]:
    clf = get_sentiment_clf_roberta()
    if isinstance(text_or_list, str):
        inputs = [text_or_list]
    else:
        inputs = text_or_list

    def input_generator():
        for t in inputs:
            yield t if (isinstance(t, str) and t.strip() != "") else " "

    results = clf(input_generator(), batch_size=batch_size, truncation=True, max_length=512)
    outputs = []
    for res in results:
        scores_dict = {d['label']: d['score'] for d in res}
        if 'LABEL_0' in scores_dict:
            p_neg = scores_dict.get('LABEL_0', 0.0)
            p_pos = scores_dict.get('LABEL_2', 0.0)
            max_label = max(scores_dict, key=scores_dict.get)
            if max_label == 'LABEL_0':
                cat = 'negative'
            elif max_label == 'LABEL_2':
                cat = 'positive'
            else:
                cat = 'neutral'
        else:
            p_neg = scores_dict.get('negative', 0.0)
            p_pos = scores_dict.get('positive', 0.0)
            cat = max(scores_dict, key=scores_dict.get)
        comp = p_pos - p_neg
        outputs.append({'category': cat, 'compound': comp})
    return outputs


def detect_player_mentions(text: Optional[str], linked_entities: list) -> tuple[bool, bool]:
    uris = set()
    if isinstance(linked_entities, list):
        for ent in linked_entities:
            if isinstance(ent, dict):
                uris.add(ent.get('uri', ''))
            elif isinstance(ent, (tuple, list)) and len(ent) > 0:
                uris.add(ent[0])
    
    is_sinner = SINNER_URI in uris
    is_alcaraz = ALCARAZ_URI in uris
    
    if not is_sinner and not is_alcaraz:
        text_lower = str(text or '').lower()
        is_sinner = any(kw in text_lower for kw in SINNER_KEYWORDS)
        is_alcaraz = any(kw in text_lower for kw in ALCARAZ_KEYWORDS)
        
    return is_sinner, is_alcaraz


