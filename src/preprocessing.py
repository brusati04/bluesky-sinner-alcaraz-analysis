import os
import re
import sys
import html
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import torch
import spacy
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tqdm import tqdm
from nrclex import NRCLex
from transformers import pipeline

from utils import (
    parse_list_col,
    build_did_to_handle,
    save_plot_copies,
    detect_player_mentions,
    score_roberta_sentiment,
    get_sentiment_clf_roberta,
    NRC_EMOTIONS,
    SINNER_URI,
    ALCARAZ_URI,
    SINNER_KEYWORDS,
    ALCARAZ_KEYWORDS,
)


for _resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{_resource}' if _resource == 'stopwords' else f'tokenizers/{_resource}')
    except LookupError:
        nltk.download(_resource, quiet=True)


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found, downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

tokenizer = TweetTokenizer()
custom_stopwords = set(stopwords.words('english')).union({'tennis', 'match', 'play', 'player', 'set'})


def clean_text(text: Optional[str]) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_text_bert(text: Optional[str]) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(text)
   
    text = re.sub(r'http\S+|www\.\S+', 'http', text)
   
    text = re.sub(r'@\w+', '@user', text)
    return re.sub(r'\s+', ' ', text).strip()



def preprocess(text: Optional[str]) -> str:

    cleaned = clean_text(text)
    tokens = tokenizer.tokenize(cleaned.lower())

    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    for name in ["tagger", "attribute_ruler", "lemmatizer"]:
        if name in nlp.pipe_names:
            doc = nlp.get_pipe(name)(doc)

    filtered_tokens = []
    for token in doc:
        t = token.text
        lemma = token.lemma_.lower()
        if lemma in custom_stopwords or t in custom_stopwords:
            continue
        if t.isalnum() or t.startswith('#') or not t.isascii():
            if len(lemma) > 1 or not t.isascii():
                filtered_tokens.append(lemma)

    return " ".join(filtered_tokens)


def load_data(filepath: str, parse_linked_entities: bool = False) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    for directory in ("data", "plots", "report"):
        os.makedirs(directory, exist_ok=True)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found!")
        return None, None
   
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', format='mixed')
    df = df.dropna(subset=['created_at'])

    list_columns = ['mentions', 'hashtags', 'links']
    if parse_linked_entities:
        list_columns.append('linked_entities')

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_col)

    return df, build_did_to_handle(df)


GOEMOTIONS_TO_NRC = {
    "joy":          ["joy", "amusement", "excitement", "love", "pride", "relief"],
    "trust":        ["admiration", "approval", "gratitude", "caring"],
    "anticipation": ["curiosity", "desire", "optimism"],
    "surprise":     ["surprise", "realization", "confusion"],
    "fear":         ["fear", "nervousness"],
    "anger":        ["anger", "annoyance", "disapproval"],
    "disgust":      ["disgust"],
    "sadness":      ["sadness", "disappointment", "grief", "remorse", "embarrassment"],
}

POSITIVE_EMOTIONS = ["bert_emotion_joy", "bert_emotion_trust", "bert_emotion_anticipation"]
NEGATIVE_EMOTIONS = ["bert_emotion_anger", "bert_emotion_disgust", "bert_emotion_sadness", "bert_emotion_fear"]
W_SENTIMENT = 0.50
W_EMOTION = 0.35
W_FREQUENCY = 0.15

STANCE_THRESHOLD = 0.05
MIN_POSTS_FOR_STANCE = 1
DEBUG = True

_emotion_clf_bert = None


def _get_emotion_clf_bert():
    global _emotion_clf_bert
    if _emotion_clf_bert is None:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[NLP] Initializing GoEmotions BERT model on device={device}...")
        _emotion_clf_bert = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            truncation=True,
            max_length=128,
            device=device,
        )
    return _emotion_clf_bert


def extract_ner(text: Optional[str]) -> list[tuple[str, str]]:
    if pd.isna(text) or text == "":
        return []
    ents = []
    for ent in nlp(text).ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            ents.append((ent.text.strip().replace("\n", " "), ent.label_))
    return list(set(ents))


def link_entities_dbpedia(text: Optional[str], confidence: float = 0.5) -> list[dict]:
    if not text or pd.isna(text) or str(text).strip() == "":
        return []

    text_lower = text.lower()
    entities = []
    if any(k in text_lower for k in SINNER_KEYWORDS):
        entities.append({"surface_form": "Jannik Sinner", "uri": SINNER_URI})
    if any(k in text_lower for k in ALCARAZ_KEYWORDS):
        entities.append({"surface_form": "Carlos Alcaraz", "uri": ALCARAZ_URI})
    return entities


def score_emotions(text: str) -> dict[str, float]:
    try:
        if not text or pd.isna(text) or str(text).strip() == "":
            return {e: 0.0 for e in NRC_EMOTIONS}
    except (TypeError, ValueError):
        return {e: 0.0 for e in NRC_EMOTIONS}

    try:
        emotion_obj = NRCLex()
        emotion_obj.load_raw_text(str(text))
        freqs = emotion_obj.affect_frequencies
        return {e: freqs.get(e, 0.0) for e in NRC_EMOTIONS}
    except Exception:
        return {e: 0.0 for e in NRC_EMOTIONS}


def add_emotion_columns(df: pd.DataFrame, text_col: str = 'cleaned_text', backend: str = "nrc") -> pd.DataFrame:
    prefix = f"{backend}_emotion_"
    dom_col = f"{backend}_dominant_emotion"
    if backend == "bert":
        clf = _get_emotion_clf_bert()
        batch_size = 128
        emotion_inputs = df[text_col].tolist()

        def emotion_generator():
            for t in emotion_inputs:
                yield t if (isinstance(t, str) and t.strip() != "") else " "

        print(f"[NLP] Running GPU parallelized inference with batch_size={batch_size}...")
        raw_results = clf(
            emotion_generator(),
            batch_size=batch_size,
            truncation=True,
            max_length=128,
        )

        emotion_rows = []
        dominant_emotions = []
        for res in tqdm(raw_results, total=len(emotion_inputs), desc="GoEmotions BERT"):
            scores = {d["label"]: d["score"] for d in res}
            nrc_scores = {
                nrc: float(sum(scores.get(g, 0.0) for g in go))
                for nrc, go in GOEMOTIONS_TO_NRC.items()
            }
            emotion_rows.append(nrc_scores)

            raw_max_label = max(scores, key=scores.get)
            if raw_max_label == "neutral":
                dominant_emotions.append("neutral")
            else:
                dominant_emotions.append(max(nrc_scores, key=nrc_scores.get))

        emotion_df = pd.DataFrame(emotion_rows, index=df.index)
        emotion_df = emotion_df[NRC_EMOTIONS]
    else:
        emotion_df = pd.DataFrame(df[text_col].apply(score_emotions).tolist(), index=df.index)

    emotion_df.columns = [f'{prefix}{e}' for e in NRC_EMOTIONS]
    stale_cols = [c for c in emotion_df.columns if c in df.columns]
    if dom_col in df.columns:
        stale_cols.append(dom_col)
    if stale_cols:
        df = df.drop(columns=stale_cols)
    df = pd.concat([df, emotion_df], axis=1)

    if backend == "bert":
        df[dom_col] = dominant_emotions
    else:
        emotion_cols = list(emotion_df.columns)
        df[dom_col] = df[emotion_cols].idxmax(axis=1).str.replace(prefix, '', regex=False)
        df.loc[df[emotion_cols].sum(axis=1) == 0, dom_col] = 'neutral'
    return df


def derive_player_sentiment_scores(df: pd.DataFrame) -> dict[str, list[float]]:
    sinner_scores = df.loc[df['is_sinner'], 'sentiment_compound'].tolist()
    alcaraz_scores = df.loc[df['is_alcaraz'], 'sentiment_compound'].tolist()
    return {"sinner_scores": sinner_scores, "alcaraz_scores": alcaraz_scores}


def run_nlp_enrichment(df: pd.DataFrame, output_filepath: str = "data/sinner_alcaraz_processed.csv", emotion_backend: str = "both") -> dict:
    df = df.copy()
    total_posts = len(df)
    print(f"[NLP] Starting BERT/RoBERTa NLP enrichment for {total_posts} posts...")

    print("[NLP] Step 1/6: Cleaning and preprocessing post text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['preprocessed_text'] = df['text'].apply(preprocess)
    print("[NLP] Step 1/6 completed.")

    print("[NLP] Step 2/6: Scoring RoBERTa sentiments on GPU...")
    bert_inputs = df['text'].apply(clean_text_bert).tolist()
    sentiment_outputs = score_roberta_sentiment(bert_inputs, batch_size=128)
    df['sentiment_category'] = [item['category'] for item in sentiment_outputs]
    df['sentiment_compound'] = [item['compound'] for item in sentiment_outputs]
    print("[NLP] Step 2/6 completed.")

    if emotion_backend == "nrc":
        print("[NLP] Step 3/6: Running NRC Emotion Lexicon analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='nrc')
        print("[NLP] Step 3/6 completed.")
    elif emotion_backend == "bert":
        print("[NLP] Step 3/6: Running GoEmotions BERT emotion analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='bert')
        print("[NLP] Step 3/6 completed.")
    else:
        print("[NLP] Step 3/6: Running NRC Emotion Lexicon analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='nrc')
        print("[NLP] Step 3/6 completed.")

        print("[NLP] Step 4/6: Running GoEmotions BERT emotion analysis...")
        df = add_emotion_columns(df, text_col='cleaned_text', backend='bert')
        print("[NLP] Step 4/6 completed.")

    print("[NLP] Step 5/6: Extracting Named Entities (spaCy NER)...")
    df['entities'] = df['cleaned_text'].apply(extract_ner)
    print("[NLP] Step 5/6 completed.")

    print("[NLP] Step 6/6: Linking entities to DBpedia resources...")
    df['linked_entities'] = df['cleaned_text'].apply(link_entities_dbpedia)
    print("[NLP] Step 6/6 completed.")

    is_sinner_list = []
    is_alcaraz_list = []
    for _, row in df.iterrows():
        is_s, is_a = detect_player_mentions(row['cleaned_text'], row['linked_entities'])
        is_sinner_list.append(is_s)
        is_alcaraz_list.append(is_a)
    df['is_sinner'] = is_sinner_list
    df['is_alcaraz'] = is_alcaraz_list

    scores = derive_player_sentiment_scores(df)

    df.to_csv(output_filepath, index=False)
    print(f"[NLP] Saved processed dataset to {output_filepath}")
    return {"df": df, **scores}


def _sentence_sentiment(sentence: str) -> float:
    if not sentence or not sentence.strip():
        return 0.0

    cleaned = clean_text_bert(sentence)
    if not cleaned.strip():
        return 0.0

    clf = get_sentiment_clf_roberta()
    res = clf(cleaned, truncation=True, max_length=512)
    scores_dict = {d['label']: d['score'] for d in res[0]}

    if 'LABEL_0' in scores_dict:
        p_neg = scores_dict.get('LABEL_0', 0.0)
        p_pos = scores_dict.get('LABEL_2', 0.0)
    else:
        p_neg = scores_dict.get('negative', 0.0)
        p_pos = scores_dict.get('positive', 0.0)

    return p_pos - p_neg


def _sentence_mentions(sentence: str) -> tuple[bool, bool]:
    text_lower = sentence.lower()
    has_sinner = any(kw in text_lower for kw in SINNER_KEYWORDS)
    has_alcaraz = any(kw in text_lower for kw in ALCARAZ_KEYWORDS)
    return has_sinner, has_alcaraz


def _split_dual_mention(text: str, sentence_scores: dict[str, float]) -> tuple[float, float]:
    sentences = sent_tokenize(str(text))

    sinner_scores = []
    alcaraz_scores = []

    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        mentions_sinner, mentions_alcaraz = _sentence_mentions(sent_clean)
        sentiment = sentence_scores.get(sent_clean, 0.0)

        if mentions_sinner and not mentions_alcaraz:
            sinner_scores.append(sentiment)
        elif mentions_alcaraz and not mentions_sinner:
            alcaraz_scores.append(sentiment)
        elif mentions_sinner and mentions_alcaraz:
            sinner_scores.append(sentiment * 0.5)
            alcaraz_scores.append(sentiment * 0.5)

    sinner_score = np.mean(sinner_scores) if sinner_scores else 0.0
    alcaraz_score = np.mean(alcaraz_scores) if alcaraz_scores else 0.0

    return sinner_score, alcaraz_score


def detect_players(df: pd.DataFrame) -> pd.DataFrame:
    if 'is_sinner' in df.columns and 'is_alcaraz' in df.columns:
        return df

    def _detect_row(row):
        return detect_player_mentions(row.get('text', ''), row.get('linked_entities', []))

    detection = df.apply(_detect_row, axis=1, result_type='expand')
    detection.columns = ['is_sinner', 'is_alcaraz']
    return pd.concat([df, detection], axis=1)


def compute_post_stances(df: pd.DataFrame) -> pd.DataFrame:
    compound = df['sentiment_compound'].fillna(0)

    single_sinner = df['is_sinner'] & ~df['is_alcaraz']
    single_alcaraz = df['is_alcaraz'] & ~df['is_sinner']
    both_players = df['is_sinner'] & df['is_alcaraz']
    neither = ~df['is_sinner'] & ~df['is_alcaraz']

    df['stance_sinner'] = 0.0
    df['stance_alcaraz'] = 0.0
    df['emotion_sinner'] = 0.0
    df['emotion_alcaraz'] = 0.0

    pos_sum = df[POSITIVE_EMOTIONS].sum(axis=1)
    neg_sum = df[NEGATIVE_EMOTIONS].sum(axis=1)
    net_emotion = pos_sum - neg_sum

    df.loc[single_sinner, 'stance_sinner'] = compound[single_sinner]
    df.loc[single_alcaraz, 'stance_alcaraz'] = compound[single_alcaraz]
    df.loc[single_sinner, 'emotion_sinner'] = net_emotion[single_sinner]
    df.loc[single_alcaraz, 'emotion_alcaraz'] = net_emotion[single_alcaraz]
    df.loc[both_players, 'emotion_sinner'] = net_emotion[both_players] * 0.5
    df.loc[both_players, 'emotion_alcaraz'] = net_emotion[both_players] * 0.5

    if both_players.any():
        if DEBUG:
            print(f"\n[STANCE] Extracting sentences for {both_players.sum()} dual-mention posts...")
        dual_indices = df[both_players].index
        dual_texts = df.loc[both_players, 'text']

        post_sentences = {}
        all_unique_sentences = set()

        for idx, text in zip(dual_indices, dual_texts):
            sents = sent_tokenize(str(text))
            post_sentences[idx] = sents
            for s in sents:
                s_clean = s.strip()
                if s_clean:
                    all_unique_sentences.add(s_clean)

        unique_list = list(all_unique_sentences)
        sentence_scores = {}

        if unique_list:
            clf = get_sentiment_clf_roberta()
            batch_size = 128

            bert_inputs = [clean_text_bert(s) for s in unique_list]

            def input_generator():
                for t in bert_inputs:
                    yield t if (isinstance(t, str) and t.strip() != "") else " "

            print(f"[STANCE] Scoring {len(unique_list)} unique sentences in parallel batches of size {batch_size}...")
            raw_results = clf(
                input_generator(),
                batch_size=batch_size,
                truncation=True,
                max_length=512
            )

            for s, res in zip(unique_list, raw_results):
                scores_dict = {d['label']: d['score'] for d in res}
                if 'LABEL_0' in scores_dict:
                    p_neg = scores_dict.get('LABEL_0', 0.0)
                    p_pos = scores_dict.get('LABEL_2', 0.0)
                else:
                    p_neg = scores_dict.get('negative', 0.0)
                    p_pos = scores_dict.get('positive', 0.0)
                sentence_scores[s] = p_pos - p_neg

        sinner_scores = []
        alcaraz_scores = []
        for idx in dual_indices:
            s_score, a_score = _split_dual_mention(df.loc[idx, 'text'], sentence_scores)
            sinner_scores.append(s_score)
            alcaraz_scores.append(a_score)

        df.loc[both_players, 'stance_sinner'] = sinner_scores
        df.loc[both_players, 'stance_alcaraz'] = alcaraz_scores

    if DEBUG:
        print(f"[STANCE] Post-level stance scores computed.")
        print(f"  Single Sinner:  {single_sinner.sum():>6} posts")
        print(f"  Single Alcaraz: {single_alcaraz.sum():>6} posts")
        print(f"  Both players:   {both_players.sum():>6} posts (sentence-split)")
        print(f"  Neither:        {neither.sum():>6} posts")
        print(f"  Stance Sinner range:  {df['stance_sinner'].min():.3f} to {df['stance_sinner'].max():.3f}")
        print(f"  Stance Alcaraz range: {df['stance_alcaraz'].min():.3f} to {df['stance_alcaraz'].max():.3f}")

    return df


def compute_user_stances(df: pd.DataFrame, min_posts: int = MIN_POSTS_FOR_STANCE) -> pd.DataFrame:
    df = df.copy()
    if 'emotion_sinner' not in df.columns:
        df['emotion_sinner'] = 0.0
    if 'emotion_alcaraz' not in df.columns:
        df['emotion_alcaraz'] = 0.0

    grouped = df.groupby('author_handle')
    user_stances = grouped.agg(
        mean_sinner=('stance_sinner', 'mean'),
        mean_alcaraz=('stance_alcaraz', 'mean'),
        mean_emo_sinner=('emotion_sinner', 'mean'),
        mean_emo_alcaraz=('emotion_alcaraz', 'mean'),
        n_sinner=('is_sinner', 'sum'),
        n_alcaraz=('is_alcaraz', 'sum'),
        post_count=('stance_sinner', 'count')
    )

    total_users_before = len(user_stances)
    user_stances = user_stances[user_stances['post_count'] >= min_posts]
    if DEBUG:
        print(f"\n[STANCE] User filtering: {total_users_before} -> {len(user_stances)} users (min {min_posts} posts)")

    user_stances[['mean_sinner', 'mean_alcaraz', 'mean_emo_sinner', 'mean_emo_alcaraz']] = user_stances[['mean_sinner', 'mean_alcaraz', 'mean_emo_sinner', 'mean_emo_alcaraz']].fillna(0.0)

    total_mentions = user_stances['n_sinner'] + user_stances['n_alcaraz']
    user_stances['freq_sinner'] = (user_stances['n_sinner'] / total_mentions.replace(0, np.nan)).fillna(0)
    user_stances['freq_alcaraz'] = (user_stances['n_alcaraz'] / total_mentions.replace(0, np.nan)).fillna(0)

    user_stances['net_stance'] = (
        W_SENTIMENT * (user_stances['mean_sinner'] - user_stances['mean_alcaraz'])
        + W_FREQUENCY * (user_stances['freq_sinner'] - user_stances['freq_alcaraz'])
        + W_EMOTION * (user_stances['mean_emo_sinner'] - user_stances['mean_emo_alcaraz'])
    )

    if DEBUG:
        print(f"[STANCE] Net stance range: {user_stances['net_stance'].min():.3f} to {user_stances['net_stance'].max():.3f}")
        print(f"[STANCE] Net stance mean:  {user_stances['net_stance'].mean():.3f}")
        print(f"[STANCE] Net stance std:   {user_stances['net_stance'].std():.3f}")

    return user_stances


def classify_stances(user_stances: pd.DataFrame, threshold: float = STANCE_THRESHOLD) -> pd.DataFrame:
    def label(score):
        if score > threshold:
            return 'sinner'
        elif score < -threshold:
            return 'alcaraz'
        return 'neutral'

    user_stances['stance_leaning'] = user_stances['net_stance'].apply(label)

    if DEBUG:
        counts = user_stances['stance_leaning'].value_counts()
        total = len(user_stances)
        for stance in ['sinner', 'alcaraz', 'neutral']:
            n = counts.get(stance, 0)
            print(f"[STANCE] {stance.capitalize():>8}: {n:>5} ({n/total*100:.1f}%)")

    return user_stances


def prepare_dataset(raw_filepath: str, processed_filepath: str) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    df = None
    did_to_handle = None
    if os.path.exists(processed_filepath):
        print(f"[INFO] Processed dataset found at {processed_filepath}. Loading it directly...")
        df, did_to_handle = load_data(processed_filepath, parse_linked_entities=True)
    else:
        raw_df, did_to_handle = load_data(raw_filepath, parse_linked_entities=False)
        if raw_df is None:
            return None, None

        df = run_nlp_enrichment(raw_df, output_filepath=processed_filepath)["df"]

    if df is not None:
        required_cols = [
            'stance_sinner', 'stance_alcaraz', 
            'emotion_sinner', 'emotion_alcaraz',
            'author_stance_score', 'author_stance_leaning',
            'author_dominant_emotion_nrc', 'author_dominant_emotion_bert'
        ]
        already_computed = all(col in df.columns for col in required_cols)

        if already_computed:
            print("[INFO] Stance and dominant emotion columns already computed. Skipping recalculation...")
        else:
            print("[INFO] Recalculating stance and dominant emotion columns to processed dataset...")

            df = compute_post_stances(df)
            user_stances = compute_user_stances(df, min_posts=1)
            user_stances = classify_stances(user_stances, threshold=0.05)
            
            df['author_stance_score'] = df['author_handle'].map(user_stances['net_stance']).fillna(0.0)
            df['author_stance_leaning'] = df['author_handle'].map(user_stances['stance_leaning']).fillna('neutral')
            
            for backend in ("nrc", "bert"):
                col = "nrc_dominant_emotion" if backend == "nrc" else "bert_dominant_emotion"
                user_emotions = df.groupby("author_handle")[col].agg(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else "neutral"
                ).to_dict()
                df[f'author_dominant_emotion_{backend}'] = df['author_handle'].map(user_emotions).fillna("neutral")
                
            df.to_csv(processed_filepath, index=False)
            print(f"[INFO] Processed dataset cache updated at {processed_filepath}")
            
    return df, did_to_handle


def plot_community_wordcloud(df: pd.DataFrame, output_dir: str = "plots") -> None:

    
    if 'preprocessed_text' not in df.columns:
        print("Warning: 'preprocessed_text' column not found. Skipping word cloud.")
        return

    from wordcloud import WordCloud

    valid_texts = df['preprocessed_text'].dropna().astype(str).tolist()
    valid_texts = [t for t in valid_texts if t.strip() != ""]

    if not valid_texts:
        print("Warning: No words found for community word cloud. Skipping.")
        return

    community_text = " ".join(valid_texts)

    wordcloud_community = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        random_state=42,
        max_words=300
    ).generate(community_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_community, interpolation='bilinear')
    plt.axis('off')
    plt.title("Bluesky Community - Overall Word Cloud\n(US Open 2025)", fontsize=16, pad=15, weight='bold')
    plt.tight_layout()
    save_plot_copies("community_wordcloud.png", output_dir=output_dir)
    plt.close()
