import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv
from atproto import Client, models
from atproto_client.exceptions import RequestException

load_dotenv()


def create_client() -> Client:
    """Validate Bluesky credentials from the environment and return a logged-in client"""
    handle = os.environ.get("BSKY_HANDLE")
    app_password = os.environ.get("BSKY_APP_PASSWORD")
    if not handle or not app_password:
        raise EnvironmentError(
            "Missing Bluesky credentials"
            "Set BSKY_HANDLE and BSKY_APP_PASSWORD as environment variables "
            "or create a .env file. See .env.example for the required format."
        )

    client = Client()
    client.login(handle, app_password)
    return client


def _parse_dt_utc(iso_str: str) -> datetime:
    """Parse an ISO datetime string into a timezone-aware UTC datetime"""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now_utc_ts() -> int:
    """Return the current UTC time as a Unix timestamp"""
    return int(datetime.now(timezone.utc).timestamp())


def call_with_retries(callable_fn: Callable, *args: Any, max_retries: int = 10, **kwargs: Any) -> Any:
    """Call an atproto function with retries, honouring 429 rate-limit headers and exponential backoff"""
    for attempt in range(max_retries):
        try:
            return callable_fn(*args, **kwargs)
        except RequestException as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            headers = getattr(resp, "headers", {}) or {}

            if status == 429:
                reset = headers.get("ratelimit-reset") or headers.get("RateLimit-Reset")
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                if reset:
                    wait_s = max(1, int(reset) - _now_utc_ts()) + 1
                elif retry_after:
                    wait_s = int(float(retry_after))
                else:
                    wait_s = min(60, 2 ** attempt)
                print(f"[429] rate limited, sleeping {wait_s}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            wait_s = min(10, 2 ** attempt)
            print(f"[{status}] request failed, sleeping {wait_s}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait_s)

    raise RuntimeError("Too many retries / repeated failures")


def extract_facets(record: Any) -> dict:
    """Extract hashtags, mentions and links from a post record's richtext facets"""
    hashtags: list[str] = []
    mentions: list[dict] = []
    links: list[str] = []

    for facet in getattr(record, "facets", None) or []:
        for feat in getattr(facet, "features", None) or []:
            ftype = getattr(feat, "py_type", None) or getattr(feat, "type", None)
            if ftype == "app.bsky.richtext.facet#tag":
                tag = getattr(feat, "tag", None)
                if tag:
                    hashtags.append(tag)
            elif ftype == "app.bsky.richtext.facet#mention":
                did = getattr(feat, "did", None)
                if did:
                    mentions.append({"did": did})
            elif ftype == "app.bsky.richtext.facet#link":
                uri = getattr(feat, "uri", None)
                if uri:
                    links.append(uri)

    return {"hashtags": hashtags, "mentions": mentions, "links": links}


def get_reply_details(record: Any) -> dict:
    """Extract reply parent/root URIs and the parent author's DID (parsed from the AT URI)"""
    reply = getattr(record, "reply", None)
    parent_uri = root_uri = parent_author_did = None

    if reply:
        parent = getattr(reply, "parent", None)
        root = getattr(reply, "root", None)
        if parent:
            parent_uri = getattr(parent, "uri", None)
        if root:
            root_uri = getattr(root, "uri", None)
        if parent_uri and parent_uri.startswith("at://"):
            parts = parent_uri.replace("at://", "").split("/")
            if parts:
                parent_author_did = parts[0]

    return {"reply_parent_uri": parent_uri, "reply_root_uri": root_uri, "reply_parent_did": parent_author_did}


def search_posts_time_window(
    client: Client,
    query: str,
    since_iso: str,
    until_iso: str,
    max_posts: int = 1000,
    page_size: int = 50,
    polite_sleep: float = 0.25,
    print_every_page: bool = True,
) -> pd.DataFrame:
    """Collect up to max_posts posts within [since_iso, until_iso) using cursor pagination"""
    cursor = None
    rows = []
    page = 0
    since_dt = _parse_dt_utc(since_iso)
    until_dt = _parse_dt_utc(until_iso)

    while len(rows) < max_posts:
        page += 1
        params = models.AppBskyFeedSearchPosts.Params(
            q=query,
            sort="latest",
            since=since_iso,
            until=until_iso,
            limit=min(page_size, max_posts - len(rows)),
            cursor=cursor,
        )
        res = call_with_retries(client.app.bsky.feed.search_posts, params)

        cursor = res.cursor
        posts = res.posts or []
        if not posts:
            break

        for p in posts:
            rec = getattr(p, "record", None)
            created = getattr(rec, "created_at", None)
            if not created:
                continue

            created_dt = _parse_dt_utc(created)
            if not (since_dt <= created_dt < until_dt):
                continue

            author = getattr(p, "author", None)
            facets_data = extract_facets(rec)
            reply_data = get_reply_details(rec)

            rows.append({
                "created_at": created_dt,
                "uri": p.uri,
                "cid": getattr(p, "cid", None),
                "text": getattr(rec, "text", None),
                "author_handle": getattr(author, "handle", None),
                "author_did": getattr(author, "did", None),
                "reply_count": getattr(p, "reply_count", None),
                "like_count": getattr(p, "like_count", None),
                "repost_count": getattr(p, "repost_count", None),
                "quote_count": getattr(p, "quote_count", None),
                "hashtags": facets_data["hashtags"],
                "mentions": facets_data["mentions"],
                "links": facets_data["links"],
                "reply_parent_uri": reply_data["reply_parent_uri"],
                "reply_root_uri": reply_data["reply_root_uri"],
                "reply_parent_did": reply_data["reply_parent_did"],
            })

            if len(rows) >= max_posts:
                break

        if print_every_page:
            print(f"    Page {page}: fetched {len(posts)} posts. Valid matches in window so far: {len(rows)}")

        if cursor is None:
            break
        time.sleep(polite_sleep)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df = df.sort_values("created_at", ascending=True).reset_index(drop=True)
    return df


def search_posts_day_by_day(
    client: Client,
    query: str,
    since_iso: str,
    until_iso: str,
    max_posts_per_day: int = 1000,
    page_size: int = 100,
    polite_sleep: float = 0.25,
) -> pd.DataFrame:
    """Collect posts by slicing the timeframe into 24h windows to bypass pagination limits"""
    since_dt = _parse_dt_utc(since_iso)
    until_dt = _parse_dt_utc(until_iso)

    current_dt = since_dt
    all_dfs = []

    while current_dt < until_dt:
        next_dt = min(current_dt + timedelta(days=1), until_dt)
        day_since_str = current_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        day_until_str = next_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"  -> Crawling time window: {day_since_str} to {day_until_str}...")
        df_day = search_posts_time_window(
            client=client,
            query=query,
            since_iso=day_since_str,
            until_iso=day_until_str,
            max_posts=max_posts_per_day,
            page_size=page_size,
            polite_sleep=polite_sleep,
            print_every_page=True,
        )

        if not df_day.empty:
            print(f"     Found {len(df_day)} matching posts for this day")
            all_dfs.append(df_day)
        else:
            print("     No matching posts found for this day")

        current_dt = next_dt

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


if __name__ == "__main__":
    client = create_client()

    since_iso = "2025-08-21T00:00:00.000Z"
    until_iso = "2025-09-10T00:00:00.000Z"
    QUERIES = {
        "sinner": '(jannik | sinner | Jannik | Sinner) (tennis | usopen | "us open" | alcaraz | Alcaraz | slam | match)',
        "alcaraz": '(alcaraz | Alcaraz) (tennis | usopen | "us open" | jannik | sinner | Jannik | Sinner | slam | match)',
    }
    MAX_POSTS_PER_DAY = 5000

    df_list = []
    for query_name, query_str in QUERIES.items():
        print("\n" + "=" * 50)
        print(f"STARTING CRAWL FOR QUERY: '{query_name}'")
        print("=" * 50)
        df_q = search_posts_day_by_day(
            client=client,
            query=query_str,
            since_iso=since_iso,
            until_iso=until_iso,
            max_posts_per_day=MAX_POSTS_PER_DAY,
            page_size=100,
            polite_sleep=0.25,
        )

        if not df_q.empty:
            df_q["query_source"] = query_name
            df_list.append(df_q)
            print(f"Finished crawling '{query_name}' query. Total unique day-matched posts: {len(df_q)}")
        else:
            print(f"Finished crawling '{query_name}' query. No posts were returned.")

    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
        print(f"\nCombining crawled data. Combined total across queries: {len(df_all)} posts.")
        df_all["in_both"] = df_all.duplicated(subset="uri", keep=False)
        df_all = df_all.drop_duplicates(subset="uri", keep="first")
        print(f"Deduplicated total unique posts: {len(df_all)} posts.")
        df_all = df_all.sort_values("created_at", ascending=True).reset_index(drop=True)

        cols_to_drop = ["cid", "quote_count", "hashtags", "reply_parent_uri", "reply_root_uri"]
        df_all = df_all.drop(columns=[col for col in cols_to_drop if col in df_all.columns], errors="ignore")

        os.makedirs("data", exist_ok=True)
        output_file = "data/sinner_alcaraz_posts.csv"
        df_all.to_csv(output_file, index=False)
        print(f"Successfully saved crawled posts to: {output_file}")
    else:
        print("\nNo posts found")
