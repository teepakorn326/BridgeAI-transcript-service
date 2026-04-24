import glob
import json
import logging
import os
import shutil
import tempfile

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BridgeAI Transcript Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MAX_DURATION_CAPTIONS = int(os.getenv("MAX_DURATION_CAPTIONS", "14400"))  # 4 hours
MAX_DURATION_WHISPER = int(os.getenv("MAX_DURATION_WHISPER", "3600"))  # 1 hour

logger.info(
    f"Loading Whisper model: {WHISPER_MODEL}, "
    f"max captions: {MAX_DURATION_CAPTIONS}s, max whisper: {MAX_DURATION_WHISPER}s"
)
whisper_model = WhisperModel(WHISPER_MODEL, compute_type="int8")
logger.info("Whisper model loaded")

# Covers every language in the frontend LANGUAGES list.
LANG_MAP = {
    "Thai": "th",
    "English": "en",
    "Chinese": "zh",
    "Chinese (Simplified)": "zh-Hans",
    "Japanese": "ja",
    "Korean": "ko",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Hindi": "hi",
}

YOUTUBE_PLAYER_CLIENTS = ["web", "tv_embedded", "ios", "android", "mweb"]
YOUTUBE_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 "
    "Mobile/15E148 Safari/604.1"
)

YT_PROXY = os.getenv("YT_PROXY") or None
if YT_PROXY:
    # Log only the host:port, never the full URL (contains credentials).
    try:
        from urllib.parse import urlparse
        _p = urlparse(YT_PROXY)
        logger.info(f"YT_PROXY configured: {_p.hostname}:{_p.port}")
    except Exception:
        logger.info("YT_PROXY configured (unparseable — check format)")
else:
    logger.info("YT_PROXY not set — requests go directly from this server's IP")


def youtube_transcript_api_client(cookies_path):
    """Build a YouTubeTranscriptApi client with proxy + cookies if configured."""
    kwargs = {}
    if cookies_path:
        kwargs["cookie_path"] = cookies_path
    if YT_PROXY:
        kwargs["proxy_config"] = GenericProxyConfig(
            http_url=YT_PROXY,
            https_url=YT_PROXY,
        )
    return YouTubeTranscriptApi(**kwargs)


def get_cookies_path():
    """Locate a cookies.txt; return None if not present."""
    for candidate in ("/etc/secrets/cookies.txt", "cookies.txt"):
        if os.path.exists(candidate):
            return candidate
    return None


def writable_cookies(tmpdir):
    """Copy the read-only Render secret to a writable temp path for yt-dlp."""
    src = get_cookies_path()
    if not src:
        return None
    dst = os.path.join(tmpdir, "cookies.txt")
    shutil.copy2(src, dst)
    return dst


def preferred_lang_code(preferred_lang):
    if not preferred_lang:
        return None
    return LANG_MAP.get(preferred_lang, preferred_lang.lower()[:2])


def clamp_end_times(segments):
    # YouTube auto-captions overlap (each caption fades in before the prior
    # one fades out). Clamp each end to the next start so frontend matching
    # picks exactly one active line at any instant.
    for i in range(len(segments) - 1):
        if segments[i]["end_seconds"] > segments[i + 1]["start_seconds"]:
            segments[i]["end_seconds"] = segments[i + 1]["start_seconds"]
    return segments


# ── Path 1: youtube-transcript-api (1.0.x instance API) ────────────────────
def fetch_youtube_transcript_api(video_id, preferred_lang=None):
    """Fast path — hits YouTube's timedtext endpoint directly. Returns None on any failure."""
    cookies_path = get_cookies_path()
    api = youtube_transcript_api_client(cookies_path)

    try:
        transcript_list = api.list(video_id)

        # Priority: manual EN → generated EN → preferred lang → first available
        transcript = None
        for finder, langs, label in (
            (transcript_list.find_manually_created_transcript, ["en", "en-US", "en-GB"], "manual EN"),
            (transcript_list.find_generated_transcript, ["en", "en-US", "en-GB"], "generated EN"),
        ):
            try:
                transcript = finder(langs)
                logger.info(f"Using {label} transcript for {video_id}")
                break
            except Exception:
                continue

        if transcript is None:
            lang_code = preferred_lang_code(preferred_lang)
            if lang_code:
                try:
                    transcript = transcript_list.find_transcript([lang_code])
                    logger.info(f"Using {lang_code} transcript for {video_id}")
                except Exception:
                    pass

        if transcript is None:
            transcript = next(iter(transcript_list))
            logger.info(f"Using fallback transcript ({transcript.language_code}) for {video_id}")

        data = transcript.fetch()
    except Exception as e:
        logger.warning(f"YouTubeTranscriptApi failed for {video_id}: {e}")
        return None

    # FetchedTranscriptSnippet uses attribute access in 1.0+.
    segments = []
    for snippet in data:
        start = round(snippet.start, 2)
        if start >= MAX_DURATION_CAPTIONS:
            break
        segments.append({
            "start_seconds": start,
            "end_seconds": min(round(snippet.start + snippet.duration, 2), MAX_DURATION_CAPTIONS),
            "text": snippet.text,
        })
    return clamp_end_times(segments) or None


# ── Path 2: yt-dlp subtitle download (json3) ───────────────────────────────
def fetch_ytdlp_subtitles(video_id, preferred_lang=None):
    """
    Middle path — uses yt-dlp's subtitle fetcher (NOT audio download), which
    hits a different YouTube endpoint than the transcript API and sometimes
    succeeds when path 1 doesn't. Much cheaper than Whisper.
    Returns None if no subtitles are available via this path.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    langs = ["en", "en-US", "en-GB"]
    pref = preferred_lang_code(preferred_lang)
    if pref and pref not in langs:
        langs.append(pref)

    with tempfile.TemporaryDirectory() as tmpdir:
        cookies = writable_cookies(tmpdir)
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": langs,
            "subtitlesformat": "json3",  # easier to parse than vtt
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "cookiefile": cookies,
            "extractor_args": {"youtube": {"player_client": YOUTUBE_PLAYER_CLIENTS}},
            "http_headers": {"User-Agent": YOUTUBE_MOBILE_UA},
            "quiet": True,
            "no_warnings": True,
            "retries": 2,
        }
        if YT_PROXY:
            ydl_opts["proxy"] = YT_PROXY

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)  # writes subtitle files to tmpdir
        except Exception as e:
            logger.warning(f"yt-dlp subtitle fetch failed for {video_id}: {e}")
            return None

        # yt-dlp writes files like <video_id>.<lang>.json3 — pick the best match.
        for lang in langs:
            matches = glob.glob(os.path.join(tmpdir, f"*.{lang}*.json3"))
            if matches:
                try:
                    with open(matches[0], "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    segments = parse_json3(payload)
                    if segments:
                        logger.info(f"Using yt-dlp {lang} subtitles for {video_id}")
                        return segments
                except Exception as e:
                    logger.warning(f"Failed to parse {matches[0]}: {e}")

    return None


def parse_json3(payload):
    """Convert YouTube's json3 subtitle format into our segment shape."""
    segments = []
    for event in payload.get("events", []):
        if "segs" not in event or "tStartMs" not in event:
            continue
        start = event["tStartMs"] / 1000.0
        if start >= MAX_DURATION_CAPTIONS:
            break
        duration = event.get("dDurationMs", 2000) / 1000.0
        text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
        if not text:
            continue
        segments.append({
            "start_seconds": round(start, 2),
            "end_seconds": min(round(start + duration, 2), MAX_DURATION_CAPTIONS),
            "text": text,
        })
    return clamp_end_times(segments)


# ── Path 3: Whisper (download audio, transcribe) ───────────────────────────
def fetch_whisper_transcript(video_id):
    """Slow fallback — downloads audio and runs Whisper locally."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        cookies = writable_cookies(tmpdir)
        if cookies:
            logger.info(f"Copied cookies to {cookies}")

        ydl_opts = {
            # Progressively more lenient: audio-only (any codec) → audio-only
            # (strict) → best combined stream. Mobile player clients return
            # narrower format sets, so we need fallbacks.
            "format": "bestaudio*/bestaudio/best",
            "outtmpl": audio_path,
            "cookiefile": cookies,
            "extractor_args": {"youtube": {"player_client": YOUTUBE_PLAYER_CLIENTS}},
            "http_headers": {"User-Agent": YOUTUBE_MOBILE_UA},
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "96",
            }],
            "download_ranges": lambda info, ydl: [{"start_time": 0, "end_time": MAX_DURATION_WHISPER}],
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
        }
        if YT_PROXY:
            ydl_opts["proxy"] = YT_PROXY
            logger.info("Using YT_PROXY for yt-dlp")

        logger.info(f"Downloading audio for {video_id} (player_client={','.join(YOUTUBE_PLAYER_CLIENTS)})")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as e:
            # download_ranges requires formats that support byte-range
            # extraction. Some YouTube streams don't, producing
            # "Requested format is not available". Retry full-download
            # with a more permissive format spec; we still truncate to
            # MAX_DURATION_WHISPER in the segment loop below.
            if "Requested format is not available" not in str(e):
                raise
            logger.warning(
                f"[Whisper] range-download failed for {video_id}, "
                f"retrying full download: {e}"
            )
            ydl_opts.pop("download_ranges", None)
            ydl_opts["format"] = "bestaudio/best"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        if not os.path.exists(audio_path) and os.path.exists(audio_path + ".mp3"):
            audio_path = audio_path + ".mp3"

        logger.info(f"Transcribing with Whisper ({WHISPER_MODEL})")
        result_segments, _ = whisper_model.transcribe(audio_path)

        segments = []
        for seg in result_segments:
            if seg.start >= MAX_DURATION_WHISPER:
                break
            segments.append({
                "start_seconds": round(seg.start, 2),
                "end_seconds": min(round(seg.end, 2), MAX_DURATION_WHISPER),
                "text": seg.text.strip(),
            })
    return segments


def is_bot_detection(err_msg):
    """Match YouTube's bot-challenge message — handles both ASCII and Unicode apostrophes."""
    lowered = err_msg.lower()
    needles = (
        "confirm you’re not a bot",  # ’
        "confirm you're not a bot",       # '
        "sign in to confirm",
        "http error 403",
    )
    return any(n in lowered for n in needles)


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "service": "transcript-service"}


@app.get("/transcript/{video_id}")
def get_transcript(video_id: str, lang: str = None):
    # 1. Cheap API path
    logger.info(f"Path 1: YouTubeTranscriptApi for {video_id} (preferred lang: {lang})")
    segments = fetch_youtube_transcript_api(video_id, lang)
    if segments:
        return {"video_id": video_id, "segments": segments, "source": "youtube"}

    # 2. yt-dlp subtitle fetch (different endpoint than path 1)
    logger.info(f"Path 2: yt-dlp subtitles for {video_id}")
    segments = fetch_ytdlp_subtitles(video_id, lang)
    if segments:
        return {"video_id": video_id, "segments": segments, "source": "ytdlp-subs"}

    # 3. Whisper (downloads audio — most expensive, most likely to be blocked)
    try:
        logger.info(f"Path 3: Whisper for {video_id}")
        segments = fetch_whisper_transcript(video_id)
        return {"video_id": video_id, "segments": segments, "source": "whisper"}
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Whisper failed: {err_msg}")

        if is_bot_detection(err_msg):
            detail = (
                "YouTube is blocking this server's IP. Options: "
                "(a) export fresh cookies.txt from a browser actively logged into YouTube "
                "and upload as a Render Secret File at /etc/secrets/cookies.txt, "
                "(b) set YT_PROXY env var to a residential proxy URL, "
                "(c) use videos with public captions (path 1 works on cached videos). "
                f"Upstream error: {err_msg}"
            )
        else:
            detail = f"Transcription failed: {err_msg}"
        raise HTTPException(status_code=500, detail=detail)
