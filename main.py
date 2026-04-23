import os
import shutil
import tempfile
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel
import yt_dlp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BridgeAI Transcript Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once at startup (use "base" for speed, "small" for accuracy)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MAX_DURATION_CAPTIONS = int(os.getenv("MAX_DURATION_CAPTIONS", "14400"))  # 4 hours — just text
MAX_DURATION_WHISPER = int(os.getenv("MAX_DURATION_WHISPER", "3600"))    # 1 hour — CPU-intensive
logger.info(f"Loading Whisper model: {WHISPER_MODEL}, max captions: {MAX_DURATION_CAPTIONS}s, max whisper: {MAX_DURATION_WHISPER}s")
whisper_model = WhisperModel(WHISPER_MODEL, compute_type="int8")
logger.info("Whisper model loaded")


def fetch_youtube_transcript(video_id: str):
    """Try to get transcript from YouTube's built-in captions."""
    # Find cookies for authentication
    cookies_path = None
    for candidate in ("/etc/secrets/cookies.txt", "cookies.txt"):
        if os.path.exists(candidate):
            cookies_path = candidate
            logger.info(f"Using cookies from {candidate} for YouTubeTranscriptApi")
            break

    try:
        # YouTubeTranscriptApi.get_transcript is the standard way to fetch captions
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=["en", "en-US", "en-GB"], 
                cookies=cookies_path
            )
        except Exception:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                cookies=cookies_path
            )
    except Exception as e:
        logger.error(f"YouTubeTranscriptApi failed: {e}")
        raise e

    segments = []
    for snippet in transcript:
        start = round(snippet["start"], 2)
        if start >= MAX_DURATION_CAPTIONS:
            break
        segments.append(
            {
                "start_seconds": start,
                "end_seconds": min(round(snippet["start"] + snippet["duration"], 2), MAX_DURATION_CAPTIONS),
                "text": snippet["text"],
            }
        )
    return segments


def fetch_whisper_transcript(video_id: str):
    """Download audio and transcribe with Whisper as fallback."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # Use cookies to bypass YouTube bot detection on cloud IPs.
        # Looks for /etc/secrets/cookies.txt (Render Secret Files) or ./cookies.txt.
        # yt-dlp writes refreshed session cookies back to this file, so the
        # source must be copied to a writable location — Render's /etc/secrets
        # is a read-only mount.
        cookies_path = None
        for candidate in ("/etc/secrets/cookies.txt", "cookies.txt"):
            if os.path.exists(candidate):
                writable = os.path.join(tmpdir, "cookies.txt")
                shutil.copy2(candidate, writable)
                cookies_path = writable
                logger.info(f"Copied cookies from {candidate} to {writable}")
                break

        # Using ios + android player clients bypasses most of YouTube's
        # "sign in to confirm you're not a bot" challenges on cloud IPs —
        # those clients use different internal APIs that don't enforce the
        # same checks as the web player.
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "cookiefile": cookies_path,
            "extractor_args": {
                "youtube": {
                    "player_client": ["ios", "android", "mweb"],
                }
            },
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
                ),
            },
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "96",
                }
            ],
            "download_ranges": lambda info, ydl: [{"start_time": 0, "end_time": MAX_DURATION_WHISPER}],
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
        }

        # Opt-in residential proxy when cloud IPs are entirely blocklisted.
        # Set YT_PROXY=http://user:pass@host:port on Render if you hit this.
        proxy = os.getenv("YT_PROXY")
        if proxy:
            ydl_opts["proxy"] = proxy
            logger.info("Using YT_PROXY for yt-dlp")

        logger.info(f"Downloading audio for {video_id} (player_client=android,ios,web)")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp may add .mp3 extension
        if not os.path.exists(audio_path) and os.path.exists(audio_path + ".mp3"):
            audio_path = audio_path + ".mp3"

        logger.info(f"Transcribing with Whisper ({WHISPER_MODEL})")
        result_segments, _ = whisper_model.transcribe(audio_path)

        segments = []
        for seg in result_segments:
            if seg.start >= MAX_DURATION_WHISPER:
                break
            segments.append(
                {
                    "start_seconds": round(seg.start, 2),
                    "end_seconds": min(round(seg.end, 2), MAX_DURATION_WHISPER),
                    "text": seg.text.strip(),
                }
            )
    return segments


@app.get("/health")
def health():
    return {"status": "healthy", "service": "transcript-service"}


@app.get("/transcript/{video_id}")
def get_transcript(video_id: str):
    # Try YouTube captions first
    try:
        logger.info(f"Trying YouTube captions for {video_id}")
        segments = fetch_youtube_transcript(video_id)
        logger.info(f"Got {len(segments)} segments from YouTube captions")
        return {"video_id": video_id, "segments": segments, "source": "youtube"}
    except Exception as e:
        logger.info(f"YouTube captions unavailable: {e}")

    # Fall back to Whisper
    try:
        logger.info(f"Falling back to Whisper for {video_id}")
        segments = fetch_whisper_transcript(video_id)
        logger.info(f"Got {len(segments)} segments from Whisper")
        return {"video_id": video_id, "segments": segments, "source": "whisper"}
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not transcribe video '{video_id}': {e}",
        )
