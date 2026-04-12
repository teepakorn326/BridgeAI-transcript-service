import os
import tempfile
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel
import yt_dlp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EduCaption Transcript Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once at startup (use "base" for speed, "small" for accuracy)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MAX_DURATION = int(os.getenv("MAX_DURATION", "900"))  # 15 minutes default
logger.info(f"Loading Whisper model: {WHISPER_MODEL}, max duration: {MAX_DURATION}s")
whisper_model = WhisperModel(WHISPER_MODEL, compute_type="int8")
logger.info("Whisper model loaded")


def fetch_youtube_transcript(video_id: str):
    """Try to get transcript from YouTube's built-in captions."""
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except Exception:
        transcript = ytt_api.fetch(video_id)

    segments = []
    for snippet in transcript:
        start = round(snippet.start, 2)
        if start >= MAX_DURATION:
            break
        segments.append(
            {
                "start_seconds": start,
                "end_seconds": min(round(snippet.start + snippet.duration, 2), MAX_DURATION),
                "text": snippet.text,
            }
        )
    return segments


def fetch_whisper_transcript(video_id: str):
    """Download audio and transcribe with Whisper as fallback."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # Use cookies to bypass YouTube bot detection on cloud IPs.
        # Looks for /etc/secrets/cookies.txt (Render secret file) or ./cookies.txt
        cookies_path = None
        for candidate in ("/etc/secrets/cookies.txt", "cookies.txt"):
            if os.path.exists(candidate):
                cookies_path = candidate
                logger.info(f"Using cookies from {candidate}")
                break

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "cookiefile": cookies_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "96",
                }
            ],
            "download_ranges": lambda info, ydl: [{"start_time": 0, "end_time": MAX_DURATION}],
            "quiet": True,
            "no_warnings": True,
        }

        logger.info(f"Downloading audio for {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp may add .mp3 extension
        if not os.path.exists(audio_path) and os.path.exists(audio_path + ".mp3"):
            audio_path = audio_path + ".mp3"

        logger.info(f"Transcribing with Whisper ({WHISPER_MODEL})")
        result_segments, _ = whisper_model.transcribe(audio_path)

        segments = []
        for seg in result_segments:
            if seg.start >= MAX_DURATION:
                break
            segments.append(
                {
                    "start_seconds": round(seg.start, 2),
                    "end_seconds": min(round(seg.end, 2), MAX_DURATION),
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
