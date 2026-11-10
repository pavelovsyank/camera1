"""Detect motion in the MediaMTX stream and save fixed-duration clips."""

import os

# Bound RTSP reads so a disconnected camera can be retried.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import logging
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime

import cv2


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
URL = os.environ.get("RTSP_URL", "rtsp://127.0.0.1:8554/cam1")
DURATION = int(os.environ.get("RECORD_SECONDS", "15"))
MIN_AREA = int(os.environ.get("MOTION_MIN_AREA", "1500"))
REQUIRED_FRAMES = int(os.environ.get("MOTION_FRAMES", "3"))
QUIET_SECONDS = float(os.environ.get("QUIET_SECONDS", "5"))
OUTPUT = Path("/recordings")
stopping = False


def stop(_signum, _frame):
    global stopping
    stopping = True


def start_recording():
    filename = OUTPUT / f"motion_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}.mkv"
    logging.info("Motion detected: starting ffmpeg for %s seconds to %s", DURATION, filename)
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-rtsp_transport", "tcp", "-timeout", "10000000", "-i", URL,
        "-t", str(DURATION), "-map", "0:v:0", "-map", "0:a?",
        "-c", "copy", "-n", str(filename),
    ])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    recording = None
    cap = None
    armed = True
    quiet_since = None
    try:
        while not stopping:
            if cap is None:
                cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
                ])
                background = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=25, detectShadows=False,
                )
                warmup_until = time.monotonic() + 5
                consecutive = 0
                quiet_since = None
            ok, frame = cap.read()
            if not ok:
                logging.warning("Stream unavailable; retrying in 3 seconds")
                cap.release()
                cap = None
                time.sleep(3)
                continue

            # Area threshold is measured on a frame scaled to 640 pixels wide.
            height = max(1, round(frame.shape[0] * 640 / frame.shape[1]))
            frame = cv2.resize(frame, (640, height))
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            mask = background.apply(gray)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            moving = any(cv2.contourArea(c) >= MIN_AREA for c in contours)
            now = time.monotonic()

            if recording is not None and recording.poll() is not None:
                logging.info("Recording finished, ffmpeg exit code: %s", recording.returncode)
                recording = None
                # Require a quiet interval after each clip before rearming.
                quiet_since = None

            if now < warmup_until:
                continue
            consecutive = consecutive + 1 if moving else 0
            if moving:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            if recording is None and quiet_since is not None and now - quiet_since >= QUIET_SECONDS:
                armed = True
            if armed and recording is None and consecutive >= REQUIRED_FRAMES:
                recording = start_recording()
                armed = False
    finally:
        if cap is not None:
            cap.release()
        if recording is not None and recording.poll() is None:
            # SIGINT lets ffmpeg finalize the container before Docker stops us.
            recording.send_signal(signal.SIGINT)
            try:
                recording.wait(timeout=5)
            except subprocess.TimeoutExpired:
                recording.kill()
                recording.wait()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
