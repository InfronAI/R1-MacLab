"""Record real CLI output and render a portable MP4 terminal replay (not a fabricated screen capture)."""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="experiments/m1-final-seed42")
    parser.add_argument("--output", default="assets/demo")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("Install ffmpeg first (brew install ffmpeg). Training does not require it.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, "-u", "-m", "open_r1.local.cli"]
    commands = [
        prefix + ["doctor"],
        prefix + ["verify", "--run", args.run],
        [
            sys.executable,
            "-u",
            "-c",
            "import json,pathlib; r=pathlib.Path(__import__('sys').argv[1]); m=json.loads((r/'manifest.json').read_text()); "
            "print('Run status:', m['status']); "
            "print('CHECKPOINT         PASS@1   FORMAT'); "
            "[print(f\"{x['stage']:18} {x['pass_at_1']:6.1%}   {x['format_rate']:6.1%}\") "
            "for x in json.loads((r/'summary.json').read_text())]",
            args.run,
        ],
        prefix
        + [
            "generate",
            "--run",
            args.run,
            "--stage",
            "alignment_rl",
            "--greedy",
            "--prompt",
            "Question: Compute 2 + 3 . Answer:",
        ],
        prefix
        + [
            "generate",
            "--run",
            args.run,
            "--stage",
            "distillation",
            "--greedy",
            "--prompt",
            "Question: Write expression for x plus 3 . Answer:",
        ],
    ]
    events, receipts = [], []
    started = time.monotonic()
    for command in commands:
        displayed = ["python"] + command[1:]
        events.append([time.monotonic() - started, "o", "$ " + shlex.join(displayed) + "\n"])
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            events.append([time.monotonic() - started, "o", line])
        code = proc.wait()
        receipts.append({"argv": displayed, "returncode": code})
        if code:
            raise RuntimeError(f"Demo command failed with exit {code}; refusing to render a success demo")
        events.append([time.monotonic() - started, "o", "\n"])
    header = {
        "version": 2,
        "width": 104,
        "height": 28,
        "timestamp": int(time.time()),
        "title": "Open R1 · real M1 CLI recording",
    }
    with (output / "session.cast").open("w") as f:
        f.write(json.dumps(header) + "\n")
        for event in events:
            f.write(json.dumps(event) + "\n")
    (output / "commands.json").write_text(json.dumps(receipts, indent=2) + "\n")

    font_paths = ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    font_path = next((p for p in font_paths if Path(p).exists()), None)
    font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default(size=18)
    title_font = ImageFont.truetype(font_path, 25) if font_path else ImageFont.load_default(size=25)
    width, height, fps = 1280, 800, 10
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "open-r1-m1.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    screen, frames, poster = [], 0, None

    def frame(seconds):
        nonlocal frames, poster
        picture = Image.new("RGB", (width, height), "#09151f")
        draw = ImageDraw.Draw(picture)
        draw.rounded_rectangle(
            (24, 20, width - 24, height - 25), radius=16, fill="#112532", outline="#275169", width=2
        )
        draw.text((48, 39), "OPEN R1 / APPLE SILICON", font=title_font, fill="#74e0c1")
        draw.text(
            (48, 78), "Actual CLI output · recorded on M1 Pro · synthetic-scale experiment", font=font, fill="#b7cbd9"
        )
        draw.line((48, 111, width - 48, 111), fill="#275169", width=2)
        for i, line in enumerate(screen[-24:]):
            draw.text((48, 132 + i * 24), line, font=font, fill="#74e0c1" if line.startswith("$") else "#e2edf4")
        draw.text(
            (48, 738),
            "Training / sampling / verification run locally. No paper-score claim.",
            font=font,
            fill="#9aafbd",
        )
        for _ in range(max(1, int(seconds * fps))):
            encoder.stdin.write(picture.tobytes())
            frames += 1
        poster = picture

    previous = 0.0
    for timestamp, _, text in events:
        # Cap idle gaps, and add readable dwell time. The .cast preserves unmodified wall-clock timing.
        if screen:
            frame(min(1.0, max(0.0, timestamp - previous)))
        for line in text.splitlines():
            screen.extend(textwrap.wrap(line, 104, replace_whitespace=False) or [""])
            frame(0.3 if not line.startswith("$") else 1.0)
        previous = timestamp
    frame(4)
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("ffmpeg failed")
    poster.save(output / "poster.png")
    (output / "recording.json").write_text(
        json.dumps(
            {
                "source": "real subprocess stdout/stderr",
                "run": args.run,
                "recorded_seconds": time.monotonic() - started,
                "video_seconds": frames / fps,
                "timing": "Terminal replay; idle gaps capped and per-line reading time added. session.cast retains raw timing.",
            },
            indent=2,
        )
    )
    print(output / "open-r1-m1.mp4")


if __name__ == "__main__":
    main()
