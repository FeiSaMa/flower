"""用 Playwright 逐帧截图并合成训练动画 MP4/GIF。

帧目录固定在 /tmp/opencode/anim-frames，录完用 gio trash 清理。
"""

import os
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "report" / "training_animation.html"
OUTPUT_MP4 = ROOT / "report" / "figures" / "training_animation.mp4"
OUTPUT_GIF = ROOT / "report" / "figures" / "training_animation.gif"
FRAMES = Path("/tmp/opencode/anim-frames")


def run(command):
    print("+", " ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def cleanup_frames():
    if not FRAMES.exists():
        return
    subprocess.run(["gio", "trash", str(FRAMES)], check=False)
    if not FRAMES.exists():
        return
    # /tmp 不支持 gio 回收站；先移到 home，再交给 gio trash（仍不使用 rm -rf）。
    fallback = Path.home() / ".cache" / f"anim-frames-trash-{os.getpid()}"
    subprocess.run(["mv", str(FRAMES), str(fallback)], check=True)
    subprocess.run(["gio", "trash", str(fallback)], check=False)


def main():
    if not HTML.is_file():
        raise SystemExit(f"缺少 {HTML}，先运行 scripts/build_animation.py")
    OUTPUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    cleanup_frames()
    FRAMES.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
        page.goto(HTML.resolve().as_uri(), wait_until="load")
        page.wait_for_function("window.__ready === true", timeout=30000)
        page.evaluate("async () => { await document.fonts.ready; }")
        for index in range(300):
            page.evaluate("t => window.setTime(t)", index / 10)
            page.screenshot(path=str(FRAMES / f"frame_{index:04d}.png"))
        browser.close()

    run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-i",
            str(FRAMES / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(OUTPUT_MP4),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-i",
            str(FRAMES / "frame_%04d.png"),
            "-t",
            "18",
            "-vf",
            "fps=8,scale=960:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(OUTPUT_GIF),
        ]
    )

    cleanup_frames()
    print(f"wrote {OUTPUT_MP4} {OUTPUT_MP4.stat().st_size} bytes")
    print(f"wrote {OUTPUT_GIF} {OUTPUT_GIF.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
