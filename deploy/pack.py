"""Make crypto-bot-upload.zip: code + your paper account + price history. Never includes .env."""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "crypto-bot-upload.zip"
skip = {".env", "heartbeat", "KILL"}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ["app", "ui", "deploy", "data", "dashboard.py", "requirements.txt", ".env.example"]:
        p = root / top
        for f in ([p] if p.is_file() else p.rglob("*")):
            if f.is_file() and f.name not in skip and "__pycache__" not in f.parts and "pit" not in f.parts and f.suffix != ".log" and not f.name.endswith("_1h.parquet"):
                z.write(f, f.relative_to(root).as_posix())  # forward slashes, so Linux unzips it cleanly
print(f"Created {out} ({out.stat().st_size / 1e6:.0f} MB)")
