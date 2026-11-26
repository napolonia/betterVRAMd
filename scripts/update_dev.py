
from datetime import datetime
from pathlib import Path

DEV_FILE = Path(__file__).resolve().parents[1] / "PROJECT_DEV.txt"

def update_timestamp():
    content = DEV_FILE.read_text(encoding="utf-8")

    timestamp = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    marker = "Last updated:"

    if marker in content:
        lines = content.splitlines()

        for index, line in enumerate(lines):
            if line.startswith(marker):
                lines[index] = f"{marker} {timestamp}"
                break

        content = "\n".join(lines) + "\n"
    else:
        content = f"{marker} {timestamp}\n\n{content}"

    DEV_FILE.write_text(content, encoding="utf-8")
    print(f"Updated: {timestamp}")


if __name__ == "__main__":
    update_timestamp()
