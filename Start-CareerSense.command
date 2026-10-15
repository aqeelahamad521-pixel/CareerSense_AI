#!/bin/bash

# Double-click launcher for macOS.
# CareerSense model artifacts require Python 3.11.x.

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

fail() {
    printf "\nCareerSense could not start.\n%s\n\n" "$1"
    read -r -p "Press Return to close this window..."
    exit 1
}

printf "\nCareerSense\n"
printf "Preparing the local application...\n\n"

if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
    fail "The project is incomplete. Keep this launcher beside app.py and requirements.txt."
fi

PYTHON_BIN=""
for candidate in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    fail "Python 3.11 was not found. Install Python 3.11 from https://www.python.org/downloads/ and then double-click this file again."
fi

VENV_DIR="$PROJECT_DIR/.venv-mac"
VENV_PYTHON="$VENV_DIR/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    printf "Creating the private Python environment...\n"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || fail "The Python environment could not be created."
fi

if ! "$VENV_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' >/dev/null 2>&1; then
    fail "The existing .venv-mac environment is not Python 3.11. Delete that folder and run this launcher again."
fi

REQUIREMENTS_HASH="$(shasum -a 256 requirements.txt | awk '{print $1}')"
STAMP_FILE="$VENV_DIR/.requirements.sha256"
INSTALLED_HASH=""
if [ -f "$STAMP_FILE" ]; then
    INSTALLED_HASH="$(<"$STAMP_FILE")"
fi

if [ "$REQUIREMENTS_HASH" != "$INSTALLED_HASH" ]; then
    printf "Installing the exact application dependencies (first run may take a few minutes)...\n"
    "$VENV_PYTHON" -m pip install --disable-pip-version-check -r requirements.txt || \
        fail "Dependencies could not be installed. Check your internet connection and try again."
    printf "%s\n" "$REQUIREMENTS_HASH" > "$STAMP_FILE"
fi

printf "Starting CareerSense at http://localhost:8501\n"
printf "Keep this window open while using the application. Press Control-C here to stop it.\n\n"

"$VENV_PYTHON" -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false
EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ] && [ "$EXIT_CODE" -ne 130 ]; then
    fail "The application stopped with exit code $EXIT_CODE. Review the messages above for details."
fi

printf "\nCareerSense stopped.\n"
