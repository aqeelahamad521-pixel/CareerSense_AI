import sys
import os
import asyncio
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from streamlit.web import cli as stcli

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false"
    ]
    sys.exit(stcli.main())
