# run_app.py  (FULL REPLACEMENT)

import os
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv
from streamlit.web import cli as stcli

def main():
    # --- Resolve roots ---
    if getattr(sys, "frozen", False):
        app_root = Path(sys.executable).parent          # dist/CutlerPlatformDebug/
        internal_root = app_root / "_internal"          # dist/CutlerPlatformDebug/_internal
    else:
        app_root = Path(__file__).resolve().parent
        internal_root = app_root

    # Ensure internal_root is importable
    if str(internal_root) not in sys.path:
        sys.path.insert(0, str(internal_root))

    # --- Writable Streamlit config ---
    user_home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    user_cfg_dir = user_home / "CutlerPlatform" / ".streamlit"
    user_cfg_dir.mkdir(parents=True, exist_ok=True)
    user_cfg_file = user_cfg_dir / "config.toml"

    if not user_cfg_file.exists():
        user_cfg_file.write_text(
            "[general]\n"
            "email = \"\"\n\n"
            "[global]\n"
            "developmentMode = false\n",
            encoding="utf-8",
        )

    os.environ["STREAMLIT_CONFIG_FILE"] = str(user_cfg_file)

    # --- Logging for early crashes ---
    log_dir = user_home / "CutlerPlatform" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    boot_log = log_dir / "bootstrap.log"

    def log_boot(msg: str):
        try:
            boot_log.write_text(msg, encoding="utf-8")
        except Exception:
            pass

    # --- Load .env BEFORE Streamlit starts ---
    env1 = app_root / ".env"
    env2 = internal_root / ".env"
    if env1.exists():
        load_dotenv(env1)
    elif env2.exists():
        load_dotenv(env2)

    # --- Playwright browsers path ---
    pw1 = app_root / "ms-playwright"
    pw2 = internal_root / "ms-playwright"
    if pw1.exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(pw1)
    elif pw2.exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(pw2)

    # Run from _internal so relative paths match local dev
    os.chdir(internal_root)

    final_py = internal_root / "final.py"
    if not final_py.exists():
        raise FileNotFoundError(f"final.py not found at {final_py}")

    # Extra stability for frozen apps
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    sys.argv = [
        "streamlit",
        "run",
        str(final_py),
        "--server.headless=false",
        "--global.developmentMode=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
    ]

    try:
        sys.exit(stcli.main())
    except BaseException:
        log_boot(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
