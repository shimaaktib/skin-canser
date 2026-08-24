"""Optional Supabase persistence integration.

Inference and the local UI remain usable without Supabase credentials.  History persistence is
explicitly reported as unavailable until the three Supabase environment variables are configured.
"""

import logging
import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()
logger = logging.getLogger("supabase_client")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "lesion-images")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        logger.info("Supabase persistence enabled.")
    except Exception:
        logger.exception("Supabase credentials were present but the client failed to initialize.")
else:
    logger.warning(
        "Supabase persistence disabled: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to enable history and result saving."
    )
