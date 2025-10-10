import logging
import os

if os.getenv("KANI_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
