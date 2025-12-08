import logging
import os
import random

if os.getenv("KANI_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# random prompts
PROMPTS = [
    "Tell me about the Boeing 737.",
    "Without using the Shinkansen, how do I get from Oku-Tama to Komagome?",
    "Help me come up with a new magic item for D&D called the Blade of Kani.",
    "How do I set up vLLM?",
    "Please output as many of the letter 'a' as possible.",
    "How many 'a's are in the word 'strawberry'?",
]
SEED = random.randint(0, 99999)
