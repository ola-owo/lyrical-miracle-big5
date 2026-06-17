"""Small script that loads the fine-tuned model and encodes some test inputs"""

# import huggingface_hub
import polars as pl
from sentence_transformers import SentenceTransformer

from globalvars import BIG5_TRAITS_SHORT, LORA_MODEL

# huggingface_hub.login()
model = SentenceTransformer(LORA_MODEL)

test_samples = [
    'I love meeting new people and being the center of attention.',
    'I often feel anxious and worry about small things.',
    'I have a very short temper',
    'I enjoy exploring abstract concepts and complex ideas.',
    "I'm very opinionated and often debate with others",
    'I prefer intimate spaces over large crowds',
]

embeddings = model.encode(test_samples)

results_df = pl.DataFrame(embeddings, schema=BIG5_TRAITS_SHORT)
results_df = results_df.insert_column(0, pl.Series('text', test_samples))

print(results_df)
