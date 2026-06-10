HF_USER = 'ola-owo'
HF_DATASET = 'ola-owo/big-five-personality-traits'

BASE_MODEL_NAME = 'distilbert-base-multilingual-cased'
BASE_MODEL = f'distilbert/{BASE_MODEL_NAME}'
BASE_MODEL_ARCH = 'DistilBertModel'

MODEL_NAME = 'distilbert-bigfive-sentence-transformer'
LORA_MODEL_NAME = MODEL_NAME + '-lora'
MODEL = f'{HF_USER}/{MODEL_NAME}'
LORA_MODEL = f'{HF_USER}/{LORA_MODEL_NAME}'

WANDB_PROJECT = 'LyricalMiracle'

BIG5_TRAITS = [
    'Openness',
    'Conscientiousness',
    'Extraversion',
    'Agreeableness',
    'Neuroticism',
]
BIG5_TRAITS_SHORT = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
BIG_TRAITS_LETTERS = list('OCEAN')
