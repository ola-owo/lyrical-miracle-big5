HF_USER = 'ola-owo'
HF_DATASET = 'ola-owo/big-five-personality-traits'

MODEL_NAME = 'big5-sentence-transformer-lora'
MODEL = f'{HF_USER}/{MODEL_NAME}'

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
