# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=1.26.0,<2.0",
# ]
# ///
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import polars.selectors as cs

    import torch
    from torch import nn

    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        SentenceTransformerModelCardData,
    )
    from sentence_transformers.base.modules import Transformer, Dense
    from sentence_transformers.sentence_transformer.modules import Pooling

    import wandb
    from datasets import load_dataset, load_dataset_builder
    import huggingface_hub
    from peft import LoraConfig, TaskType

    from big5.globalvars import (
        HF_USER,
        HF_DATASET,
        BASE_MODEL_NAME,
        BASE_MODEL,
        BASE_MODEL_ARCH,
        MODEL_NAME,
        LORA_MODEL_NAME,
        MODEL,
        LORA_MODEL,
        WANDB_PROJECT,
        BIG5_TRAITS,
        BIG5_TRAITS_SHORT,
        BIG_TRAITS_LETTERS,
    )

    return (
        BASE_MODEL,
        BIG5_TRAITS,
        Dense,
        HF_DATASET,
        LORA_MODEL,
        LORA_MODEL_NAME,
        LoraConfig,
        MODEL,
        Pooling,
        SentenceTransformer,
        SentenceTransformerModelCardData,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        TaskType,
        Transformer,
        WANDB_PROJECT,
        cs,
        huggingface_hub,
        load_dataset,
        nn,
        pl,
        torch,
        wandb,
    )


@app.cell
def _(huggingface_hub):
    huggingface_hub.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load dataset
    """)
    return


@app.cell
def _(BIG5_TRAITS, HF_DATASET, cs, load_dataset, pl):
    def preprocess_dset(df, with_nans=False):
        '''
        preprocess dataset 1 in polars

        input columns:
            trait: trait name (one of BIG5_TRAITS)
            level: 1 to 5 score
            description: personality trait description

        output columns:
            sentence: (unchanged)
            label: length-5 array of scores
                all scores are 0 except the target class which ranges from -1 to 1
            trait: (unchanged)
        '''
        SOFT_CLASS_WT = 0.7 # weight given to 2/5 and 4/5 ratings
        df = df.with_columns(
            pl.col('level') # 1 to 5
            .replace_strict([1, 2, 3, 4, 5], [-1.0, -SOFT_CLASS_WT, 0.0, SOFT_CLASS_WT, 1.0]) # -1 to 1
            .alias('score')
        )
        if with_nans:
            df = df.with_columns(
                (pl.col('trait') == trait).cast(float)
                .replace(0.0, float('nan'))
                .mul(pl.col('score')) # -1 to 1 for this trait, nan for other traits
                .add(1).truediv(2) # 0 to 1 for this trait, nan for other traits
                .replace(float('nan'), -1) # 0-1 for this trait, -1 for other traits
                .alias(f'trait_{trait}')
                for trait in BIG5_TRAITS
            )
        else:
            df = df.with_columns(
                (pl.col('trait') == trait).cast(int)
                .mul(pl.col('score')) # -1 to 1 for this trait, 0 for other traits
                .add(1).truediv(2) # 0 to 1 for this trait, 0.5 for other traits
                .alias(f'trait_{trait}')
                for trait in BIG5_TRAITS
            )
        df = (
            df.with_columns(label=pl.concat_arr(cs.starts_with('trait_')))
            .rename({'description': 'sentence'})
            .select('sentence', 'label', 'trait')
            # .to_arrow()
            .to_pandas()
        )
        return df


    ds = (
        load_dataset(HF_DATASET, split='all')
        .with_format('polars')
        .map(preprocess_dset, batched=True)
        .with_format(None)
        .class_encode_column('trait')  # ty:ignore[unresolved-attribute]
        .train_test_split(test_size=0.1, stratify_by_column='trait')  # ty:ignore[unresolved-attribute]
        .remove_columns(['trait'])
    )

    ds_with_nans = (
        load_dataset(HF_DATASET, split='all')
        .with_format('polars')
        .map(lambda d: preprocess_dset(d, with_nans=True), batched=True)
        .with_format(None)
        .class_encode_column('trait')  # ty:ignore[unresolved-attribute]
        .train_test_split(test_size=0.1, stratify_by_column='trait')  # ty:ignore[unresolved-attribute]
        .remove_columns(['trait'])
    )
    return ds, ds_with_nans


@app.cell
def _(
    BASE_MODEL,
    Dense,
    HF_DATASET,
    MODEL,
    Pooling,
    SentenceTransformer,
    SentenceTransformerModelCardData,
    Transformer,
    nn,
):
    def build_model(peft_config=None):
        '''Instantiate a new untrained model'''
        # Start with the pretrained base model (DistilBERT)
        word_embedding_module = Transformer(BASE_MODEL, max_seq_length=512)


        # Add a pooling layer
        # 'cls' pooling uses the CLS token which is the last hidden state's 1st token
        pooling_module = Pooling(
            word_embedding_module.get_embedding_dimension(),
            pooling_mode='cls'
        )

        # Add a dense layer that outputs a size-5 vector
        dense_module = Dense(
            in_features=word_embedding_module.get_embedding_dimension(),
            out_features=5,
            activation_function=nn.Identity() # Logits are returned directly
        )

        model = SentenceTransformer(
            modules=[word_embedding_module, pooling_module, dense_module],
            model_card_data=SentenceTransformerModelCardData(
                language=["en", "es"],
                model_name="distilBERT-based Big-5 personality scorer",
                model_id=MODEL,
                train_datasets=[{'id': HF_DATASET}],
                eval_datasets=[{'id': HF_DATASET}],
                task_name='feature extraction',
                tags=['feature-extraction'],
            )
        )
        if peft_config:
            model.add_adapter(peft_config)
        return model


    return (build_model,)


@app.cell
def _(SentenceTransformer, nn, torch):
    class MultiLabelBCEWithLogitsLoss(nn.Module):
        def __init__(self, model: SentenceTransformer):
            super(MultiLabelBCEWithLogitsLoss, self).__init__()
            self.model = model
            # self.criterion = nn.BCEWithLogitsLoss()
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, sentence_features: list[dict[str, torch.Tensor]], labels: torch.Tensor):
            # sentence_features is a list containing the feature dict for each input text
            # For standard classification/regression tasks in ST, we take the first element
            outputs = self.model(sentence_features[0])
            logits = outputs['sentence_embedding']

            # labels expected shape: (batch_size, 5)
            # return self.criterion(logits, labels.float())
            # return self.criterion(logits, labels.float()).nanmean()
            return self.criterion(logits[labels >= 0], labels[labels >= 0].float()).nanmean()


    return (MultiLabelBCEWithLogitsLoss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter sweeep
    """)
    return


@app.cell
def _(
    WANDB_PROJECT,
    LoraConfig,
    MultiLabelBCEWithLogitsLoss,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    TaskType,
    build_model,
    ds,
    ds_with_nans,
    torch,
    wandb,
):
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'eval/loss',
            'goal': 'minimize'
        },
        'parameters': {
            'r': {
                'values': [16, 32, 64, 128]
            },
            'lora_alpha': {
                'values': [32, 64, 128, 256]
            },
            'lora_dropout': {
                'values': [0.0, 0.05, 0.1]
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 3e-3,
            },
            'target_modules': {
                # all-linear is supposedly better (thinkingmachines.ai/blog/lora/)
                # but trying both methods anyway
                 'values': ['attention_only', 'all-linear']
            },
            'dataset': {
                'values': ['orig', 'nans']
            }
        }
    }

    def sweep_params():
        # Initialize the sweep run
        with wandb.init() as run:
            config = wandb.config
            config_modules = ['q_lin', 'k_lin', 'v_lin', 'out_lin'] \
                if config.target_modules == 'attention_only' else "all-linear"
            config_ds = ds if config.dataset == 'orig' else ds_with_nans

            model = build_model(peft_config=LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config_modules
            ))

            trainer_args = SentenceTransformerTrainingArguments(
                output_dir="param_sweep_results",
                learning_rate=config.learning_rate,
                num_train_epochs=1,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                eval_strategy="steps",
                eval_steps=0.5,
                logging_steps=0.2,
                report_to="wandb",     # Turn W&B logging ON
                run_name=wandb.run.name # Sync HF run name with W&B UI # ty:ignore[unresolved-attribute]
            )

            trainer = SentenceTransformerTrainer(
                model=model,
                args=trainer_args,
                train_dataset=config_ds['train'],
                eval_dataset=config_ds['test'],
                loss=MultiLabelBCEWithLogitsLoss(model)
            )
            trainer.train()

        # cleanup
        del model, trainer
        torch.cuda.empty_cache()


    # Initialize and run the sweep experiment
    # adjust count based on alloted compute time
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=sweep_params, count=20)
    return sweep_config, sweep_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get the best params
    """)
    return


@app.cell
def _(WANDB_PROJECT, ds, ds_with_nans, sweep_config, sweep_id, wandb):
    api = wandb.Api()

    sweep = api.sweep(f'ola-owo/{WANDB_PROJECT}/sweeps/{sweep_id}')
    best_run = sweep.best_run()
    best_config = best_run.config

    train_config = {k:v for k,v in best_config.items() if k in sweep_config['parameters']}
    train_config['lora_r'] = train_config.pop('r')
    trainer_ds = ds_with_nans if train_config['dataset'] == 'nans' else ds

    train_config
    return train_config, trainer_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training loop
    """)
    return


@app.cell
def _(
    LORA_MODEL_NAME,
    LoraConfig,
    MultiLabelBCEWithLogitsLoss,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    TaskType,
    build_model,
    train_config,
    trainer_ds,
):
    model = build_model(peft_config=LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=train_config['lora_r'],
        lora_alpha=train_config['lora_alpha'],
        lora_dropout=train_config['lora_dropout'],
        target_modules=train_config['target_modules'],
    ))

    trainer_args = SentenceTransformerTrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=0.1,
        learning_rate=train_config['learning_rate'],
        metric_for_best_model='eval_loss',
        report_to='wandb',
        hub_revision='v2',
        logging_strategy='epoch',
        eval_strategy="epoch",
        save_strategy="best",
        output_dir=LORA_MODEL_NAME,
        run_name=LORA_MODEL_NAME,  # Will be used in W&B if `wandb` is installed
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=trainer_ds['train'],
        eval_dataset=trainer_ds['test'],
        loss=MultiLabelBCEWithLogitsLoss(model)
    )
    trainer.train()
    return model, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Save/load the model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save
    """)
    return


@app.cell
def _(LORA_MODEL_NAME, model, trainer):
    model.save_pretrained(LORA_MODEL_NAME)
    trainer.push_to_hub(commit_message='End of training (new dset with nans)', revision='main')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _(LORA_MODEL, SentenceTransformer):
    def load_lora():
        model = SentenceTransformer(LORA_MODEL)
        return model

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inference
    """)
    return


@app.cell
def _(BIG5_TRAITS, model, pl):
    test_samples = [
        "I love meeting new people and being the center of attention.",
        "Sometimes I feel like I'm being watched...",
        "I'm so ADHD brained teehee",
        "I have a very short temper",
        "I'll try anything once ;)'",
        "I'm very opinionated and like to argue with others",
        "I prefer cozy cafes over loud clubs",
    ]


    def test_inference(samples):
        embeddings = model.encode(test_samples)
        results_df = pl.DataFrame(embeddings, schema=BIG5_TRAITS)
        results_df = results_df.insert_column(0, pl.Series("text", test_samples))
        return results_df

    test_inference(test_samples)
    return


if __name__ == "__main__":
    app.run()
