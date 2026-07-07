# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=1.26.0,<2.0",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
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
    from datasets import load_dataset
    import huggingface_hub
    from peft import LoraConfig, TaskType

    from big5.globalvars import (
        HF_DATASET,
        HF_USER,
        MODEL,
        MODEL_NAME,
        WANDB_PROJECT,
        BIG5_TRAITS,
    )

    return (
        BIG5_TRAITS,
        Dense,
        HF_DATASET,
        HF_USER,
        LoraConfig,
        MODEL,
        MODEL_NAME,
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
        """
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
        """
        SOFT_CLASS_WT = 0.7  # weight given to 2/5 and 4/5 ratings
        df = df.with_columns(
            pl.col('level')  # 1 to 5
            .replace_strict(
                [1, 2, 3, 4, 5], [-1.0, -SOFT_CLASS_WT, 0.0, SOFT_CLASS_WT, 1.0]
            )  # -1 to 1
            .alias('score')
        )
        if with_nans:
            df = df.with_columns(
                (pl.col('trait') == trait)
                .cast(float)
                .replace(0.0, float('nan'))
                .mul(pl.col('score'))  # -1 to 1 for this trait, nan for other traits
                .add(1)
                .truediv(2)  # 0 to 1 for this trait, nan for other traits
                .replace(float('nan'), -1)  # 0-1 for this trait, -1 for other traits
                .alias(f'trait_{trait}')
                for trait in BIG5_TRAITS
            )
        else:
            df = df.with_columns(
                (pl.col('trait') == trait)
                .cast(int)
                .mul(pl.col('score'))  # -1 to 1 for this trait, 0 for other traits
                .add(1)
                .truediv(2)  # 0 to 1 for this trait, 0.5 for other traits
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Build model
    """)
    return


@app.cell
def _(
    Dense,
    HF_DATASET,
    HF_USER,
    MODEL,
    Pooling,
    SentenceTransformer,
    SentenceTransformerModelCardData,
    Transformer,
    nn,
):
    def build_model_from_tf(base_model, peft_config=None):
        """Instantiate a new untrained model"""
        word_embedding_module = Transformer(base_model, max_seq_length=512)

        pooling_module = Pooling(
            word_embedding_module.get_embedding_dimension(), pooling_mode='cls'
        )

        dense_module = Dense(
            in_features=word_embedding_module.get_embedding_dimension(),
            out_features=5,
            activation_function=nn.Identity(),
        )

        model = SentenceTransformer(
            modules=[word_embedding_module, pooling_module, dense_module],
            model_card_data=SentenceTransformerModelCardData(
                language=['en', 'es'],
                model_name=f'Big-5 personality scorer based on {base_model}',
                model_id=f'{HF_USER}/big5-distilbert-lora',
                base_model=base_model,
                train_datasets=[{'id': HF_DATASET}],
                eval_datasets=[{'id': HF_DATASET}],
                task_name='feature extraction',
                tags=['feature-extraction'],
            ),
        )
        # model_card_data.set_base_model(base_model)

        if peft_config:
            model.add_adapter(peft_config)

        return model


    def build_model_from_st(base_model, model_id=None, peft_config=None):
        model_card_data = SentenceTransformerModelCardData(
            language=['en', 'es'],
            model_name='lora-finetuned test model',
            model_id=MODEL,
            train_datasets=[{'id': HF_DATASET}],
            eval_datasets=[{'id': HF_DATASET}],
            task_name='feature extraction',
            tags=['feature-extraction'],
        )

        model = SentenceTransformer(base_model, model_card_data=model_card_data)
        model.append(Dense(
            in_features=model.get_embedding_dimension(),
            out_features=5,
            activation_function=nn.Identity(),
        ))
        if peft_config:
            model.add_adapter(peft_config)
        # model.model_card_data = model_card_data
        return model


    def build_model(base_model: str, **kwargs):
        TF_MODELS = ('distilbert/distilbert-base-multilingual-cased', )
        ST_MODELS = ('google/embeddinggemma-300m', 'microsoft/harrier-oss-v1-0.6b')
        if base_model in TF_MODELS:
            return build_model_from_tf(base_model, **kwargs)
        elif base_model in ST_MODELS:
            return build_model_from_st(base_model, **kwargs)
        else:
            raise ValueError(f'not sure if "{base_model}" is a Transformer or Sentence Transformer')

    return (build_model,)


@app.cell
def _(SentenceTransformer, nn, torch):
    class MultiLabelBCEWithLogitsLoss(nn.Module):
        def __init__(self, model: SentenceTransformer):
            super(MultiLabelBCEWithLogitsLoss, self).__init__()
            self.model = model
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        def forward(
            self, sentence_features: list[dict[str, torch.Tensor]], labels: torch.Tensor
        ):
            # sentence_features is a list containing the feature dict for each input text
            # For standard classification/regression tasks in ST, we take the first element
            outputs = self.model(sentence_features[0])
            logits = outputs['sentence_embedding']

            # labels expected shape: (batch_size, 5)
            return self.criterion(
                logits[labels >= 0], labels[labels >= 0].float()
            ).nanmean()

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
    LoraConfig,
    MultiLabelBCEWithLogitsLoss,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    TaskType,
    WANDB_PROJECT,
    base_model,
    build_model,
    ds,
    ds_with_nans,
    torch,
    wandb,
):
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'eval/loss', 'goal': 'minimize'},
        'parameters': {
            'lora_r': {'values': [16, 32, 64, 128]},
            'lora_alpha': {'values': [32, 64, 128, 256]},
            'lora_dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.2,
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-4,
            },
            'target_modules': {
                # all-linear is supposedly better (thinkingmachines.ai/blog/lora/)
                'values': ['all-linear']
                # 'values': ['attention_only', 'all-linear']
            },
            'dataset': {'values': ['nans']},
            # 'dataset': {'values': ['orig', 'nans']},
            'base_model': {'values': [
                # 'distilbert/distilbert-base-multilingual-cased',
                'google/embeddinggemma-300m',
                # 'microsoft/harrier-oss-v1-0.6b',
            ]},
        },
    }

    def sweep_params():
        with wandb.init():
            config = wandb.config
            config_ds = ds if config.dataset == 'orig' else ds_with_nans
            if config.target_modules == 'attention_only':
                if config.base_model == 'distilbert/distilbert-base-multilingual-cased':
                    config_modules = ['q_lin', 'k_lin', 'v_lin', 'out_lin']
                elif config.base_model in ('google/embeddinggemma-300m', 'microsoft/harrier-oss-v1-0.6b'):
                    config_modules = ['k_proj', 'o_proj', 'q_proj', 'v_proj']
                else:
                    raise ValueError(f'unknown model: {base_model}')
            else:
                config_modules = 'all-linear'

            model = build_model(
                config.base_model,
                peft_config=LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=config_modules,
                )
            )

            trainer_args = SentenceTransformerTrainingArguments(
                output_dir='param_sweep_results',
                learning_rate=config.learning_rate,
                num_train_epochs=3,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                eval_strategy='epoch',
                # eval_steps=0.5,
                logging_steps=0.2,
                report_to='wandb',
                run_name=wandb.run.name,  # Sync HF run name with W&B UI
            )

            trainer = SentenceTransformerTrainer(
                model=model,
                args=trainer_args,
                train_dataset=config_ds['train'],
                eval_dataset=config_ds['test'],
                loss=MultiLabelBCEWithLogitsLoss(model),
            )
            trainer.train()

        # cleanup
        del model, trainer
        torch.cuda.empty_cache()

    # Initialize and run the sweep experiment
    # adjust count based on alloted compute time
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=sweep_params, count=10)
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

    train_config = {
        k: v for k, v in best_config.items() if k in sweep_config['parameters']
    }
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
    LoraConfig,
    MODEL_NAME,
    MultiLabelBCEWithLogitsLoss,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    TaskType,
    build_model,
    train_config,
    trainer_ds,
    wandb,
):
    model = build_model(
        train_config['base_model'],
        peft_config=LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=train_config['lora_r'],
            lora_alpha=train_config['lora_alpha'],
            lora_dropout=train_config['lora_dropout'],
            target_modules=train_config['target_modules'],
        )
    )

    trainer_args = SentenceTransformerTrainingArguments(
        num_train_epochs=20,
        auto_find_batch_size=True,
        warmup_steps=0.1,
        learning_rate=train_config['learning_rate'],
        metric_for_best_model='eval_loss',
        report_to='wandb',
        logging_strategy='epoch',
        eval_strategy='epoch',
        save_strategy='best',
        output_dir=MODEL_NAME,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=trainer_ds['train'],
        eval_dataset=trainer_ds['test'],
        loss=MultiLabelBCEWithLogitsLoss(model),
    )
    with wandb.init(reinit='finish_previous'):
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
def _(MODEL_NAME, model, trainer):
    model.save_pretrained(f'models/{MODEL_NAME}')
    trainer.push_to_hub(
        commit_message='End of training (switch to gemma3 base model)', revision='main'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load
    """)
    return


@app.cell
def _(MODEL, SentenceTransformer):
    def load_model():
        model = SentenceTransformer(MODEL)
        return model

    return (load_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inference
    """)
    return


@app.cell
def _(BIG5_TRAITS, load_model, pl):
    test_samples = [
        'I love meeting new people and being the center of attention.',
        "Sometimes I feel like I'm being watched...",
        "I'm so ADHD brained teehee",
        'I have a very short temper',
        "I'll try anything once ;)'",
        "I'm very opinionated and like to argue with others",
        'I prefer cozy cafes over loud clubs',
        'Where the party at?!'
    ]

    def test_inference(model, test_samples):
        embeddings = model.encode(test_samples)
        results_df = pl.DataFrame(embeddings, schema=BIG5_TRAITS)
        results_df = results_df.insert_column(0, pl.Series('text', test_samples))
        return results_df

    test_inference(load_model(), test_samples)
    return


if __name__ == "__main__":
    app.run()
