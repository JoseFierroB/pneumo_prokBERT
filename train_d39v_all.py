#!/usr/bin/env python3
"""
Finetuning D39V ordenado - 3 datasets × (default + optuna 20)
- Fidelidad oficial: ProkBERTTokenizer k6s1 + AutoModelForSequenceClassification (HF) + Trainer
- Cambios minimos: dataset CSVs propios 81bp, num_labels segun tarea, optuna 20 + default
- Nombres claros y concisos, 5090 solo cuda:0 fp16 auto

Datasets:
  d39v_3way_SigA_SigX_None      973  (681/146/146) 3way 0:SigA 1:SigX 2:None
  d39v_binary_SigA_vs_ComX      411  (287/62/62)  binary 0:SigA 1:ComX(SigX)
  d39v_binary_SigA_vs_Neg      1390  (973/208/209) binary 0:Neg 1:SigA
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import sys
sys.path.insert(0,".")
import torch
torch.set_num_threads(4)
print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} -> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
USE_CUDA=torch.cuda.is_available()
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import optuna
from pathlib import Path
set_seed(42)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
ProkBERTTokenizer.vocab_size=property(lambda self: len(self.get_vocab()) if hasattr(self, 'get_vocab') else 4101)
_orig_save=ProkBERTTokenizer.save_pretrained
def _patched_save(self, save_directory, **kwargs):
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    try:
        return _orig_save(self, save_directory, **kwargs)
    except NotImplementedError:
        (Path(save_directory)/"tokenizer_config.json").write_text('{"tokenizer_class":"ProkBERTTokenizer","kmer":6,"shift":1}')
        return ()
ProkBERTTokenizer.save_pretrained=_patched_save
tokenizer=ProkBERTTokenizer(tokenization_params={'kmer':6,'shift':1}, operation_space='sequence')
print("Tokenizer k6s1 ready")

DATASETS={
    "d39v_3way_SigA_SigX_None": {"path":"datasets/d39v_3way_SigA_SigX_None","num_labels":3},
    "d39v_binary_SigA_vs_ComX": {"path":"datasets/d39v_binary_SigA_vs_ComX","num_labels":2},
    "d39v_binary_SigA_vs_Neg": {"path":"datasets/d39v_binary_SigA_vs_Neg","num_labels":2},
}

def load_tokenized(dataset_name):
    cfg=DATASETS[dataset_name]
    ds=load_dataset("csv", data_files={
        "train": f"{cfg['path']}/train.csv",
        "eval": f"{cfg['path']}/eval.csv",
        "test": f"{cfg['path']}/test.csv",
    })
    print(f"{dataset_name} {ds} num_labels {cfg['num_labels']}")
    def tok_fn(ex):
        enc=tokenizer(ex["segment"], add_special_tokens=True, padding=False, truncation=False)
        enc["labels"]=ex["y"]
        return enc
    tokenized=ds.map(tok_fn, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
    return tokenized, cfg["num_labels"]

def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    preds=np.argmax(logits, axis=-1)
    probs=np.exp(logits)/np.sum(np.exp(logits),axis=-1,keepdims=True)
    # binary roc_auc needs pos probs
    try:
        auc=roc_auc_score(labels, probs[:,1])
    except:
        auc=float('nan')
    return {"mcc":matthews_corrcoef(labels,preds),"accuracy":accuracy_score(labels,preds),"roc_auc":auc,"f1":f1_score(labels,preds,zero_division=0)}

def compute_metrics_multiclass(eval_pred):
    logits, labels = eval_pred
    preds=np.argmax(logits, axis=-1)
    # mcc multiclass
    mcc=matthews_corrcoef(labels,preds)
    acc=accuracy_score(labels,preds)
    # roc_auc ovr if possible
    try:
        # need probs for ovr
        probs=np.exp(logits)/np.sum(np.exp(logits),axis=-1,keepdims=True)
        auc=roc_auc_score(labels, probs, multi_class='ovr')
    except:
        auc=float('nan')
    return {"mcc":mcc,"accuracy":acc,"roc_auc":auc,"f1":f1_score(labels,preds,average='weighted',zero_division=0)}

def get_class_weights(tokenized_train, num_labels):
    from collections import Counter
    import numpy as np
    labels = tokenized_train["labels"]
    # compute balanced weights
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_labels), y=labels)
    # handle missing class in train (e.g., SigX 15) -> still compute
    # If some class has 0 count, weight will be inf, clamp
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0)
    print(f"Class weights {weights} for classes {list(range(num_labels))} counts {Counter(labels)}")
    return torch.tensor(weights, dtype=torch.float)

def train_default(dataset_name):
    tokenized,num_labels=load_tokenized(dataset_name)
    compute_fn=compute_metrics_multiclass if num_labels==3 else compute_metrics_binary
    model=AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True, num_labels=num_labels, ignore_mismatched_sizes=True)
    print(f"Model {model.__class__.__name__} num_labels {num_labels}")
    # class weights for imbalanced (SigX 15 vs 272)
    from collections import Counter
    class_weights = get_class_weights(tokenized["train"], num_labels)
    outdir=f"./models/{dataset_name}__default"
    args=TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-05,
        weight_decay=0.0,
        warmup_ratio=0.0,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        fp16=USE_CUDA,
        bf16=False,
        seed=42,
        logging_steps=10,
        save_total_limit=2,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )
    trainer=WeightedTrainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"], data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_fn, class_weights=class_weights)
    trainer.train()
    print(f"=== {dataset_name} default eval ===")
    print(trainer.evaluate(tokenized["eval"]))
    print(trainer.evaluate(tokenized["test"]))
    trainer.save_model(outdir+"_final")
    print(f"Saved {outdir}_final")
    del trainer, model
    if USE_CUDA: torch.cuda.empty_cache()

def train_optuna(dataset_name, n_trials=20):
    tokenized,num_labels=load_tokenized(dataset_name)
    compute_fn=compute_metrics_multiclass if num_labels==3 else compute_metrics_binary
    from collections import Counter
    class_weights = get_class_weights(tokenized["train"], num_labels)
    def objective(trial):
        lr=trial.suggest_float("learning_rate",1e-5,1e-4,log=True)
        wd=trial.suggest_float("weight_decay",0.0,0.3)
        warmup=trial.suggest_float("warmup_ratio",0.0,0.2)
        epochs=trial.suggest_int("num_train_epochs",3,6)
        model=AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True, num_labels=num_labels, ignore_mismatched_sizes=True)
        args=TrainingArguments(
            output_dir=f"./optuna_{dataset_name}/trial_{trial.number}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=lr,
            weight_decay=wd,
            warmup_ratio=warmup,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            fp16=USE_CUDA,
            bf16=False,
            seed=42,
            logging_steps=10,
            save_total_limit=1,
            disable_tqdm=True,
            dataloader_num_workers=0,
        )
        trainer=WeightedTrainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"], data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_fn, class_weights=class_weights)
        trainer.train()
        metrics=trainer.evaluate(tokenized["eval"])
        trial.set_user_attr("eval_accuracy", metrics["eval_accuracy"])
        trial.set_user_attr("eval_roc_auc", metrics["eval_roc_auc"])
        mcc=metrics["eval_mcc"]
        del trainer, model
        if USE_CUDA: torch.cuda.empty_cache()
        return mcc
    study=optuna.create_study(direction="maximize")
    print(f"=== Optuna {n_trials} trials {dataset_name} ===")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best {dataset_name} MCC {study.best_value:.4f} params {study.best_params} trial {study.best_trial.number}")
    # retrain best
    best=study.best_params
    model=AutoModelForSequenceClassification.from_pretrained(".", trust_remote_code=True, num_labels=num_labels, ignore_mismatched_sizes=True)
    outdir=f"./models/{dataset_name}__optuna_best"
    args=TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=best["learning_rate"],
        weight_decay=best["weight_decay"],
        warmup_ratio=best["warmup_ratio"],
        num_train_epochs=best["num_train_epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        report_to="none",
        fp16=USE_CUDA,
        bf16=False,
        seed=42,
        logging_steps=10,
        save_total_limit=2,
        disable_tqdm=False,
        dataloader_num_workers=0,
    )
    trainer=WeightedTrainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["eval"], data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_fn, class_weights=class_weights)
    trainer.train()
    print(f"=== {dataset_name} optuna_best eval ===")
    print(trainer.evaluate(tokenized["eval"]))
    print(trainer.evaluate(tokenized["test"]))
    trainer.save_model(outdir+"_final")
    print(f"Saved {outdir}_final")
    # save trials
    try:
        df=study.trials_dataframe()
        df.to_csv(f"{outdir}_trials.csv", index=False)
        print(f"Trials {outdir}_trials.csv")
    except Exception as e:
        print(e)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), help="dataset especifico")
    ap.add_argument("--optuna", action="store_true", help="ejecutar optuna 20")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--all", action="store_true", help="todos los datasets")
    args=ap.parse_args()
    targets=[args.dataset] if args.dataset else list(DATASETS.keys()) if args.all else list(DATASETS.keys())
    for ds in targets:
        print(f"\n===== {ds} DEFAULT =====")
        train_default(ds)
        if args.optuna or args.all:
            print(f"\n===== {ds} OPTUNA {args.trials} =====")
            train_optuna(ds, n_trials=args.trials)
