import sys
sys.path.insert(0,'/home/fierro/pneumo_prokBERT')
from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
import csv
tok=ProkBERTTokenizer(tokenization_params={'kmer':6,'shift':1}, operation_space='sequence')
def eval_csv(model_path, csv_path):
    rows=list(csv.DictReader(open(csv_path)))
    seqs=[r['segment'] for r in rows]
    y=[int(r['y']) for r in rows]
    model=AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    dev='cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    probs=[]
    for i in range(0,len(seqs),32):
        enc=tok.batch_encode_plus(seqs[i:i+32], return_tensors='pt')
        with torch.no_grad():
            probs.extend(torch.softmax(model(input_ids=enc['input_ids'].to(dev), attention_mask=enc['attention_mask'].to(dev)).logits,dim=-1)[:,1].cpu().tolist())
    pred=[1 if p>=0.5 else 0 for p in probs]
    return matthews_corrcoef(y,pred), accuracy_score(y,pred), roc_auc_score(y,probs)

for name in ['d39v_full_989__default_final','d39v_full_989__optuna_best_final']:
    mcc,acc,auc=eval_csv(f'/home/fierro/pneumo_prokBERT/models/{name}', '/home/fierro/pneumo_prokBERT/datasets/d39v_full_989/test.csv')
    print(name + f" test296 mcc {mcc:.4f} acc {acc:.4f} auc {auc:.4f}")
    mcc2,acc2,auc2=eval_csv(f'/home/fierro/pneumo_prokBERT/models/{name}', '/home/fierro/pneumo_prokBERT/datasets/d39v_full_989/eval.csv')
    print(name + f" eval296 mcc {mcc2:.4f} acc {acc2:.4f} auc {auc2:.4f}")
