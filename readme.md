# UniPruning

## 1. Environment Setup
```bash
conda create --name Uniprune python==3.10
conda activate Uniprune
pip install -r requirements.txt
```
## 2. Quick Start
```bash
bash train_24_final.sh # 2:4 pruning
bash final_unstructured.sh # unstructured pruning
```
## 3. searching & extract mask 
```
python main.py # for searching gamma  
python extract_mask_mixed.py # for extract mask using gamma  
```

## 4. evaluate ppl
ppl.py # for evaluate ppl results

