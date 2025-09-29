1) 
conda create --name Uniprune python==3.10
conda activate Uniprune
pip install -r requirement.txt

2) 
bash train_24_final.sh # 2:4 pruning
bash final_unstructured.sh # unstructured pruning

3）
main.py # for searching gamma
extract_mask_mixed.py # for extract mask using gamma
ppl.py # for evaluate ppl results

