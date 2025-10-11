
Train model:
```
python Integrated_BFA_BitShield/train_iotid20.py --data-root ../dataset --model custom1 --epochs 5 --out "./save/custom1.pth"
```

Attack (no DIG)
```
python Integrated_BFA_BitShield\run_pipeline.py ... --disable-dig
```

Attack (with DIG)
```
python Integrated_BFA_BitShield\run_pipeline.py ... --disable-dig
```