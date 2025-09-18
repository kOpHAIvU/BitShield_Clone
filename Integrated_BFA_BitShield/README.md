
Train model:
```
python Integrated_BFA_BitShield\train_iotid20.py --data-root "D:/Programming/BFA_remake/BFA_w_IoTID20 dataset" --model custom1 --epochs 60 --out "D:/Programming/BFA_remake/Integrated_BFA_BitShield/save/custom1.pth"
```

Attack (no DIG)
```
python Integrated_BFA_BitShield\run_pipeline.py ... --disable-dig
```

Attack (with DIG)
```
python Integrated_BFA_BitShield\run_pipeline.py ... --disable-dig
```