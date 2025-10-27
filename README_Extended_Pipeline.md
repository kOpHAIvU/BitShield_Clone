# BitShield Extended Pipeline

## 🎯 **Tổng quan**

Pipeline mở rộng hỗ trợ 3 datasets tabular:
- **IoTID20**: IoT Intrusion Detection Dataset 2020
- **WUSTL**: WUSTL-IIoT-2021 Dataset  
- **CICIoT2023**: CIC IoT Dataset 2023

## 📁 **Cấu trúc thư mục**

```
BitShield_Clone/
├── support/
│   ├── dataman_extended.py          # Data manager mở rộng
│   ├── models/
│   │   ├── train_extended.py        # Training script mở rộng
│   │   └── ...                      # Các model architectures
│   └── dataset/
│       ├── IoTID20/                 # IoTID20 dataset
│       ├── WUSTL/                   # WUSTL dataset
│       └── CICIoT2023/              # CICIoT2023 dataset
├── attack_with_defense_extended.py  # Attack simulation mở rộng
├── demo_extended_pipeline.py        # Demo script
└── README_Extended_Pipeline.md      # Hướng dẫn này
```

## 🚀 **Cách sử dụng**

### **1. Training trên dataset mới:**

```bash
# Training ResNetSEBlockIoT trên WUSTL
python support/models/train_extended.py ResNetSEBlockIoT WUSTL --epochs 10 --device cpu

# Training SimpleCNNIoT trên CICIoT2023 với class weights
python support/models/train_extended.py SimpleCNNIoT CICIoT2023 --epochs 15 --use-class-weights --device cpu

# Training với tùy chọn nâng cao
python support/models/train_extended.py ResNetSEBlockIoT IoTID20 --epochs 20 --batch-size 128 --learning-rate 0.001 --weight-decay 0.0001 --device cpu
```

### **2. Testing defense mechanisms:**

```bash
# Test DIG defense
python attack_with_defense_extended.py dig ResNetSEBlockIoT WUSTL --device cpu

# Test CIG defense  
python attack_with_defense_extended.py cig SimpleCNNIoT CICIoT2023 --device cpu

# Test combined defense
python attack_with_defense_extended.py combined ResNetSEBlockIoT IoTID20 --device cpu
```

### **3. Demo tự động:**

```bash
# Demo tất cả
python demo_extended_pipeline.py --mode all

# Demo chỉ training
python demo_extended_pipeline.py --mode training

# Demo chỉ defense testing
python demo_extended_pipeline.py --mode defense

# Demo specific dataset và model
python demo_extended_pipeline.py --mode specific --dataset WUSTL --model ResNetSEBlockIoT --epochs 10
```

## 📊 **Datasets được hỗ trợ**

### **IoTID20**
- **File**: `support/dataset/IoTID20/train.csv`
- **Features**: 69 features
- **Classes**: 5 classes
- **Preprocessing**: StandardScaler + LabelEncoder

### **WUSTL**
- **File**: `support/dataset/WUSTL/wustl_iiot_2021_reduced.csv`
- **Features**: ~40 features (after removing unnecessary columns)
- **Classes**: Variable (based on Traffic column)
- **Preprocessing**: Remove duplicates, drop unnecessary columns, StandardScaler + LabelEncoder

### **CICIoT2023**
- **File**: `support/dataset/CICIoT2023/CIC_IoT_Dataset2023.csv`
- **Features**: Variable (after removing Cat column)
- **Classes**: Variable (based on Label column)
- **Preprocessing**: Remove duplicates, handle NaN, normalize specific columns, StandardScaler + LabelEncoder

## 🔧 **Tùy chọn training**

| Parameter | Mô tả | Default |
|-----------|-------|---------|
| `--epochs` | Số epochs | 10 |
| `--batch-size` | Batch size | 256 |
| `--device` | Device (cpu/cuda) | cpu |
| `--use-class-weights` | Sử dụng class weights | False |
| `--learning-rate` | Learning rate | 1e-3 |
| `--weight-decay` | Weight decay | 1e-4 |

## 🛡️ **Defense mechanisms**

### **DIG (Detection of Input Gradient)**
- Sử dụng Tabular DIG cho tất cả datasets
- Phát hiện dựa trên gradient norm, entropy, confidence
- Adaptive thresholds dựa trên clean data

### **CIG (Code Integrity Guard)**
- Kiểm tra tính toàn vẹn parameters
- So sánh với trạng thái gốc
- Threshold-based detection

### **Combined Defense**
- Kết hợp DIG + CIG
- Detection rate = max(DIG, CIG)
- Comprehensive protection

## 📈 **Metrics được tính**

- **Accuracy**: Tỷ lệ dự đoán đúng
- **MCC**: Matthews Correlation Coefficient
- **TPR**: True Positive Rate (average)
- **F1 Score**: F1 Score (average)
- **Confusion Matrix**: Ma trận nhầm lẫn
- **Per-class metrics**: TPR và F1 cho từng class

## 🔍 **Kết quả được lưu**

### **Training results:**
- `results/models/{dataset}/{model}/{model}.pt`: Model weights
- `results/models/{dataset}/{model}/{model}_best.pt`: Best model
- `results/models/{dataset}/{model}/{model}_results.json`: Training metrics

### **Defense results:**
- `results/defense_results/{dataset}_{model}_dig_attack.json`: DIG results
- `results/defense_results/{dataset}_{model}_cig_attack.json`: CIG results
- `results/defense_results/{dataset}_{model}_combined_attack.json`: Combined results

## ⚠️ **Lưu ý quan trọng**

1. **Dataset files**: Đảm bảo các file dataset được đặt đúng đường dẫn
2. **Memory usage**: Datasets lớn có thể cần nhiều RAM
3. **GPU support**: Thay `--device cpu` thành `--device cuda` nếu có GPU
4. **Class imbalance**: Sử dụng `--use-class-weights` cho datasets có class imbalance
5. **Early stopping**: Training sẽ dừng sớm nếu validation accuracy không cải thiện

## 🐛 **Troubleshooting**

### **Lỗi "Dataset not found":**
- Kiểm tra đường dẫn dataset files
- Đảm bảo tên file chính xác

### **Lỗi "Model not found":**
- Kiểm tra model đã được train chưa
- Kiểm tra tên model có đúng không

### **Lỗi "CUDA out of memory":**
- Giảm batch size
- Sử dụng `--device cpu`

### **Lỗi "Class weights calculation":**
- Kiểm tra dataset có labels hợp lệ không
- Thử không sử dụng `--use-class-weights`

## 📞 **Hỗ trợ**

Nếu gặp vấn đề, hãy kiểm tra:
1. Dataset files có tồn tại và đúng format
2. Model architecture có hỗ trợ input_size và output_size
3. Python environment có đầy đủ dependencies
4. Log files để xem chi tiết lỗi
