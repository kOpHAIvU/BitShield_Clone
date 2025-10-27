# Hướng dẫn Training IoTID20 Dataset

## Tổng quan

Hệ thống này cho phép bạn train các model deep learning trên dataset IoTID20 (IoT Network Intrusion Detection) sử dụng file `train.py` gốc với các model architectures được tối ưu hóa cho IoT.

## Cấu trúc thư mục

```
BitShield_Clone/
├── support/
│   ├── models/
│   │   ├── train.py                    # Script training chính (đã được cập nhật)
│   │   ├── quantized_layers.py         # Các lớp quantization
│   │   ├── resnet_seblock_iot.py       # ResNet với SE blocks
│   │   ├── simple_cnn_iot.py          # Simple CNN
│   │   ├── purecnn_iot.py             # Pure CNN baseline
│   │   ├── efficientcnn_iot.py        # Efficient CNN
│   │   └── __init__.py                # Export các model
│   ├── dataman_iotid20.py             # Module tiền xử lý dữ liệu IoTID20
│   ├── dataset/                       # Thư mục chứa dataset
│   │   ├── IoT_Network_Intrusion_Dataset.csv  # File dữ liệu gốc
│   │   ├── train.csv                  # Tập train (tự động tạo)
│   │   └── test.csv                   # Tập test (tự động tạo)
│   ├── demo_iotid20_training.py       # Script demo
│   └── README_IoTID20.md              # Tài liệu chi tiết
└── README_IoTID20_Training.md         # File này
```

## Cài đặt và chuẩn bị

### 1. Yêu cầu hệ thống
- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- numpy
- tqdm

### 2. Chuẩn bị dataset
Dataset IoTID20 đã được đặt trong `support/dataset/IoT_Network_Intrusion_Dataset.csv`. Hệ thống sẽ tự động:
- Chia dữ liệu thành train/test (80%/20%)
- Chuẩn hóa features (StandardScaler)
- Mã hóa labels (LabelEncoder)
- Tạo các file `train.csv` và `test.csv`

## Cách sử dụng

### 1. Cú pháp cơ bản

```bash
python support/models/train.py <model_name> IoTID20 [options]
```

### 2. Các model có sẵn

| Model Name | Mô tả | Số tham số | Đặc điểm |
|------------|-------|------------|----------|
| `ResNetSEBlockIoT` | ResNet với SE blocks + quantization | ~2M | Kiến trúc phức tạp, hiệu suất cao |
| `SimpleCNNIoT` | CNN đơn giản với adaptive pooling | ~77K | Nhẹ, nhanh |
| `PureCNN` | CNN thuần không quantization | ~697K | Baseline để so sánh |
| `EfficientCNN` | CNN hiệu quả với depthwise separable | ~49K | Rất nhẹ, tối ưu |
| `CustomModel` | Alias cho ResNetSEBlockIoT | ~2M | Tương thích ngược |
| `CustomModel2` | Alias cho SimpleCNNIoT | ~77K | Tương thích ngược |

### 3. Các tùy chọn

```bash
python support/models/train.py <model_name> IoTID20 \
    --epochs 10 \           # Số epochs (mặc định: 10)
    --batch-size 256 \      # Batch size (mặc định: 256)
    --device cuda \         # Thiết bị: cpu/cuda (mặc định: cpu)
    --output-root ./models \ # Thư mục lưu model (mặc định: cfg.models_dir)
    --skip-existing \       # Bỏ qua nếu file đã tồn tại
    --use-class-weights \   # Sử dụng class weights cho imbalanced data
    --learning-rate 0.001 \ # Learning rate (mặc định: 0.001)
    --weight-decay 0.0001   # Weight decay cho regularization (mặc định: 0.0001)
```

## Ví dụ sử dụng

### 1. Training cơ bản

```bash
# Train ResNetSEBlockIoT với 10 epochs
python support/models/train.py ResNetSEBlockIoT IoTID20 --epochs 10 --batch-size 64

# Train PureCNN với 5 epochs
python support/models/train.py PureCNN IoTID20 --epochs 5 --batch-size 32

# Train với GPU
python support/models/train.py EfficientCNN IoTID20 --epochs 10 --batch-size 64 --device cuda
```

### 2. Training với cải tiến (Khuyến nghị)

```bash
# Training với class weights để xử lý imbalanced data
python support/models/train.py CustomModel2 IoTID20 --epochs 15 --use-class-weights

# Training với tất cả cải tiến
python support/models/train.py ResNetSEBlockIoT IoTID20 --epochs 20 \
    --use-class-weights --learning-rate 0.0005 --weight-decay 0.0001 --device cuda
```

### 3. Training nhanh để test

```bash
# Test nhanh với 1 epoch
python support/models/train.py SimpleCNNIoT IoTID20 --epochs 1 --batch-size 32

# Test với class weights
python support/models/train.py CustomModel2 IoTID20 --epochs 1 --use-class-weights
```

### 4. Training với tùy chọn nâng cao

```bash
# Train với batch size lớn và lưu vào thư mục tùy chỉnh
python support/models/train.py ResNetSEBlockIoT IoTID20 \
    --epochs 20 \
    --batch-size 128 \
    --device cuda \
    --output-root ./my_models \
    --use-class-weights
```

## Chạy demo

```bash
# Chạy script demo để xem các ví dụ
python support/demo_iotid20_training.py

# Chạy script demo cải tiến
python support/demo_improved_training.py
```

## Cải tiến mới

### 🚀 **Các tính năng cải tiến:**

1. **Class Weights** (`--use-class-weights`):
   - Tự động tính toán trọng số cho các lớp
   - Cải thiện phát hiện các lớp thiểu số
   - Giảm bias về lớp đa số

2. **Learning Rate Scheduling**:
   - Tự động giảm learning rate khi không cải thiện
   - Tránh overfitting
   - Tối ưu hóa convergence

3. **Early Stopping**:
   - Dừng sớm khi không cải thiện
   - Tránh overfitting
   - Tiết kiệm thời gian training

4. **Weight Decay** (`--weight-decay`):
   - Regularization để tránh overfitting
   - Cải thiện generalization

5. **Best Model Saving**:
   - Tự động lưu model tốt nhất
   - Sử dụng model tốt nhất cho evaluation

### 📊 **Kết quả mong đợi:**

- **TPR cao hơn**: Phát hiện tốt hơn các loại tấn công
- **Confusion Matrix cân bằng**: Ít nhầm lẫn giữa các lớp
- **MCC cao hơn**: Đánh giá tổng thể tốt hơn
- **Training ổn định**: Ít overfitting

## Kết quả

### 1. Model được lưu
Models được lưu tại: `{output_root}/IoTID20/{model_name}/{model_name}.pt`

Ví dụ: `cfg.models_dir/IoTID20/ResNetSEBlockIoT/ResNetSEBlockIoT.pt`

### 2. Thông tin dataset
- **Features**: 69 đặc trưng mạng
- **Classes**: 5 loại xâm nhập
- **Train samples**: ~80% dữ liệu
- **Test samples**: ~20% dữ liệu

### 3. Output training
```
Using IoTID20 data preprocessing...
Dataset info: 69 features, 5 classes
IoTID20 dataset: 69 features, 5 classes
Epoch 0: val_acc=0.8234
Epoch 1: val_acc=0.8567
...
Parameters saved to: cfg.models_dir/IoTID20/ResNetSEBlockIoT/ResNetSEBlockIoT.pt
```

## Troubleshooting

### 1. Lỗi import module
```bash
# Đảm bảo đang chạy từ thư mục gốc của project
cd /path/to/BitShield_Clone
python support/models/train.py ResNetSEBlockIoT IoTID20 --epochs 1
```

### 2. Lỗi dataset không tìm thấy
- Kiểm tra file `support/dataset/IoT_Network_Intrusion_Dataset.csv` có tồn tại
- Hệ thống sẽ tự động tạo `train.csv` và `test.csv` từ file gốc

### 3. Lỗi GPU
```bash
# Kiểm tra CUDA có sẵn
python -c "import torch; print(torch.cuda.is_available())"

# Nếu không có GPU, sử dụng CPU
python support/models/train.py ResNetSEBlockIoT IoTID20 --device cpu
```

### 4. Lỗi memory
- Giảm batch size: `--batch-size 16` hoặc `--batch-size 8`
- Sử dụng CPU: `--device cpu`

## So sánh các model

| Model | Ưu điểm | Nhược điểm | Phù hợp cho |
|-------|---------|------------|-------------|
| ResNetSEBlockIoT | Hiệu suất cao, có attention | Nhiều tham số, chậm | Nghiên cứu, production |
| SimpleCNNIoT | Cân bằng tốt | Hiệu suất trung bình | Prototype, baseline |
| PureCNN | Đơn giản, dễ hiểu | Không có quantization | So sánh, học tập |
| EfficientCNN | Rất nhẹ, nhanh | Hiệu suất có thể thấp hơn | Edge device, real-time |

## Lưu ý quan trọng

1. **Lần đầu chạy**: Hệ thống sẽ tự động xử lý dữ liệu, có thể mất vài phút
2. **GPU**: Sử dụng GPU sẽ tăng tốc đáng kể quá trình training
3. **Batch size**: Điều chỉnh theo khả năng của máy
4. **Epochs**: Bắt đầu với ít epochs để test, sau đó tăng dần

## Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. File README chi tiết: `support/README_IoTID20.md`
2. Script demo: `python support/demo_iotid20_training.py`
3. Logs trong quá trình training
