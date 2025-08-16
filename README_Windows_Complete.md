# BitShield - Hướng dẫn hoàn chỉnh cho Windows

## 🎯 **Tổng quan**

Dự án BitShield nghiên cứu về bảo vệ DNN khỏi tấn công bit-flip. Hướng dẫn này dành riêng cho Windows.

## 📋 **Prerequisites (Yêu cầu hệ thống)**

### **Đã cài đặt:**
- ✅ Python 3.12
- ✅ Git
- ✅ Virtual environment đã tạo
- ✅ PyTorch CPU version đã cài

### **Cần cài thêm:**
- 🔄 Docker Desktop (cho model building và experiments)
- 🔄 ImageNet dataset (tùy chọn)

## 🚀 **Hướng dẫn chạy lần đầu**

### **Bước 1: Setup ban đầu (ĐÃ HOÀN THÀNH)**
```powershell
# Đã chạy xong - không cần chạy lại
.\setup_fix.ps1
```

### **Bước 2: Kích hoạt môi trường (LÀM MỖI LẦN)**
```powershell
# Kích hoạt virtual environment
.\venv\Scripts\Activate.ps1

# Dấu hiệu thành công: (venv) ở đầu dòng prompt
```

### **Bước 3: Tải datasets (ĐÃ HOÀN THÀNH)**
```powershell
# Đã chạy xong - không cần chạy lại
python tools\ensure_datasets.py
```

### **Bước 4: Training models (CÓ THỂ CHẠY LẠI)**
```powershell
# Train ResNet50 trên CIFAR10
python support\models\train.py resnet50 CIFAR10 --epochs 5

# Train nhiều models
for ($m in @("resnet50", "densenet121", "googlenet")) {
    for ($x in @("CIFAR10", "MNISTC", "FashionC")) {
        python support\models\train.py $m $x --epochs 3
    }
}
```

### **Bước 5: Docker setup (CHƯA LÀM)**
```powershell
# Cài đặt Docker image (mất 10-30 phút)
docker\setup.bat
```

### **Bước 6: Build models (CHƯA LÀM)**
```powershell
# Build tất cả models
docker\run-in-docker.bat python buildmodels.py

# Hoặc sử dụng DVC
dvc repro
```

### **Bước 7: Chạy experiments (CHƯA LÀM)**
```powershell
# Tìm vulnerable bits
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10

# Chạy attack simulation
tools\runattacksim.bat -m resnet50 -d CIFAR10
```

## 📁 **Cấu trúc thư mục sau khi chạy**

```
D:\Programming\BitShield_Clone\
├── venv\                       # Virtual environment (ĐÃ TẠO)
├── datasets\                   # Datasets đã tải (ĐÃ TẠO)
│   ├── CIFAR10\
│   ├── CIFAR100\
│   ├── MNIST\
│   └── ...
├── models\                     # Models đã train (SẼ TẠO)
│   ├── CIFAR10\
│   │   └── resnet50\
│   │       └── resnet50.pt
│   └── ...
├── built\                      # Compiled models (SẼ TẠO)
├── results\                    # Experiment results (SẼ TẠO)
└── ...
```

## 🔄 **Hướng dẫn chạy những lần sau**

### **Mỗi lần mở PowerShell mới:**

```powershell
# 1. Di chuyển đến thư mục dự án
cd D:\Programming\BitShield_Clone

# 2. Kích hoạt môi trường (BẮT BUỘC)
.\venv\Scripts\Activate.ps1

# 3. Kiểm tra môi trường
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### **Nếu muốn training thêm models:**

```powershell
# Training model mới
python support\models\train.py googlenet CIFAR10 --epochs 5

# Training với tham số khác
python support\models\train.py resnet50 MNISTC --epochs 10 --batch-size 50
```

### **Nếu muốn chạy experiments (cần Docker):**

```powershell
# Setup Docker (chỉ làm 1 lần)
docker\setup.bat

# Build models
docker\run-in-docker.bat python buildmodels.py

# Chạy experiments
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10
```

## 🛠️ **Các lệnh hữu ích**

### **Kiểm tra trạng thái:**
```powershell
# Kiểm tra datasets
dir datasets

# Kiểm tra models đã train
dir models

# Kiểm tra kết quả
dir results
```

### **DVC operations:**
```powershell
# Pull data mới
dvc pull

# Reproduce experiments
dvc repro

# Check status
dvc status
```

### **Docker operations:**
```powershell
# Kiểm tra Docker image
docker images

# Chạy lệnh trong Docker
docker\run-in-docker.bat python --version
```

## ⚠️ **Troubleshooting**

### **Lỗi "No module named 'torch'":**
```powershell
# Kích hoạt lại virtual environment
.\venv\Scripts\Activate.ps1
```

### **Lỗi CUDA:**
```powershell
# Sử dụng CPU thay vì GPU
python support\models\train.py resnet50 CIFAR10 --device cpu
```

### **Lỗi Docker:**
```powershell
# Kiểm tra Docker Desktop đang chạy
docker version

# Restart Docker Desktop nếu cần
```

### **Lỗi permission:**
```powershell
# Chạy PowerShell với quyền Administrator
```

## 📊 **Thời gian ước tính**

- **Training ResNet50 trên CIFAR10**: 10-30 phút (CPU)
- **Docker setup**: 10-30 phút
- **Build models**: 30-60 phút
- **Bit-flip sweep**: 1-6 giờ
- **Attack simulation**: 30 phút - 2 giờ

## 🎯 **Workflow nhanh cho lần sau**

```powershell
# 1. Kích hoạt môi trường
cd D:\Programming\BitShield_Clone
.\venv\Scripts\Activate.ps1

# 2. Training (nếu cần)
python support\models\train.py resnet50 CIFAR10 --epochs 3

# 3. Experiments (nếu có Docker)
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10
```

## 📞 **Hỗ trợ**

- **Lỗi Python**: Kiểm tra virtual environment
- **Lỗi Docker**: Kiểm tra Docker Desktop
- **Lỗi training**: Giảm epochs hoặc batch size
- **Lỗi memory**: Đóng các ứng dụng khác

---

**Lưu ý**: Các bước đã hoàn thành không cần làm lại. Chỉ cần kích hoạt môi trường và chạy các bước tiếp theo.
