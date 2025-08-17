# BitShield - Hướng dẫn chạy hoàn toàn trên Docker

## Tổng quan

Hướng dẫn này sẽ giúp bạn chạy dự án BitShield hoàn toàn trong Docker, đảm bảo môi trường ổn định và tương thích trên mọi hệ thống Windows.

## Yêu cầu

### Bắt buộc
- **Docker Desktop** - Tải từ [docker.com](https://www.docker.com/products/docker-desktop/)
- **Git** - Tải từ [git-scm.com](https://git-scm.com/download/win)

### Tùy chọn
- **Python 3.8+** - Chỉ cần để chạy script setup ban đầu

## Cài đặt nhanh

### Bước 1: Chuẩn bị
```cmd
REM Clone dự án
git clone <repository-url>
cd BitShield_Clone

REM Kiểm tra Docker
docker --version
```

### Bước 2: Build Docker image
```cmd
docker\setup.bat
```

### Bước 3: Chạy thử nghiệm
```cmd
REM Tải datasets
docker\run-in-docker.bat python tools\ensure_datasets.py

REM Huấn luyện mô hình
docker\run-in-docker.bat python support\models\train.py resnet50 CIFAR10
```

## Các lệnh Docker cơ bản

### Chạy lệnh Python trong Docker
```cmd
docker\run-in-docker.bat python <script.py> [arguments]
```

### Ví dụ cụ thể
```cmd
REM Huấn luyện mô hình
docker\run-in-docker.bat python support\models\train.py resnet50 CIFAR10

REM Test mô hình
docker\run-in-docker.bat python support\models\train.py resnet50 CIFAR10 --test-only

REM Build models
docker\run-in-docker.bat python buildmodels.py

REM Tìm kiếm bit dễ bị tấn công
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10

REM Chạy attack simulation
docker\run-in-docker.bat python attacksim.py --model-name resnet50 --dataset CIFAR10
```

## Workflow hoàn chỉnh

### 1. Huấn luyện mô hình
```cmd
REM Huấn luyện ResNet50 trên CIFAR10
docker\run-in-docker.bat python support\models\train.py resnet50 CIFAR10

REM Huấn luyện nhiều mô hình
for %m in (resnet50 densenet121 googlenet) do (
    for %d in (CIFAR10 MNISTC FashionC) do (
        docker\run-in-docker.bat python support\models\train.py %m %d
    )
)
```

### 2. Build binary files
```cmd
REM Build tất cả mô hình
docker\run-in-docker.bat python buildmodels.py

REM Hoặc build từng mô hình cụ thể
docker\run-in-docker.bat python buildmodels.py --model resnet50 --dataset CIFAR10
```

### 3. Phân tích bit-flip
```cmd
REM Tìm kiếm bit dễ bị tấn công
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10

REM Chạy với các cấu hình khác nhau
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10 --cig ncnp --dig nd
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10 --cig cc2 --dig gn1
```

### 4. Mô phỏng tấn công
```cmd
REM Chạy tất cả experiments
docker\run-in-docker.bat python attacksim.py

REM Chạy với mô hình/dataset cụ thể
docker\run-in-docker.bat python attacksim.py --model-name resnet50 --dataset CIFAR10 --attacker-type w
```

## Quản lý dữ liệu

### Tải datasets
```cmd
docker\run-in-docker.bat python tools\ensure_datasets.py
```

### Sử dụng DVC
```cmd
REM Pull dữ liệu mới nhất
docker\run-in-docker.bat dvc pull

REM Reproduce experiments
docker\run-in-docker.bat dvc repro

REM Kiểm tra trạng thái
docker\run-in-docker.bat dvc status
```

## Xem kết quả

### Kết quả huấn luyện
```cmd
REM Xem models đã huấn luyện
dir models

REM Xem accuracy
docker\run-in-docker.bat python -c "import torch; print(torch.load('models/resnet50_CIFAR10.pth', map_location='cpu').keys())"
```

### Kết quả phân tích
```cmd
REM Xem kết quả sweep
dir results\sweep

REM Xem kết quả attack simulation
dir results\attacksim
```

## Troubleshooting

### Lỗi Docker không chạy
```cmd
REM Kiểm tra Docker
docker version

REM Khởi động Docker Desktop nếu cần
REM Mở Docker Desktop từ Start Menu
```

### Lỗi image không tìm thấy
```cmd
REM Build lại image
docker\setup.bat

REM Hoặc build thủ công
docker build -t debfd-runner -f docker\Dockerfile .
```

### Lỗi permission
```cmd
REM Chạy Command Prompt với quyền Administrator
REM Hoặc kiểm tra Docker Desktop settings
```

### Lỗi out of memory
```cmd
REM Tăng memory cho Docker Desktop
REM Settings > Resources > Memory > 8GB+
```

### Lỗi disk space
```cmd
REM Dọn dẹp Docker
docker system prune -a

REM Xóa images không dùng
docker image prune -a
```

## Script tiện ích

### Chạy nhiều experiments
```cmd
REM Tạo script batch
echo @echo off > run_experiments.bat
echo docker\run-in-docker.bat python support\models\train.py resnet50 CIFAR10 >> run_experiments.bat
echo docker\run-in-docker.bat python buildmodels.py >> run_experiments.bat
echo docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10 >> run_experiments.bat
echo docker\run-in-docker.bat python attacksim.py --model-name resnet50 --dataset CIFAR10 >> run_experiments.bat

REM Chạy script
run_experiments.bat
```

### Monitor resources
```cmd
REM Xem sử dụng Docker
docker stats

REM Xem logs
docker logs <container_id>
```

## Tối ưu hiệu suất

### Cấu hình Docker Desktop
- **Memory**: 8GB+ (khuyến nghị 16GB)
- **CPU**: 4+ cores
- **Disk**: SSD với 50GB+ trống
- **WSL2 backend**: Bật để tăng hiệu suất

### Cấu hình build
```cmd
REM Build với nhiều cores
docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t debfd-runner -f docker\Dockerfile .

REM Sử dụng cache
docker build --cache-from debfd-runner -t debfd-runner -f docker\Dockerfile .
```

## Backup và Restore

### Backup kết quả
```cmd
REM Copy kết quả ra ngoài container
docker run --rm -v %cd%:/workspace debfd-runner tar -czf /workspace/results_backup.tar.gz /workspace/results
```

### Restore kết quả
```cmd
REM Restore từ backup
docker run --rm -v %cd%:/workspace debfd-runner tar -xzf /workspace/results_backup.tar.gz -C /workspace
```

## Kết luận

Với Docker, bạn có thể:
- ✅ Chạy dự án trên mọi Windows machine
- ✅ Đảm bảo môi trường ổn định
- ✅ Dễ dàng chia sẻ và reproduce
- ✅ Tách biệt môi trường development
- ✅ Backup và restore dễ dàng

**Lưu ý**: Luôn sử dụng `docker\run-in-docker.bat` để chạy các lệnh Python trong dự án.
