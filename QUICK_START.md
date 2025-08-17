# BitShield - Hướng dẫn chạy nhanh với Docker

## Cài đặt nhanh (5 phút)

### 1. Chuẩn bị
- Cài đặt Docker Desktop từ [docker.com](https://www.docker.com/products/docker-desktop/)
- Cài đặt Git từ [git-scm.com](https://git-scm.com/download/win)

### 2. Clone và setup
```cmd
git clone <repository-url>
cd BitShield_Clone
run_docker.bat setup
```

### 3. Tải datasets
```cmd
run_docker.bat download-datasets
```

### 4. Chạy thử nghiệm
```cmd
run_docker.bat train resnet50 CIFAR10
```

## Các lệnh cơ bản

### Huấn luyện và test
```cmd
REM Huấn luyện mô hình
run_docker.bat train resnet50 CIFAR10

REM Test mô hình
run_docker.bat test resnet50 CIFAR10

REM Xem danh sách mô hình và datasets
run_docker.bat list-models
run_docker.bat list-datasets
```

### Build và phân tích
```cmd
REM Build binary files
run_docker.bat build

REM Tìm kiếm bit dễ bị tấn công
run_docker.bat sweep resnet50 CIFAR10

REM Mô phỏng tấn công
run_docker.bat attack resnet50 CIFAR10
```

### Workflow hoàn chỉnh
```cmd
REM Chạy toàn bộ workflow từ huấn luyện đến phân tích
run_complete_workflow.bat resnet50 CIFAR10
```

## Lệnh hữu ích

### Kiểm tra môi trường
```cmd
docker --version
run_docker.bat list-models
run_docker.bat list-datasets
```

### Mở shell trong container
```cmd
run_docker.bat shell
```

### Xem kết quả
```cmd
dir models
dir built
dir results
```

## Hướng dẫn cho người mới

### Bước 1: Thử nghiệm đơn giản
```cmd
REM Tải datasets
run_docker.bat download-datasets

REM Huấn luyện mô hình đơn giản
run_docker.bat train lenet1 MNIST

REM Test mô hình
run_docker.bat test lenet1 MNIST
```

### Bước 2: Thử nghiệm nâng cao
```cmd
REM Huấn luyện mô hình phức tạp
run_docker.bat train resnet50 CIFAR10

REM Build và phân tích
run_docker.bat build
run_docker.bat sweep resnet50 CIFAR10
```

### Bước 3: Workflow hoàn chỉnh
```cmd
REM Chạy toàn bộ quy trình
run_complete_workflow.bat resnet50 CIFAR10
```

## Xử lý lỗi nhanh

### Lỗi Docker
- Khởi động Docker Desktop
- Kiểm tra: `docker version`

### Lỗi image không tìm thấy
```cmd
run_docker.bat setup
```

### Lỗi thiếu datasets
```cmd
run_docker.bat download-datasets
```

### Lỗi quyền truy cập
- Chạy Command Prompt với quyền Administrator

### Lỗi out of memory
- Tăng memory cho Docker Desktop (8GB+)

## Tối ưu hiệu suất

### Cấu hình Docker Desktop
- **Memory**: 8GB+ (khuyến nghị 16GB)
- **CPU**: 4+ cores
- **Disk**: SSD với 50GB+ trống
- **WSL2 backend**: Bật để tăng hiệu suất

### Monitor resources
```cmd
docker stats
```

## Ví dụ workflow hoàn chỉnh

### 1. Huấn luyện nhiều mô hình
```cmd
for %m in (resnet50 densenet121 googlenet) do (
    for %d in (CIFAR10 MNISTC FashionC) do (
        run_docker.bat train %m %d
    )
)
```

### 2. Phân tích tất cả mô hình
```cmd
for %m in (resnet50 densenet121 googlenet) do (
    for %d in (CIFAR10 MNISTC FashionC) do (
        run_docker.bat sweep %m %d
        run_docker.bat attack %m %d
    )
)
```

### 3. Chạy workflow hoàn chỉnh cho tất cả
```cmd
for %m in (resnet50 densenet121 googlenet) do (
    for %d in (CIFAR10 MNISTC FashionC) do (
        run_complete_workflow.bat %m %d
    )
)
```

## Liên hệ hỗ trợ

Nếu gặp vấn đề, xem:
- `DOCKER_GUIDE.md` - Hướng dẫn Docker chi tiết
- `README.md` - Hướng dẫn tổng quan
- Kiểm tra logs trong terminal

**Lưu ý**: Tất cả lệnh đều chạy trong Docker container, đảm bảo môi trường ổn định và tương thích.
