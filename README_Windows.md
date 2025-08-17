# BitShield - Hướng dẫn Windows

## Tổng quan

BitShield là một dự án nghiên cứu bảo vệ chống lại các cuộc tấn công bit-flip trên các file thực thi mạng nơ-ron sâu (DNN). Hướng dẫn này dành riêng cho Windows.

## Yêu cầu hệ thống

### Phần mềm bắt buộc

1. **Python 3.8+** - Tải từ [python.org](https://www.python.org/downloads/)
   - Đảm bảo tích hợp "Add Python to PATH" khi cài đặt
   - Kiểm tra: `python --version`

2. **Git** - Tải từ [git-scm.com](https://git-scm.com/download/win)
   - Kiểm tra: `git --version`

3. **Docker Desktop** - Tải từ [docker.com](https://www.docker.com/products/docker-desktop/)
   - Cần thiết để build models và chạy experiments
   - Đảm bảo Docker Desktop đang chạy trước khi sử dụng

### Yêu cầu phần cứng

- **RAM tối thiểu**: 8GB
- **RAM khuyến nghị**: 16GB+
- **Dung lượng ổ cứng**: 50GB+ trống
- **Ổ cứng**: SSD được khuyến nghị

## Cài đặt nhanh

### Bước 1: Cài đặt ban đầu

```cmd
setup.bat
```

Script này sẽ:
- Khởi tạo git submodules
- Tạo virtual environment Python
- Cài đặt tất cả dependencies cần thiết
- Tạo các thư mục cần thiết

### Bước 2: Kích hoạt môi trường

```cmd
env.bat
```

### Bước 3: Tải datasets

```cmd
python tools\ensure_datasets.py
```

### Bước 4: Cài đặt Docker (nếu cần build models)

```cmd
docker\setup.bat
```

## Sử dụng cơ bản

### Huấn luyện mô hình

```cmd
REM Huấn luyện ResNet50 trên CIFAR10
python support\models\train.py resnet50 CIFAR10

REM Huấn luyện nhiều mô hình
for %m in (resnet50 densenet121 googlenet) do (
    for %x in (CIFAR10 MNISTC FashionC) do (
        python support\models\train.py %m %x
    )
)
```

### Build mô hình

```cmd
REM Build tất cả mô hình
docker\run-in-docker.bat python buildmodels.py
```

### Chạy thí nghiệm

```cmd
REM Tìm kiếm bit dễ bị tấn công
docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10

REM Mô phỏng tấn công
tools\runattacksim.bat
```

## Cấu trúc thư mục

```
BitShield_Clone/
├── datasets/          # Datasets đã tải về
├── models/            # Trọng số mô hình đã huấn luyện
├── built/             # Binary files đã compile
├── results/           # Kết quả thí nghiệm
├── ghidra/            # Files phân tích binary
├── compilers/         # TVM, Glow, NNFusion compilers
├── support/models/    # Định nghĩa mô hình
├── tools/             # Công cụ tiện ích
└── docker/            # Docker configuration
```

## Xử lý sự cố

### Lỗi thường gặp

1. **Python không tìm thấy**
   - Đảm bảo Python 3.8+ đã cài đặt và có trong PATH
   - Thử: `python --version`

2. **Docker không chạy**
   - Khởi động Docker Desktop
   - Kiểm tra: `docker version`

3. **Lỗi quyền truy cập**
   - Chạy Command Prompt với quyền Administrator

4. **Lỗi virtual environment**
   - Xóa thư mục `venv/` và chạy lại `setup.bat`

### Mẹo tăng hiệu suất

1. **Sử dụng ổ SSD** để tăng hiệu suất I/O
2. **Cấp phát nhiều RAM hơn** cho Docker Desktop (8GB+ khuyến nghị)
3. **Sử dụng WSL2 backend** cho Docker

## Sử dụng DVC

```cmd
REM Pull dữ liệu mới nhất
dvc pull

REM Reproduce experiments
dvc repro

REM Kiểm tra trạng thái
dvc status
```

## Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra README này trước
2. Đảm bảo tất cả prerequisites đã cài đặt
3. Kiểm tra Docker Desktop đang chạy
4. Đảm bảo đủ dung lượng ổ cứng và RAM

## Tài liệu tham khảo

- [Paper gốc](https://www.ndss-symposium.org/ndss-paper/bitshield-defending-against-bit-flip-attacks-on-dnn-executables/)
- [DVC Documentation](https://dvc.org/)
- [Docker Documentation](https://docs.docker.com/) 