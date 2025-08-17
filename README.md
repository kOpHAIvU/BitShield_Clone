# BitShield - Bảo vệ chống lại tấn công Bit-Flip trên DNN Executables

## Tổng quan

BitShield là một dự án nghiên cứu bảo vệ chống lại các cuộc tấn công bit-flip trên các file thực thi mạng nơ-ron sâu (DNN). **Dự án này được tối ưu hóa để chạy hoàn toàn trên Windows** và cung cấp hai chế độ sử dụng:

- **Chế độ đơn giản**: Không cần Docker, chỉ cần Python và Git
- **Chế độ đầy đủ**: Với Docker để có tất cả tính năng

## Yêu cầu hệ thống

### Phần mềm bắt buộc

1. **Python 3.8+** - Tải từ [python.org](https://www.python.org/downloads/)
   - Đảm bảo tích hợp "Add Python to PATH" khi cài đặt
   - Kiểm tra: `python --version`

2. **Git** - Tải từ [git-scm.com](https://git-scm.com/download/win)
   - Kiểm tra: `git --version`

### Phần mềm tùy chọn (cho tính năng đầy đủ)

3. **Docker Desktop** - Tải từ [docker.com](https://www.docker.com/products/docker-desktop/)
   - Cần thiết để build models và chạy experiments nâng cao
   - Đảm bảo Docker Desktop đang chạy trước khi sử dụng

### Yêu cầu phần cứng

- **RAM tối thiểu**: 8GB
- **RAM khuyến nghị**: 16GB+
- **Dung lượng ổ cứng**: 50GB+ trống
- **Ổ cứng**: SSD được khuyến nghị để tăng hiệu suất

## Cài đặt nhanh

### Bước 1: Clone và cài đặt

```cmd
git clone <repository-url>
cd BitShield_Clone
setup.bat
```

Hoặc thực hiện từng bước:

```cmd
setup.bat
env.bat
python tools\ensure_datasets.py
```

### Bước 2: Cài đặt Docker (tùy chọn)

```cmd
docker\setup.bat
```

## Sử dụng

### Kích hoạt môi trường

Trước khi làm việc với dự án, luôn kích hoạt môi trường:

```cmd
env.bat
```

### Cách 1: Sử dụng đơn giản (không cần Docker)

```cmd
REM Huấn luyện mô hình
run_simple.bat train resnet50 CIFAR10

REM Test mô hình
run_simple.bat test resnet50 CIFAR10

REM Xem danh sách mô hình và datasets
run_simple.bat list-models
run_simple.bat list-datasets
```

### Cách 2: Sử dụng đầy đủ (với Docker)

```cmd
REM Huấn luyện mô hình
python support\models\train.py resnet50 CIFAR10

REM Build mô hình
docker\run-in-docker.bat python buildmodels.py

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
├── docker/            # Docker configuration
├── run_simple.bat     # Script chạy đơn giản
├── setup.bat          # Script cài đặt
└── env.bat            # Script kích hoạt môi trường
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
   - Kiểm tra quyền truy cập file

4. **Lỗi git submodule**
   - Chạy: `git submodule update --init --recursive`

5. **Lỗi virtual environment**
   - Xóa thư mục `venv/` và chạy lại `setup.bat`

### Mẹo tăng hiệu suất

1. **Sử dụng ổ SSD** để tăng hiệu suất I/O
2. **Cấp phát nhiều RAM hơn** cho Docker Desktop (8GB+ khuyến nghị)
3. **Sử dụng WSL2 backend** cho Docker (hiệu suất tốt hơn)

## Sử dụng DVC

Dự án sử dụng DVC để quản lý version dữ liệu:

```cmd
REM Pull dữ liệu mới nhất
dvc pull

REM Reproduce experiments
dvc repro

REM Kiểm tra trạng thái
dvc status
```

## Phát triển

### Thêm mô hình mới

1. Thêm định nghĩa mô hình trong `support/models/`
2. Cập nhật `cfg.py` với cấu hình mô hình mới
3. Huấn luyện mô hình bằng `support/models/train.py`
4. Build bằng `buildmodels.py`

### Thêm dataset mới

1. Thêm dataset vào `tools/ensure_datasets.py`
2. Cập nhật `cfg.py` với cấu hình dataset
3. Cập nhật `utils.py` nếu cần cho thuộc tính dataset-specific

## Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra README này trước
2. Đảm bảo tất cả prerequisites đã cài đặt
3. Kiểm tra Docker Desktop đang chạy (nếu sử dụng)
4. Đảm bảo đủ dung lượng ổ cứng và RAM

## Tài liệu tham khảo

- [Paper gốc](https://www.ndss-symposium.org/ndss-paper/bitshield-defending-against-bit-flip-attacks-on-dnn-executables/)
- [DVC Documentation](https://dvc.org/)
- [Docker Documentation](https://docs.docker.com/)

## Lưu ý quan trọng

- **Bắt đầu với cách 1** (không cần Docker) để làm quen với dự án
- **Chỉ sử dụng Docker** khi cần các tính năng nâng cao như build binary và phân tích bit-flip
- **Dự án được tối ưu hóa cho Windows** và đã được test kỹ lưỡng
