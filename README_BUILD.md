# Hướng Dẫn Build File .so

Tài liệu này hướng dẫn cách build các file `.so` (shared object) từ trained models để sử dụng trong BitShield pipeline.

## 📋 Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Yêu Cầu](#yêu-cầu)
3. [Cách Sử Dụng](#cách-sử-dụng)
4. [Các Modes](#các-modes)
5. [Ví Dụ](#ví-dụ)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Tổng Quan

File `.so` là binary compiled từ trained models, được tối ưu hóa để chạy inference nhanh. BitShield hỗ trợ build với:
- **TVM**: Apache TVM compiler (khuyến nghị)
- **Glow**: Facebook Glow compiler
- **NNFusion**: Microsoft NNFusion compiler

### Output Files

Sau khi build, bạn sẽ có:
- **Binary file**: `built/{compiler}-{version}-{model}-{dataset}-{cig}-{dig}.so`
- **Output definitions**: `built-aux/output-defs/{filename}.json`

---

## 📦 Yêu Cầu

### 1. Python Environment

```bash
# Activate virtual environment (nếu có)
source ~/.venvs/tvm/bin/activate  # hoặc tvm_env/bin/activate
```

### 2. TVM Setup (cho TVM compiler)

Xem hướng dẫn setup TVM trong `README.md` hoặc `QUICK_START.md`.

Kiểm tra TVM:
```bash
python -c "import tvm; print(tvm.__version__)"
```

### 3. Trained Models

Đảm bảo bạn đã train models trước:
```
models/
  └── CIFAR10/
      ├── resnet50/
      │   └── resnet50.pt
      ├── googlenet/
      │   └── googlenet.pt
      └── densenet121/
          └── densenet121.pt
```

---

## 🚀 Cách Sử Dụng

### Phương Pháp 1: Sử Dụng Script (Khuyến Nghị)

```bash
# Cấp quyền thực thi
chmod +x build_so.sh

# Build cơ bản
./build_so.sh -m resnet50 -d CIFAR10 -I nd

# Build với các options
./build_so.sh -m resnet50 -d CIFAR10 -I gn1 -f --check-acc
```

### Phương Pháp 2: Sử Dụng Python Trực Tiếp

```bash
python buildmodels.py \
    --compiler tvm \
    --model resnet50 \
    --dataset CIFAR10 \
    --dig nd \
    --cig ncnp \
    --no-check-acc
```

---

## ⚙️ Các Modes

### DIG Modes (Defense Instrumentation)

| Mode | Mô Tả | Outputs |
|------|-------|---------|
| `nd` | No Defense - Không có defense | 1 output (predictions only) |
| `gn1` | Gradient Norm L1 | 2 outputs (predictions + L1 grad norm) |
| `gn2` | Gradient Norm L2 | 2 outputs (predictions + L2 grad norm) |
| `gninf` | Gradient Norm L∞ | 2 outputs (predictions + L∞ grad norm) |
| `id` | Input Distribution | 2 outputs (predictions + input stats) |
| `rb` | Range-based | 2 outputs (predictions + range bounds) |
| `cb` | Coverage-based | 2 outputs (predictions + coverage) |

### CIG Modes (Code Integrity Guard)

| Mode | Mô Tả |
|------|-------|
| `nc` | No CIG, có prepatch |
| `ncnp` | No CIG, no prepatch (khuyến nghị cho testing) |
| `cc1` | CoopCIG V1 |
| `cc2` | CoopCIG V2 |

---

## 📝 Ví Dụ

### 1. Build Model Cơ Bản (No Defense)

```bash
./build_so.sh -m resnet50 -d CIFAR10 -I nd
```

**Output:**
- File: `built/tvm-main-resnet50-CIFAR10-ncnp-nd.so`
- Output defs: `built-aux/output-defs/tvm-main-resnet50-CIFAR10-ncnp-nd.so.json`
- Output structure: 1 output `[batch_size, num_classes]`

### 2. Build với Gradient Norm Defense

```bash
./build_so.sh -m resnet50 -d CIFAR10 -I gn1
```

**Output:**
- File: `built/tvm-main-resnet50-CIFAR10-ncnp-gn1.so`
- Output defs: `built-aux/output-defs/tvm-main-resnet50-CIFAR10-ncnp-gn1.so.json`
- Output structure: 2 outputs
  - Output 1: `[batch_size, num_classes]` - predictions
  - Output 2: `[1]` - gradient norm (for attack detection)

### 3. Build với Force Rebuild

```bash
./build_so.sh -m resnet50 -d CIFAR10 -I nd -f
```

Sẽ xóa file cũ và build lại từ đầu.

### 4. Build với Accuracy Check

```bash
./build_so.sh -m resnet50 -d CIFAR10 -I nd --check-acc
```

Sẽ kiểm tra accuracy sau khi build (cần có test dataset).

### 5. Build Multiple Models

```bash
# Build tất cả models trên CIFAR10
for model in resnet50 googlenet densenet121; do
    ./build_so.sh -m $model -d CIFAR10 -I nd
done
```

### 6. Build với Các DIG Modes Khác Nhau

```bash
# Build với các gradient norm modes
for dig in gn1 gn2 gninf; do
    ./build_so.sh -m resnet50 -d CIFAR10 -I $dig
done
```

---

## 🔧 Options Chi Tiết

### Script Options

| Option | Mô Tả | Default |
|--------|-------|---------|
| `-c, --compiler` | Compiler (tvm, glow, nnfusion) | `tvm` |
| `-v, --compiler-ver` | Compiler version | `main` |
| `-m, --model` | Model name | `resnet50` |
| `-d, --dataset` | Dataset name | `CIFAR10` |
| `-i, --cig` | CIG mode | `ncnp` |
| `-I, --dig` | DIG mode | `nd` |
| `-X, --no-avx` | Disable AVX | `false` |
| `-O, --opt-level` | Optimization level (0-3) | `3` |
| `-A, --check-acc` | Check accuracy | `false` |
| `-f, --force` | Force rebuild | `false` |

### Python Script Options

Tương tự như script, nhưng sử dụng `--` thay vì `-`:

```bash
python buildmodels.py \
    --compiler tvm \
    --compiler_ver main \
    --model resnet50 \
    --dataset CIFAR10 \
    --cig ncnp \
    --dig nd \
    --avx \
    --opt-level 3 \
    --no-check-acc \
    --force
```

---

## 📊 Output Structure

### Mode 'nd' (No Defense)

```json
[
  {
    "shape": [20, 10],
    "dtype": "float32"
  }
]
```

- **1 output**: Predictions only
- Shape: `[batch_size, num_classes]`
- Dtype: `float32`

### Mode 'gn1' (Gradient Norm L1)

```json
[
  {
    "shape": [20, 10],
    "dtype": "float32"
  },
  {
    "shape": [1],
    "dtype": "float32"
  }
]
```

- **2 outputs**:
  1. Predictions: `[batch_size, num_classes]`
  2. Gradient norm: `[1]` (scalar)

---

## 🐛 Troubleshooting

### 1. TVM Not Available

**Error:**
```
RuntimeError: TVM runtime không khả dụng
```

**Solution:**
```bash
# Activate TVM environment
source ~/.venvs/tvm/bin/activate

# Hoặc install TVM
# Xem README.md phần "Cài đặt TVM"
```

### 2. Model File Not Found

**Error:**
```
FileNotFoundError: Model file not found: models/CIFAR10/resnet50/resnet50.pt
```

**Solution:**
```bash
# Train model trước
python train_all_models.py --model resnet50 --dataset CIFAR10
```

### 3. Build Fails với Import Error

**Error:**
```
ImportError: No module named 'modman'
```

**Solution:**
```bash
# Đảm bảo đang ở đúng directory
cd /path/to/BitShield_Clone

# Kiểm tra Python path
python -c "import sys; print(sys.path)"
```

### 4. File Already Exists

**Message:**
```
Skipping building ... (use --force to rebuild)
```

**Solution:**
```bash
# Sử dụng --force để rebuild
./build_so.sh -m resnet50 -d CIFAR10 -I nd -f
```

### 5. Build Takes Too Long

**Normal build time:**
- ResNet50: ~50-60 giây
- GoogLeNet: ~40-50 giây
- DenseNet121: ~60-70 giây

**Nếu build quá lâu:**
- Giảm `--opt-level` (từ 3 xuống 2 hoặc 1)
- Disable AVX: `--no-avx`
- Kiểm tra system resources

---

## 📚 Tài Liệu Tham Khảo

- **README.md**: Tổng quan về BitShield
- **QUICK_START.md**: Hướng dẫn nhanh
- **buildmodels.py**: Source code của build script
- **modman.py**: Module management và TVM integration

---

## ✅ Checklist Trước Khi Build

- [ ] TVM đã được cài đặt và activate
- [ ] Model files đã được train và lưu trong `models/`
- [ ] Python environment đã được setup đúng
- [ ] Đã test import các modules: `python -c "import modman; import buildmodels"`
- [ ] Đã kiểm tra disk space (mỗi .so file ~90MB)

---

## 🎓 Best Practices

1. **Luôn build mode 'nd' trước** để test basic functionality
2. **Sử dụng `--no-check-acc`** khi build nhiều files để tiết kiệm thời gian
3. **Sử dụng `--force`** khi cần rebuild sau khi thay đổi code
4. **Build từng model một** để dễ debug nếu có lỗi
5. **Kiểm tra output files** sau khi build để đảm bảo thành công

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra `Troubleshooting` section
2. Xem logs trong terminal output
3. Kiểm tra file `built/` và `built-aux/` để xem files đã được tạo chưa
4. Chạy `python check_setup.py` để verify setup

---

**Happy Building! 🚀**

