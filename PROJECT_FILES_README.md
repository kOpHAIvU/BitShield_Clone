## Tổng quan các tệp và thư mục trong dự án

- Mục tiêu: Mô tả ngắn gọn vai trò/chức năng của các tệp chính để bạn nhanh nắm được cấu trúc dự án.
- Ghi chú: Những thư mục có nhiều tệp con được mô tả theo nhóm; các tệp ít quan trọng hoặc sinh ra trong quá trình chạy (logs, models, results) chỉ nêu mục đích chung.

### Thư mục gốc
- `attack_with_defense_extended.py`: Trình mô phỏng tấn công và đánh giá phòng thủ chính (DIG/CIG/combined) cho dữ liệu tabular (IoTID20/WUSTL/CICIoT2023). Hỗ trợ các chế độ tấn công: noise, PBS, RandomFlip, PBS→Random, Random→PBS; có log chi tiết theo vòng và xuất CSV.
- TÍCH HỢP OBFUS-SIG-NIDS: thêm lựa chọn `--obfus-sig` để kích hoạt Obfuscation + SIG-Lite + Bit-Fingerprint theo bài báo, với các tham số:
  - `--sig-period`, `--sig-k`, `--sig-grad {l1|l2}`, `--sig-norm`
  - `--fp-threshold` (PSI), `--fp-entropy-th` (entropy bit-plane)
  - `--obfus-mode {or|and}`, `--obfus-shadow`
  - Ví dụ:
  ```
  python attack_with_defense_extended.py dig ResNetSEBlockIoT CICIoT2023 \
    --device cpu --attack-mode pbs --attack-iters 25 \
    --obfus-sig --sig-period 500 --sig-k 3.0 --sig-grad l1 --sig-norm \
    --fp-threshold 0.1 --fp-entropy-th 0.15 --obfus-mode or
  ```
- `attack_with_defense.py`: Phiên bản trước/simplified của kịch bản tấn công + phòng thủ.
- `attack_with_defense_updated.py`: Bản trung gian dùng để so sánh/đối chiếu logic với bản extended.
- `attacksim.py`: Mô phỏng tấn công lật bit ở mức binary (DRAM/binary-level) với dữ liệu sweep + Ghidra, đánh giá hệ quả và phát hiện DIG/CIG thực tế (validator `fliptest.py`). Xuất kết quả pickle.
- `analysis.py`: Tiện ích phân tích dữ liệu sweep/kết quả tấn công ở mức bit/byte; chuyển đổi sang DataFrame, tính toán các chỉ số (độ mạnh bit, độ rơi chính xác, …).
- `backward.py`, `buildmodels.py`, `inst.py`, `modman.py`, `utils.py`: Bộ công cụ hỗ trợ build mô hình, thao tác module TVM/PyTorch, và tiện ích chung.
- `cfg.py`: Cấu hình đường dẫn/tham số hệ thống (thư mục models, results, công cụ, Ghidra, …).
- `simple_train_test.py`, `run_experiments.py`, `simple_sweep_attack.py`, `fliptest.py`, `flipsweep.py`, `fliputils.py`, `fliptest.py` (nếu có): Mã thực nghiệm/phụ trợ cho các pipeline thử nghiệm tấn công hoặc sweep bit.
- `wbbfa.py`: Tấn công lật bit dựa trên trọng số mô hình (Weight-Based Bit Flip Attack) ở mức PyTorch; có tích hợp DIG khi cần.
- `cig.py`: Cài “Code Integrity Guard” thực tế ở mức mã máy (TVM `.so`) bằng PeachPy/Ghidra; sinh mã kiểm tra (BasicCIG/Adler32CIG), quản lý vị trí chèn và patch binary.
- `record.py`, `prune.py`, `backward.py`: Tiện ích huấn luyện/ghi log/tối ưu mô hình (tùy workflow).
- `run_simple.bat`, `run_extended.bat`, `run_complete_workflow.bat`, `test_docker.bat`, `run_docker.bat`, `setup.bat`, `env.bat`, `check_windows.bat`: Script Windows hỗ trợ thiết lập/chạy nhanh.
- `DOCKER_GUIDE.md`, `README.md`, `QUICK_START.md`, `README_IoTID20_Training.md`, `README_Windows.md`: Tài liệu hướng dẫn, quick start, và ghi chú môi trường.

### Thư mục `support/`
- `support/dataman_extended.py`: Data loader cho các dataset tabular (IoTID20/WUSTL/CICIoT2023), trả về `DataLoader` train/test theo batch.
- `support/torchdig_tabular.py`: Triển khai DIG cho tabular (tính điểm nghi ngờ: gradient/entropy/feature/statistics) + tính khoảng nghi ngờ “bình thường” để phát hiện.
- `support/torchdig.py`: Phiên bản DIG gốc (cho các bài toán không tabular/không mở rộng).
- `support/models/` (20+ file): Định nghĩa các mô hình PyTorch (ví dụ `ResNetSEBlockIoT`, `EfficientCNN`, `SimpleCNNIoT`, …) và các lớp lượng tử hóa (`quan_Linear`, `quan_Conv1d`, `CustomBlock`).
- `support/obfus_sig/`: Kiến trúc OBFUS-SIG-NIDS (3 tầng, thuần mô hình):
  - `obfus_adapter.py`: `ObfusAdapter`, `ObfusPair` (permute + inverse), `wrap_last_linear_with_obfus(...)`
  - `sig_lite.py`: `SigLiteMonitor` (D_KL(u||ŷ), ‖∂KL/∂W_last‖ with median±k·MAD, L1/L2, normalize)
  - `bit_fingerprint.py`: Fingerprint histogram int8 + entropy bit-plane, cảnh báo bằng PSI/entropy drift
  - `controller.py`: Gộp alert (OR/AND), cooldown, reseed adapters
  - `runtime.py`: `ObfusSigRuntime` gắn tất cả và cung cấp API `calibrate()`/`periodic_check()`
- `support/dataman_iotid20.py`, `support/dataman_extended.py`: Bộ tải dữ liệu (tiền xử lý, scaler/encoder nếu cần).
- `support/demo_improved_training.py`, `support/demo_iotid20_training.py`: Ví dụ/trình diễn huấn luyện nhanh.
- `support/torchdig_tabular.py`: Đã chỉnh sửa để nhất quán thiết bị (CPU/GPU), ngăn lỗi device-mismatch khi tính điểm nghi ngờ.

### Thư mục `Integrated_BFA_BitShield/`
- `defense_bitshield.py`: Logic bảo vệ BitShield tích hợp (tham khảo cho cách áp dụng DIG/CIG).
- `custom_models.py`, `modelCNN.py`: Mô hình tham khảo cho pipeline BitShield.
- `attack_bfa_random.py`: Tấn công bit flip ngẫu nhiên (BFA) tích hợp pipeline.
- `run_pipeline.py`: Chạy pipeline tích hợp BFA + Defense theo cấu hình.
- `Dataset_WUSTL.ipynb`, `train_IoTID20.ipynb`, `TrainModel_CIC2023.ipynb`: Notebook huấn luyện/tấn công tham khảo (PBS/RandomFlip và biến thể).
- Thư mục `data/`, `logs/`: Dữ liệu/nhật ký liên quan pipeline này.

### Thư mục `eval/`
- `eval/evalutils.py`: Hàm đánh giá/tiện ích đo lường, lấy khoảng nghi ngờ (sus_score_range), đo accuracy binary `.so`, …

### Thư mục `ghidra/`
- Script tự động hóa Ghidra để phân tích binary, trích xuất phân tích, xác định vị trí chèn CIG, v.v. (ví dụ `export-analysis.py`, `find-cig-spots.py`, `import-binaries.sh`, …).

### Thư mục `tools/`
- Tiện ích chạy lẻ: benchmark, trích xuất graph, build intermediate representation, thống kê sweep, vv.
  - `tools/extract_graph_json.py`: Xuất JSON mô tả graph từ binary.
  - `tools/build_as_ir.py`, `tools/host-build-tvm.sh`: Hỗ trợ build TVM/IR.
  - `tools/benchmark_perf.py`: Đo hiệu năng (benchmark).
  - `tools/sweeps2csv.py`, `tools/truncate-sweep-pkl.py`, `tools/update_sweep_pkl.py`: Chuyển đổi/tiền xử lý dữ liệu sweep.
  - `tools/rh_attack.py`, `tools/runattacksim.*`: Chạy tấn công mô phỏng nhanh (Rowhammer/attacksim).

### Thư mục `resources/`
- Mã nguồn C++/headers hỗ trợ unmasker, glue code TVM/NNFusion/Glow; cấu hình build.
  - `resources/unmasker/`: `getkey.cc`, `unmasker.cc` (CIG V2), headers `dlpack.h`, `c_runtime_api.h`.
  - `resources/nnfusion/`, `resources/glow/`: Cấu hình build/khởi tạo cho các backend.

### Thư mục `support/dataset/`
- Dữ liệu/mẫu/bộ tập (CICIoT2023, IoTID20, WUSTL) – cấu trúc theo dataset; phục vụ loader trong `support/dataman_extended.py`.

### Thư mục `models/`
- Chứa mô hình đã huấn luyện (`.pt`) theo dataset/model (ví dụ `IoTID20/ResNetSEBlockIoT/…`). Được nạp bởi `load_model(...)` trong các script.

### Thư mục `results/`
- `results/defense_results/`: JSON/CSV kết quả đánh giá DIG/CIG (bao gồm per-iteration CSV cho bit‑flip).
- `results/attack_results/`, `results/original_dig_results/`, `results/sweep_results/`: Kết quả các thí nghiệm khác nhau.

### Thư mục `logs/`
- Lưu log sự kiện (`events.jsonl`) và `run.log` khi chạy các pipeline.

### Thư mục `docker/`
- Dockerfile, script khởi tạo/chạy môi trường (Linux/Windows) cho toàn bộ toolchain (Ghidra, TVM, PeachPy, …).

### Các tệp khác đáng chú ý
- `train_imbalanced_fix.py`: Huấn luyện/khắc phục mất cân bằng dữ liệu (tabular).
- `test_model.pth`, `test_resnet.pth`, `ResNetSEBlockIoT_WUSTL_best_imbalanced.pt`: Trọng số/mô hình mẫu.
- `pyrightconfig.json`, `dvc.yaml`: Cấu hình linter/type-check và data versioning (nếu dùng).

---

## Gợi ý chạy nhanh
- DIG + PBS (25 vòng) cho CICIoT2023, chạy CPU:
```
python attack_with_defense_extended.py dig ResNetSEBlockIoT CICIoT2023 --device cpu --attack-mode pbs --attack-iters 25
```
- CIG + RandomFlip (25 vòng) và xuất CSV per-iteration:
```
python attack_with_defense_extended.py cig ResNetSEBlockIoT CICIoT2023 --device cpu --attack-mode random_flip --attack-iters 25
```

## Liên hệ giữa các phần
- `attack_with_defense_extended.py` dùng loader trong `support/dataman_extended.py`, mô hình trong `support/models/`, và DIG tabular từ `support/torchdig_tabular.py`. Kết quả/CSV lưu ở `results/defense_results/`.
- `attacksim.py` hoạt động ở mức binary, dựa vào phân tích Ghidra (`ghidra/`), tiện ích `eval/evalutils.py`, và có thể kết hợp CIG thực tế trong `cig.py`.

Nếu bạn cần version “dài” (liệt kê từng file con chi tiết theo thư mục), hãy nói dataset hoặc mảng chức năng bạn quan tâm, mình sẽ mở rộng mục tương ứng.


