# BitShield_Clone - Flow Charts Chi Tiết

## Tổng Quan Dự Án

BitShield là một dự án nghiên cứu bảo vệ chống lại các cuộc tấn công bit-flip trên các file thực thi mạng nơ-ron sâu (DNN). Dự án cung cấp hai chế độ sử dụng:
- **Chế độ đơn giản**: Không cần Docker, chỉ cần Python và Git
- **Chế độ đầy đủ**: Với Docker để có tất cả tính năng

---

## 1. Flow Chart Tổng Quan Dự Án

```mermaid
graph TD
    A[Start: Clone Repository] --> B{Chọn Chế Độ}
    
    B -->|Đơn Giản| C[Setup Python Environment]
    B -->|Đầy Đủ| D[Setup Docker Environment]
    
    C --> E[Download Datasets]
    D --> E
    
    E --> F[Train Models]
    F --> G[Test Models]
    
    G --> H{Chế Độ Đầy Đủ?}
    H -->|Không| I[Kết Thúc - Chế Độ Đơn Giản]
    H -->|Có| J[Build Binary Files]
    
    J --> K[Bit-Flip Sweep Analysis]
    K --> L[Attack Simulation]
    L --> M[Ghidra Analysis]
    M --> N[Generate Results]
    N --> O[Kết Thúc - Chế Độ Đầy Đủ]
    
    style A fill:#e1f5fe
    style I fill:#c8e6c9
    style O fill:#c8e6c9
    style B fill:#fff3e0
    style H fill:#fff3e0
```

---

## 2. Flow Chart Cài Đặt và Setup

```mermaid
graph TD
    A[Clone Repository] --> B[Chạy setup.bat]
    B --> C{Kiểm Tra Prerequisites}
    
    C -->|Thiếu| D[Cài Đặt Python 3.8+]
    C -->|Thiếu| E[Cài Đặt Git]
    C -->|Thiếu| F[Cài Đặt Docker Desktop]
    
    D --> G[Tạo Virtual Environment]
    E --> G
    F --> G
    
    G --> H[Cài Đặt Dependencies]
    H --> I[Download Datasets]
    
    I --> J{Chế Độ Docker?}
    J -->|Có| K[Build Docker Image]
    J -->|Không| L[Setup Hoàn Tất - Chế Độ Đơn Giản]
    
    K --> M[Setup Hoàn Tất - Chế Độ Đầy Đủ]
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style M fill:#c8e6c9
    style C fill:#fff3e0
    style J fill:#fff3e0
```

---

## 3. Flow Chart Huấn Luyện Mô Hình

```mermaid
graph TD
    A[Start Training] --> B[Chọn Model & Dataset]
    
    B --> C{Model Type}
    C -->|TorchVision| D[Load TorchVision Model]
    C -->|Custom| E[Load Custom Model]
    
    D --> F[Setup Data Loaders]
    E --> F
    
    F --> G[Initialize Optimizer & Loss]
    G --> H[Training Loop]
    
    H --> I[Forward Pass]
    I --> J[Calculate Loss]
    J --> K[Backward Pass]
    K --> L[Update Parameters]
    
    L --> M{End of Epoch?}
    M -->|Không| I
    M -->|Có| N[Validation]
    
    N --> O{Accuracy OK?}
    O -->|Không| P{Max Epochs?}
    O -->|Có| Q[Save Model]
    
    P -->|Không| H
    P -->|Có| Q
    
    Q --> R[Training Complete]
    
    style A fill:#e1f5fe
    style R fill:#c8e6c9
    style C fill:#fff3e0
    style M fill:#fff3e0
    style O fill:#fff3e0
    style P fill:#fff3e0
```

---

## 4. Flow Chart Build Binary Files

```mermaid
graph TD
    A[Start Build Process] --> B[Load Trained Model]
    B --> C[Convert to IR Module]
    
    C --> D{Compiler Type}
    D -->|TVM| E[TVM Compilation]
    D -->|Glow| F[Glow Compilation]
    D -->|NNFusion| G[NNFusion Compilation]
    
    E --> H[Instrument Module]
    F --> H
    G --> H
    
    H --> I[Add Coverage Tracking]
    I --> J[Add DIG Protection]
    J --> K[Add CIG Protection]
    
    K --> L[Build Binary]
    L --> M[Save Binary File]
    M --> N[Generate Output Definitions]
    
    N --> O[Check Accuracy]
    O --> P{Accuracy > 0.6?}
    P -->|Không| Q[Build Failed]
    P -->|Có| R[Build Success]
    
    style A fill:#e1f5fe
    style Q fill:#ffcdd2
    style R fill:#c8e6c9
    style D fill:#fff3e0
    style P fill:#fff3e0
```

---

## 5. Flow Chart Bit-Flip Sweep Analysis

```mermaid
graph TD
    A[Start Bit-Flip Sweep] --> B[Load Binary File]
    B --> C[Load Test Dataset]
    
    C --> D[Initialize Results Storage]
    D --> E[Get Binary Info]
    
    E --> F[Calculate Total Bits]
    F --> G[Setup Progress Tracking]
    
    G --> H[For Each Bit Position]
    H --> I[Flip Bit]
    I --> J[Run Inference]
    
    J --> K[Calculate Metrics]
    K --> L[Accuracy Change]
    L --> M[Top Label Change]
    M --> N[LPIPS Score]
    N --> O[FID Score]
    
    O --> P[Calculate Suspicious Score]
    P --> Q[Store Results]
    
    Q --> R{More Bits?}
    R -->|Có| H
    R -->|Không| S[Save Sweep Results]
    
    S --> T[Sweep Complete]
    
    style A fill:#e1f5fe
    style T fill:#c8e6c9
    style H fill:#fff3e0
    style R fill:#fff3e0
```

---

## 6. Flow Chart Attack Simulation

```mermaid
graph TD
    A[Start Attack Simulation] --> B[Load Sweep Results]
    B --> C[Initialize Memory Model]
    
    C --> D[Setup Vulnerable Bits]
    D --> E[Load Victim Binary]
    
    E --> F[For Each Attack Scenario]
    F --> G[Select Target Bit]
    G --> H[Apply Bit Flip]
    
    H --> I[Run Binary]
    I --> J[Check Output]
    
    J --> K{Attack Success?}
    K -->|Có| L[Record Success]
    K -->|Không| M[Check Detection]
    
    M --> N{DIG Detected?}
    N -->|Có| O[Record DIG Detection]
    N -->|Không| P{CIG Detected?}
    
    P -->|Có| Q[Record CIG Detection]
    P -->|Không| R[Record No Effect]
    
    L --> S{More Scenarios?}
    O --> S
    Q --> S
    R --> S
    
    S -->|Có| F
    S -->|Không| T[Generate Attack Report]
    
    T --> U[Attack Simulation Complete]
    
    style A fill:#e1f5fe
    style U fill:#c8e6c9
    style F fill:#fff3e0
    style K fill:#fff3e0
    style N fill:#fff3e0
    style P fill:#fff3e0
    style S fill:#fff3e0
```

---

## 7. Flow Chart Ghidra Analysis

```mermaid
graph TD
    A[Start Ghidra Analysis] --> B[Import Binary Files]
    B --> C[Setup Ghidra Project]
    
    C --> D[Analyze Binary Structure]
    D --> E[Extract Instructions]
    
    E --> F[Map Bit Positions]
    F --> G[Identify Critical Sections]
    
    G --> H[Export Analysis Data]
    H --> I[Generate Offset Maps]
    
    I --> J[Create Instruction Database]
    J --> K[Save Analysis Results]
    
    K --> L[Ghidra Analysis Complete]
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
```

---

## 8. Flow Chart Complete Workflow

```mermaid
graph TD
    A[Start Complete Workflow] --> B[Setup Environment]
    B --> C[Download Datasets]
    
    C --> D[Train Model]
    D --> E[Test Model]
    
    E --> F{Build Success?}
    F -->|Không| G[Continue Without Build]
    F -->|Có| H[Build Binary]
    
    G --> I[Skip Binary Analysis]
    H --> J[Bit-Flip Sweep]
    
    I --> K[Workflow Complete - Simple Mode]
    J --> L[Attack Simulation]
    
    L --> M[Ghidra Analysis]
    M --> N[Generate Final Results]
    
    N --> O[Workflow Complete - Full Mode]
    
    style A fill:#e1f5fe
    style K fill:#c8e6c9
    style O fill:#c8e6c9
    style F fill:#fff3e0
```

---

## 9. Flow Chart Data Management

```mermaid
graph TD
    A[Data Management] --> B{Data Type}
    
    B -->|Datasets| C[Download Datasets]
    B -->|Models| D[Save Trained Models]
    B -->|Binaries| E[Store Built Binaries]
    B -->|Results| F[Save Analysis Results]
    
    C --> G[datasets/]
    D --> H[models/]
    E --> I[built/]
    F --> J[results/]
    
    G --> K[DVC Version Control]
    H --> K
    I --> K
    J --> K
    
    K --> L[Data Management Complete]
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style B fill:#fff3e0
```

---

## 10. Flow Chart Error Handling

```mermaid
graph TD
    A[Process Start] --> B[Execute Operation]
    B --> C{Success?}
    
    C -->|Có| D[Continue]
    C -->|Không| E[Error Detected]
    
    E --> F{Error Type}
    F -->|Docker| G[Check Docker Status]
    F -->|Python| H[Check Python Installation]
    F -->|Memory| I[Check Available Memory]
    F -->|Permission| J[Check File Permissions]
    F -->|Network| K[Check Network Connection]
    
    G --> L[Restart Docker]
    H --> M[Reinstall Python]
    I --> N[Free Memory]
    J --> O[Run as Administrator]
    K --> P[Check Internet]
    
    L --> Q[Retry Operation]
    M --> Q
    N --> Q
    O --> Q
    P --> Q
    
    Q --> R{Retry Success?}
    R -->|Có| D
    R -->|Không| S[Log Error & Exit]
    
    D --> T[Process Complete]
    
    style A fill:#e1f5fe
    style T fill:#c8e6c9
    style S fill:#ffcdd2
    style C fill:#fff3e0
    style F fill:#fff3e0
    style R fill:#fff3e0
```

---

## 11. Flow Chart Model Configuration

```mermaid
graph TD
    A[Model Configuration] --> B{Model Type}
    
    B -->|ResNet50| C[ResNet50 Config]
    B -->|DenseNet121| D[DenseNet121 Config]
    B -->|GoogLeNet| E[GoogLeNet Config]
    B -->|LeNet1| F[LeNet1 Config]
    
    C --> G[Setup Model Parameters]
    D --> G
    E --> G
    F --> G
    
    G --> H{Dataset Type}
    H -->|CIFAR10| I[CIFAR10 Config]
    H -->|MNIST| J[MNIST Config]
    H -->|FashionC| K[FashionC Config]
    H -->|ImageNet| L[ImageNet Config]
    
    I --> M[Configure Data Loaders]
    J --> M
    K --> M
    L --> M
    
    M --> N[Set Training Parameters]
    N --> O[Model Configuration Complete]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style B fill:#fff3e0
    style H fill:#fff3e0
```

---

## 12. Flow Chart Protection Mechanisms

```mermaid
graph TD
    A[Protection Mechanisms] --> B{Protection Type}
    
    B -->|DIG| C[DIG Protection]
    B -->|CIG| D[CIG Protection]
    B -->|Combined| E[Combined Protection]
    
    C --> F[Add Detection Logic]
    D --> G[Add Coverage Tracking]
    E --> H[Add Both Mechanisms]
    
    F --> I[Instrument Code]
    G --> I
    H --> I
    
    I --> J[Generate Protected Binary]
    J --> K[Test Protection]
    
    K --> L{Protection Working?}
    L -->|Có| M[Protection Active]
    L -->|Không| N[Debug Protection]
    
    N --> O[Fix Issues]
    O --> I
    
    M --> P[Protection Complete]
    
    style A fill:#e1f5fe
    style P fill:#c8e6c9
    style B fill:#fff3e0
    style L fill:#fff3e0
```

---

## 13. Flow Chart Logic Mã Nguồn - BinaryInfo Class

```mermaid
graph TD
    A[BinaryInfo Class] --> B[Initialize with Parameters]
    B --> C[compiler, compiler_ver, model_name, dataset]
    C --> D[cig, dig, avx, opt_level]
    
    D --> E[Property Methods]
    E --> F[core_model_name]
    E --> G[fname - Generate filename]
    E --> H[fpath - Get file path]
    E --> I[nchans - Number of channels]
    E --> J[nclasses - Number of classes]
    E --> K[input_img_size - Input image size]
    E --> L[output_img_size - Output image size]
    
    F --> M[Model Configuration Logic]
    G --> N[File Naming Logic]
    H --> O[Path Resolution Logic]
    I --> P[Channel Configuration]
    J --> Q[Class Configuration]
    K --> R[Size Configuration]
    L --> S[Output Configuration]
    
    M --> T[BinaryInfo Ready]
    N --> T
    O --> T
    P --> T
    Q --> T
    R --> T
    S --> T
    
    style A fill:#e1f5fe
    style T fill:#c8e6c9
    style E fill:#fff3e0
```

---

## 14. Flow Chart Logic Mã Nguồn - Data Management

```mermaid
graph TD
    A[Data Management Logic] --> B{Data Type}
    
    B -->|MergedDataset| C[Create MergedDataset]
    B -->|Standard Dataset| D[Load Standard Dataset]
    B -->|Custom Dataset| E[Load Custom Dataset]
    
    C --> F[Load Multiple Datasets]
    F --> G[Combine Data References]
    G --> H[Calculate Mean/Std]
    H --> I[Apply Transforms]
    
    D --> J[Load Single Dataset]
    J --> K[Apply Standard Transforms]
    
    E --> L[Load Custom Data]
    L --> M[Apply Custom Transforms]
    
    I --> N[Create DataLoader]
    K --> N
    M --> N
    
    N --> O[Data Management Complete]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style B fill:#fff3e0
```

---

## 15. Flow Chart Logic Mã Nguồn - CIG Protection

```mermaid
graph TD
    A[CIG Protection Logic] --> B[CodeIntegrityGuard Class]
    B --> C[Initialize with checked_offset, patch_offset]
    
    C --> D[Encode Protection Code]
    D --> E[Generate Checksum]
    E --> F[Create Guard Instructions]
    
    F --> G[SpacemakerPass]
    G --> H[Add Random Instructions]
    H --> I[Create CIG Spots]
    
    I --> J[Prepatch Function]
    J --> K[Discover CIG Spots]
    K --> L[Replace with NOP]
    L --> M[Save Spots Info]
    
    M --> N[Apply Protection]
    N --> O[Insert Guard Code]
    O --> P[Verify Integrity]
    
    P --> Q{CIG Protection Active?}
    Q -->|Có| R[CIG Protection Complete]
    Q -->|Không| S[Debug CIG Issues]
    
    S --> T[Fix CIG Problems]
    T --> N
    
    style A fill:#e1f5fe
    style R fill:#c8e6c9
    style Q fill:#fff3e0
```

---

## 16. Flow Chart Logic Mã Nguồn - Instrumentation

```mermaid
graph TD
    A[Instrumentation Logic] --> B[CovModeConfig Class]
    B --> C[Setup Coverage Mode]
    
    C --> D{Instrumentation Type}
    D -->|Basic Coverage| E[lcov_basic_diff]
    D -->|Dual Bounds| F[extra_params_dual_bounds]
    D -->|Custom Coverage| G[Custom Coverage Logic]
    
    E --> H[Layer Coverage Type Transform]
    F --> I[Extra Parameters Setup]
    G --> J[Custom Coverage Setup]
    
    H --> K[Layer Output to Coverage]
    I --> L[Bounds Parameters]
    J --> M[Custom Parameters]
    
    K --> N[Coverage Calculation]
    L --> N
    M --> N
    
    N --> O[Overall Coverage]
    O --> P[Coverage Validation]
    
    P --> Q{Coverage Valid?}
    Q -->|Có| R[Instrumentation Complete]
    Q -->|Không| S[Fix Coverage Issues]
    
    S --> T[Adjust Parameters]
    T --> N
    
    style A fill:#e1f5fe
    style R fill:#c8e6c9
    style Q fill:#fff3e0
```

---

## 17. Flow Chart Logic Mã Nguồn - Bit Flip Utilities

```mermaid
graph TD
    A[Bit Flip Utilities] --> B[LoadedModInfo Class]
    B --> C[Load Module Information]
    
    C --> D[parse_maps_line]
    D --> E[Parse Memory Maps]
    E --> F[Extract Region Info]
    
    F --> G[load_mod Function]
    G --> H[Load Built Model]
    H --> I[Find Target Region]
    I --> J[Set Memory Permissions]
    
    J --> K[flip_bits Function]
    K --> L[Read Memory Content]
    L --> M[Apply Bit Flips]
    M --> N[Write Modified Content]
    N --> O[Recover Permissions]
    
    O --> P[random_flip_bits]
    P --> Q[Generate Random Flips]
    Q --> R[Apply Random Changes]
    
    R --> S{Bit Flip Success?}
    S -->|Có| T[Bit Flip Complete]
    S -->|Không| U[Handle Flip Error]
    
    U --> V[Log Error]
    V --> W[Retry or Abort]
    
    style A fill:#e1f5fe
    style T fill:#c8e6c9
    style S fill:#fff3e0
```

---

## 18. Flow Chart Logic Mã Nguồn - Model Building Process

```mermaid
graph TD
    A[Model Building Process] --> B[buildmodels.py]
    B --> C[Load Configuration]
    
    C --> D[For Each Binary Info]
    D --> E[Get IR Module]
    E --> F{Module Type}
    
    F -->|DIG Instrumented| G[Add DIG Protection]
    F -->|CIG Instrumented| H[Add CIG Protection]
    F -->|Standard| I[Standard Build]
    
    G --> J[Instrument DIG Logic]
    H --> K[Instrument CIG Logic]
    I --> L[Standard Compilation]
    
    J --> M[Build Module]
    K --> M
    L --> M
    
    M --> N[Check Accuracy]
    N --> O{Accuracy > Threshold?}
    O -->|Không| P[Build Failed]
    O -->|Có| Q[Save Binary]
    
    Q --> R[Generate Output Defs]
    R --> S{More Binaries?}
    S -->|Có| D
    S -->|Không| T[Build Process Complete]
    
    style A fill:#e1f5fe
    style T fill:#c8e6c9
    style O fill:#fff3e0
    style S fill:#fff3e0
```

---

## 19. Flow Chart Logic Mã Nguồn - Sweep Analysis Process

```mermaid
graph TD
    A[Sweep Analysis Process] --> B[flipsweep.py]
    B --> C[Initialize FlipResult Classes]
    
    C --> D[Load Binary File]
    D --> E[Setup Test Dataset]
    E --> F[Calculate Total Bits]
    
    F --> G[For Each Bit Position]
    G --> H[Create FlipResult]
    H --> I[Flip Specific Bit]
    
    I --> J[Run Inference]
    J --> K[Calculate Metrics]
    K --> L[Accuracy Change]
    L --> M[Top Label Change]
    M --> N[LPIPS Score]
    N --> O[FID Score]
    
    O --> P[Calculate Suspicious Score]
    P --> Q[Store Results]
    
    Q --> R{More Bits?}
    R -->|Có| G
    R -->|Không| S[Save Sweep Results]
    
    S --> T[Generate Analysis Report]
    T --> U[Sweep Analysis Complete]
    
    style A fill:#e1f5fe
    style U fill:#c8e6c9
    style G fill:#fff3e0
    style R fill:#fff3e0
```

---

## 20. Flow Chart Logic Mã Nguồn - Attack Simulation Process

```mermaid
graph TD
    A[Attack Simulation Process] --> B[attacksim.py]
    B --> C[Initialize Memory Model]
    
    C --> D[Setup Vulnerable Bits]
    D --> E[Load Victim Binary]
    E --> F[Load Sweep Results]
    
    F --> G[For Each Attack Scenario]
    G --> H[Select Target Bit]
    H --> I[Apply Bit Flip]
    
    I --> J[Run Binary]
    J --> K[Check Output]
    
    K --> L{Attack Success?}
    L -->|Có| M[Record Success]
    L -->|Không| N[Check Detection]
    
    N --> O{DIG Detected?}
    O -->|Có| P[Record DIG Detection]
    O -->|Không| Q{CIG Detected?}
    
    Q -->|Có| R[Record CIG Detection]
    Q -->|Không| S[Record No Effect]
    
    M --> T{More Scenarios?}
    P --> T
    R --> T
    S --> T
    
    T -->|Có| G
    T -->|Không| U[Generate Attack Report]
    
    U --> V[Attack Simulation Complete]
    
    style A fill:#e1f5fe
    style V fill:#c8e6c9
    style G fill:#fff3e0
    style L fill:#fff3e0
    style O fill:#fff3e0
    style Q fill:#fff3e0
    style T fill:#fff3e0
```

---

## Tóm Tắt Các Thành Phần Chính

### 1. **Core Components**
- **Training Module**: Huấn luyện các mô hình DNN
- **Build Module**: Chuyển đổi mô hình thành binary files
- **Sweep Module**: Phân tích bit-flip vulnerabilities
- **Attack Module**: Mô phỏng tấn công
- **Analysis Module**: Phân tích binary với Ghidra

### 2. **Protection Mechanisms**
- **DIG (Detection of Integrity Guard)**: Phát hiện thay đổi integrity
- **CIG (Coverage Integrity Guard)**: Theo dõi coverage để phát hiện anomalies

### 3. **Supported Models**
- ResNet50, DenseNet121, GoogLeNet, LeNet1
- CIFAR10, MNIST, FashionC, ImageNet datasets

### 4. **Compilers**
- TVM, Glow, NNFusion

### 5. **Workflow Modes**
- **Simple Mode**: Chỉ training và testing
- **Full Mode**: Toàn bộ pipeline từ training đến attack simulation

### 6. **Key Classes và Functions**
- **BinaryInfo**: Quản lý thông tin binary files
- **MergedDataset**: Xử lý datasets phức tạp
- **CodeIntegrityGuard**: Triển khai CIG protection
- **CovModeConfig**: Cấu hình coverage modes
- **LoadedModInfo**: Quản lý loaded modules
- **FlipResult**: Lưu trữ kết quả bit-flip

### 7. **Core Processes**
- **Model Building**: Chuyển đổi PyTorch models thành binaries
- **Instrumentation**: Thêm protection mechanisms
- **Bit-Flip Sweep**: Phân tích vulnerabilities
- **Attack Simulation**: Mô phỏng tấn công thực tế
- **Memory Management**: Quản lý memory permissions và bit manipulation

Các flow chart này cung cấp cái nhìn tổng quan chi tiết về cách dự án BitShield hoạt động, từ cài đặt ban đầu đến việc thực hiện các thí nghiệm bảo mật phức tạp, bao gồm cả logic mã nguồn chi tiết của từng thành phần.
