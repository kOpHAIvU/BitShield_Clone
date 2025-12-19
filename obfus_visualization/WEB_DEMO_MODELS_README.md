# 🚀 Web Demo Models - Hướng dẫn sử dụng

## 📋 Tổng quan

Script này tạo **3 models** cho web demo:
1. **Original Model** (`original.pt`) - Model gốc, có thể detect
2. **Attacked Model** (`attacked.pt`) - Model bị tấn công, **KHÔNG** detect được
3. **Protected Model** (`protected.pt`) - Model được bảo vệ với OBFUS, **CÓ THỂ** detect được

---

## 🎯 Mục đích

Phục vụ cho **web demo** để:
- So sánh performance giữa 3 models
- Demo khả năng bảo vệ của OBFUS
- Test detection capability

---

## 📦 Tạo Models

### Command:

```bash
cd obfus_visualization

python prepare_web_demo_models.py ResNetSEBlockIoT IoTID20 \
  --attack-mode pbs \
  --attack-iters 25 \
  --device cuda \
  --output-dir models/web_demo/IoTID20_ResNetSEBlockIoT
```

### Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Tên model (ResNetSEBlockIoT, ...) | Required |
| `dataset_name` | Tên dataset (IoTID20, CICIoT2023, ...) | Required |
| `--attack-mode` | Loại attack: `pbs` hoặc `random` | `pbs` |
| `--attack-iters` | Số lần attack | `25` |
| `--device` | Device: `cuda` hoặc `cpu` | `cuda` |
| `--output-dir` | Thư mục output | `models/web_demo/{dataset}_{model}` |

### Output:

```
models/web_demo/IoTID20_ResNetSEBlockIoT/
├── original.pt          # Model gốc
├── attacked.pt          # Model bị tấn công
├── protected.pt         # Model được bảo vệ
└── obfus_config.json   # Config cho OBFUS runtime
```

---

## 📂 Cấu trúc Files

### 1. `original.pt`
- Model gốc sau khi train
- **Accuracy cao** (e.g., 90%+)
- Có thể detect attacks

### 2. `attacked.pt`
- Model sau khi bị bit-flip attack
- **Accuracy thấp** (e.g., <50%)
- **KHÔNG thể detect** attacks

### 3. `protected.pt`
- Model gốc + OBFUS defense
- **Accuracy cao** (gần bằng original)
- **CÓ THỂ detect** attacks

### 4. `obfus_config.json`
```json
{
  "sig_period": 500,
  "sig_k": 3.0,
  "grad_norm_type": "l1",
  "normalize_grad": true,
  "fp_threshold": 0.1,
  "fp_entropy_threshold": 0.15,
  "obfus_targets": ["linear", "conv1d"],
  "max_obfus_layers": 3
}
```

---

## 🔧 Load Models trong Web App

### Python (Flask/FastAPI):

```python
from obfus_visualization.load_web_demo_models import load_web_demo_model, predict_with_model
import torch

# Load models
original_model, _ = load_web_demo_model(
    model_name='ResNetSEBlockIoT',
    dataset_name='IoTID20',
    model_type='original',
    device='cpu'
)

attacked_model, _ = load_web_demo_model(
    model_name='ResNetSEBlockIoT',
    dataset_name='IoTID20',
    model_type='attacked',
    device='cpu'
)

protected_model, obfus_runtime = load_web_demo_model(
    model_name='ResNetSEBlockIoT',
    dataset_name='IoTID20',
    model_type='protected',
    device='cpu'
)

# Make predictions
def predict(model, x, obfus_runtime=None):
    model.eval()
    with torch.no_grad():
        if obfus_runtime:
            outputs = obfus_runtime.model(x)
        else:
            outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

# Example: Predict with original model
x = torch.randn(1, 39, 1)  # Example input
pred_original = predict(original_model, x)
pred_attacked = predict(attacked_model, x)
pred_protected = predict(protected_model, x, obfus_runtime)
```

### JavaScript (Node.js với PyTorch.js):

```javascript
// Note: You'll need to convert .pt to .pth or use ONNX format
// Or use a Python backend API

// Example API call
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_type: 'original',  // or 'attacked', 'protected'
    input: inputData
  })
});
```

---

## 🧪 Test Models

### Verify Requirements:

```python
# Load models
original_model, _ = load_web_demo_model('ResNetSEBlockIoT', 'IoTID20', 'original')
attacked_model, _ = load_web_demo_model('ResNetSEBlockIoT', 'IoTID20', 'attacked')
protected_model, obfus_runtime = load_web_demo_model('ResNetSEBlockIoT', 'IoTID20', 'protected')

# Evaluate on test set
from support.dataman_extended import get_benign_loader_extended

test_loader = get_benign_loader_extended('IoTID20', 32, 'test', batch_size=128, 
                                        shuffle=False, num_workers=0, image_size=None)

def evaluate(model, loader, obfus_runtime=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            if obfus_runtime:
                outputs = obfus_runtime.model(x)
            else:
                outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

acc_original = evaluate(original_model, test_loader)
acc_attacked = evaluate(attacked_model, test_loader)
acc_protected = evaluate(protected_model, test_loader, obfus_runtime)

print(f"Original:  {acc_original:.4f}")
print(f"Attacked:  {acc_attacked:.4f}")
print(f"Protected: {acc_protected:.4f}")

# Verify requirements
assert acc_original > 0.5, "Original model should detect"
assert acc_attacked < acc_original * 0.5, "Attacked model should NOT detect"
assert acc_protected > acc_attacked * 1.5, "Protected model should detect"
```

---

## 📊 Expected Results

### Example với IoTID20:

| Model | Accuracy | Status |
|-------|----------|--------|
| **Original** | 91.69% | ✅ Can detect |
| **Attacked** | 52.00% | ❌ Cannot detect |
| **Protected** | 59.00% | ✅ Can detect |

### Requirements:

1. ✅ **Original model**: Accuracy > 50% (can detect)
2. ❌ **Attacked model**: Accuracy < Original * 50% (cannot detect)
3. ✅ **Protected model**: Accuracy > Attacked * 150% (can detect)

---

## 🔍 Troubleshooting

### Issue: Attack không hiệu quả

**Symptom:** Attacked model vẫn có accuracy cao (>80%)

**Solution:**
```bash
# Tăng số lần attack
python prepare_web_demo_models.py ... --attack-iters 50

# Hoặc dùng PBS thay vì random
python prepare_web_demo_models.py ... --attack-mode pbs
```

### Issue: Protected model accuracy thấp

**Symptom:** Protected model accuracy < 50%

**Solution:**
- Check OBFUS config
- Đảm bảo `initial_reseed=False`
- Kiểm tra calibration data

### Issue: Không load được protected model

**Symptom:** Error khi load protected model

**Solution:**
- Đảm bảo `obfus_config.json` tồn tại
- Check OBFUS runtime initialization
- Verify model architecture matches

---

## 🎨 Web Demo Integration

### Flask Example:

```python
from flask import Flask, request, jsonify
from obfus_visualization.load_web_demo_models import load_web_demo_model, predict_with_model
import torch

app = Flask(__name__)

# Load models at startup
models_cache = {}

@app.before_first_request
def load_models():
    models_cache['original'] = load_web_demo_model('ResNetSEBlockIoT', 'IoTID20', 'original')
    models_cache['attacked'] = load_web_demo_model('ResNetSEBlockIoT', 'IoTID20', 'attacked')
    models_cache['protected'] = load_web_demo_model('ResNetSEBlockIoT', 'IoTID20', 'protected')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.get('model_type', 'original')
    input_data = torch.tensor(data['input'])
    
    model, obfus_runtime = models_cache[model_type]
    prediction, alerts = predict_with_model(model, input_data, obfus_runtime)
    
    return jsonify({
        'prediction': prediction.tolist(),
        'alerts': alerts
    })

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📝 Notes

1. **Model Size**: Mỗi model ~10-50MB (tùy architecture)
2. **Loading Time**: ~1-3 seconds per model (CPU)
3. **Memory**: ~500MB-2GB RAM (tùy model size)
4. **OBFUS Runtime**: Chỉ cần cho protected model
5. **Calibration**: Protected model cần calibration data để OBFUS hoạt động đúng

---

## ✅ Checklist

- [x] Create `prepare_web_demo_models.py`
- [x] Create `load_web_demo_models.py`
- [x] Create README documentation
- [ ] Test với IoTID20
- [ ] Test với CICIoT2023
- [ ] Create Flask/FastAPI example
- [ ] Create JavaScript/Node.js example

---

**Date:** 2025-12-15  
**Status:** ✅ Ready for use  
**Next Steps:** Test và integrate vào web demo

