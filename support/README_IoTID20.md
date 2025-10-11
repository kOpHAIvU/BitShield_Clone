# IoTID20 Dataset Training

This directory contains the necessary modules for training models on the IoTID20 dataset using the existing `train.py` script.

## Files

- `dataman_iotid20.py` - Data preprocessing module for IoTID20 dataset
- `models/` - Contains the model definitions:
  - `quantized_layers.py` - Quantized layers and blocks
  - `resnet_seblock_iot.py` - ResNet with SE blocks
  - `simple_cnn_iot.py` - Simple CNN architecture
  - `purecnn_iot.py` - Pure CNN baseline
  - `efficientcnn_iot.py` - Efficient CNN with depthwise separable convolutions
  - `train.py` - Updated training script with IoTID20 support
- `dataset/` - Contains the IoTID20 dataset files

## Usage

### Basic Training

```bash
python support/models/train.py <model_name> IoTID20 --epochs 10 --batch-size 64
```

### Available Models

- `ResNetSEBlockIoT` - ResNet with Squeeze-and-Excitation blocks and quantization
- `SimpleCNNIoT` - Simple CNN with adaptive pooling
- `PureCNN` - Pure CNN without quantization (baseline)
- `EfficientCNN` - Lightweight CNN with depthwise separable convolutions
- `CustomModel` - Alias for ResNetSEBlockIoT (backward compatibility)
- `CustomModel2` - Alias for SimpleCNNIoT (backward compatibility)

### Examples

```bash
# Train ResNetSEBlockIoT for 10 epochs
python support/models/train.py ResNetSEBlockIoT IoTID20 --epochs 10 --batch-size 64

# Train PureCNN for 5 epochs with smaller batch size
python support/models/train.py PureCNN IoTID20 --epochs 5 --batch-size 32

# Train with GPU (if available)
python support/models/train.py EfficientCNN IoTID20 --epochs 10 --batch-size 64 --device cuda
```

### Data Preprocessing

The data preprocessing is handled automatically. The script will:

1. Check for existing `train.csv` and `test.csv` files in `support/dataset/`
2. If not found, process the source CSV file (`IoT_Network_Intrusion_Dataset.csv`)
3. Split the data into train/test sets (80%/20%)
4. Apply feature scaling and label encoding
5. Create PyTorch DataLoaders

### Output

Trained models are saved to `cfg.models_dir/IoTID20/<model_name>/<model_name>.pt`

## Dataset Information

- **Features**: 69 network traffic features
- **Classes**: 5 intrusion detection categories
- **Preprocessing**: StandardScaler normalization, LabelEncoder for labels
- **Split**: 80% train, 20% test

## Model Information

| Model | Parameters | Description |
|-------|------------|-------------|
| ResNetSEBlockIoT | ~2M | ResNet with SE blocks and quantization |
| SimpleCNNIoT | ~77K | Simple CNN with adaptive pooling |
| PureCNN | ~697K | Pure CNN baseline without quantization |
| EfficientCNN | ~49K | Lightweight CNN with depthwise separable convolutions |
