# Contoh Penggunaan Zero-DCE++ dalam Evaluasi Baseline

## 1. Tanpa Zero-DCE++ (hanya baseline tradisional)

```python
evaluate_baseline("dataset/our485", max_images=5)
```

## 2. Dengan Zero-DCE++ (perlu model weights)

```python
evaluate_baseline("dataset/our485", max_images=5, zerodcepp_weight="models/zerodcepp_best.pth")
```

## Struktur yang diperlukan untuk Zero-DCE++:

### A. Model Definition (buat file `models/zerodcepp_model.py`):

```python
import torch
import torch.nn as nn

class ZeroDCEPP(nn.Module):
    def __init__(self):
        super(ZeroDCEPP, self).__init__()
        # Implementasi model Zero-DCE++
        # Ganti dengan arsitektur yang sesuai
        pass

    def forward(self, x):
        # Forward pass
        # Return enhanced_image, additional_outputs
        return x, None  # Placeholder
```

### B. Update fungsi load_zerodcepp_model di evaluate_baseline.py:

```python
def load_zerodcepp_model(model_path, device):
    """Load Zero-DCE++ model from checkpoint"""
    try:
        from models.zerodcepp_model import ZeroDCEPP
        model = ZeroDCEPP()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])  # Sesuaikan dengan format checkpoint
        model.eval()
        return model.to(device)
    except Exception as e:
        print(f"Failed to load Zero-DCE++ model: {e}")
        return None
```

### C. File struktur yang diperlukan:

```
models/
├── zerodcepp_model.py       # Definisi model
├── zerodcepp_best.pth       # Trained weights
└── __init__.py
```

## Saat ini:

Script sudah siap untuk menerima Zero-DCE++ model, tapi:

1. Fungsi `load_zerodcepp_model` return None (placeholder)
2. Perlu implementasi model Zero-DCE++ yang sebenarnya
3. Perlu trained weights

## Testing:

Untuk saat ini bisa test dengan baseline methods saja:

```bash
python scripts/evaluate_baseline.py
```
