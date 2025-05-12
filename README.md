## 📦 Dataset Setup

This project uses the [CCPD dataset](https://github.com/detectRecog/CCPD).

### 🔧 Environment Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

### 📂 Dataset Download and Structure

3. Download the dataset from the official source (e.g., Google Drive or Baidu link).

4. Extract and organize it as follows:

```
dataset/
├── train/
├── val/
├── test/
```

Each folder should contain CCPD `.jpg` images.

### 📝 Parse Annotations

5. Run the dataset parser to generate annotation files:

```bash
python dataset/parse_ccpd.py
```

This will create:
- `dataset/train.json`
- `dataset/val.json`
- `dataset/test.json`
