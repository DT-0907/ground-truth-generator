# Installing Tracking Libraries

## ByteTrack (Required - Recommended)

ByteTrack is the recommended and required tracker for this project.

### ✅ Recommended: Install from GitHub
```bash
pip install git+https://github.com/ifzhang/ByteTrack.git
```

**This is the primary installation method and should work on macOS/Linux.**

### ⚠️ Alternative: Install via pip (if GitHub fails)
```bash
pip install bytetrack
```

**Note**: The `bytetrack` package may not always be up-to-date. Prefer GitHub installation.

### 🔧 Manual Installation (if both above fail)
```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
pip install -e .
cd ..
```

**After manual installation, you may need to add ByteTrack to your Python path.**

## DeepSORT (Alternative)

If you prefer DeepSORT for better occlusion handling:

```bash
pip install deep-sort-realtime
```

## BoT-SORT (Alternative)

For a stronger variant:

```bash
pip install bot-sort
```

## Note

The project will work with any of these trackers. ByteTrack is recommended because:
- Fast and lightweight
- Good accuracy for traffic scenarios
- Fewer dependencies
- Easy to integrate

We'll implement a tracker wrapper that can work with any of these libraries.

