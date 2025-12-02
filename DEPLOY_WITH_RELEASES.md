# How to Deploy Lab 3 (Model Too Large for Git)

## The Solution

Since the model files are too large for GitHub, we'll:
1. Push the code WITHOUT model files
2. Upload model files to GitHub Releases
3. Dockerfile will download them during build

## Step-by-Step Instructions

### Step 1: Push Code to GitHub (Without Models)

```bash
# Make sure model files are NOT staged
git reset model.onnx model.onnx.data class_labels.json 2>/dev/null

# Add updated files
git add Dockerfile scripts/select_best_model.py .gitignore

# Commit
git commit -m "Update Dockerfile to download model from GitHub Releases"

# Push
git push origin master
```

### Step 2: Create GitHub Release with Model Files

1. **Go to your repository**: https://github.com/ainhoupna/MLOPs-Lab3

2. **Click "Releases"** (right sidebar)

3. **Click "Create a new release"**

4. **Fill in the form**:
   - **Tag**: `v1.0`
   - **Release title**: `Lab 3 - Model Files v1.0`
   - **Description**: 
     ```
     ResNet50 ONNX model for pet breed classification
     - Validation accuracy: 90.90%
     - 37 pet breed classes
     ```

5. **Upload files** (drag and drop):
   - `model.onnx` (or `model.onnx.data` if you have it)
   - `class_labels.json`

6. **Click "Publish release"**

### Step 3: Verify Release URLs

After publishing, your files will be available at:
- `https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/model.onnx`
- `https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/class_labels.json`

### Step 4: Deploy to Render

1. Go to https://render.com/dashboard
2. Your service should auto-deploy from the GitHub push
3. **OR** click "Manual Deploy" → "Deploy latest commit"
4. Wait 10-15 minutes for build

The Dockerfile will automatically download the model files from your GitHub Release!

### Step 5: Test Render Deployment

```bash
# Replace with your Render URL
curl https://your-app.onrender.com/predict \
  -F "file=@data/oxford-iiit-pet/images/leonberger_135.jpg"
```

### Step 6: Create HuggingFace Branch

```bash
# Create hf-space branch
git checkout -b hf-space

# Edit app.py - update API_URL to your Render URL
# Then commit and push
git add app.py
git commit -m "Update API URL for HuggingFace"
git push origin hf-space
```

### Step 7: Deploy to HuggingFace

```bash
# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/mlops-lab3

# Push hf-space branch to HuggingFace
git push hf hf-space:main
```

## Troubleshooting

**Q: Render build fails - "model.onnx not found"**
A: Make sure you created the GitHub Release and the URLs are correct

**Q: How do I update the model later?**
A: Create a new release (v1.1, v1.2, etc.) and update the Dockerfile URLs

**Q: Can I use a different storage?**
A: Yes! You can use:
- HuggingFace Hub
- Google Drive (with direct download link)
- Dropbox
- Any public URL

Just update the `wget` URLs in the Dockerfile.

## Alternative: Use Smaller Model

If you want to avoid GitHub Releases, you can train a smaller model:

```bash
# Train MobileNetV2 (much smaller)
uv run python scripts/train.py --model-name mobilenet_v2 --epochs 5
```

Then re-run the model selection script.

## Quick Reference

**Your model files location**:
```
/home/alumno/Desktop/datos/MLOPS/MLOPs-Lab3/model.onnx
/home/alumno/Desktop/datos/MLOPS/MLOPs-Lab3/class_labels.json
```

**GitHub Release URL format**:
```
https://github.com/USERNAME/REPO/releases/download/TAG/FILENAME
```

**Your specific URLs** (after creating v1.0 release):
```
https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/model.onnx
https://github.com/ainhoupna/MLOPs-Lab3/releases/download/v1.0/class_labels.json
```
