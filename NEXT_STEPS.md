# Next Steps After GitHub Release

## ✅ What You've Done
- Uploaded model.onnx.data and class_labels.json to GitHub Releases

## 📋 Next Steps

### Step 1: Commit and Push Code to GitHub

```bash
# Add all the updated files
git add Dockerfile .gitignore scripts/select_best_model.py DEPLOY_WITH_RELEASES.md .github/ requirements.txt .dockerignore

# Commit
git commit -m "Lab 3: Complete setup with GitHub Releases

- Modified Dockerfile to download model from GitHub Releases
- Added GitHub Actions for Docker Hub CI/CD
- ResNet50 model (90.90% validation accuracy)
- All tests passing (21/21)"

# Push to GitHub
git push origin master
```

### Step 2: Set Up GitHub Secrets (If Not Done)

Go to: https://github.com/ainhoupna/MLOPs-Lab3/settings/secrets/actions

Add these secrets:
- `DOCKER_USERNAME` = Your Docker Hub username
- `DOCKER_PASSWORD` = Your Docker Hub access token

### Step 3: Watch GitHub Actions Build

1. Go to: https://github.com/ainhoupna/MLOPs-Lab3/actions
2. You should see a workflow running
3. Wait ~10 minutes for Docker image to build
4. Check Docker Hub: https://hub.docker.com/r/YOUR_USERNAME/mlops-lab3

### Step 4: Deploy to Render

1. Go to: https://render.com/dashboard
2. Click "New +" → "Web Service"
3. Connect your GitHub repo: `ainhoupna/MLOPs-Lab3`
4. Settings:
   - **Name**: mlops-lab3
   - **Branch**: master
   - **Runtime**: Docker
   - **Instance Type**: Free
5. Click "Create Web Service"
6. Wait 10-15 minutes for first deployment

### Step 5: Test Render Deployment

Once deployed, test it:
```bash
# Replace with your actual Render URL
RENDER_URL="https://mlops-lab3-xxxx.onrender.com"

# Test health check
curl $RENDER_URL/

# Test prediction
curl -X POST -F "file=@data/oxford-iiit-pet/images/leonberger_135.jpg" $RENDER_URL/predict
```

### Step 6: Create HuggingFace Branch

```bash
# Create and switch to hf-space branch
git checkout -b hf-space

# Edit app.py - update line 11 with your Render URL:
# API_URL = "https://mlops-lab3-xxxx.onrender.com"

# Commit and push
git add app.py
git commit -m "Update API URL for Render deployment"
git push origin hf-space
```

### Step 7: Deploy to HuggingFace

```bash
# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/mlops-lab3

# Push hf-space branch to HuggingFace
git push hf hf-space:main
```

## 🎯 Quick Commands

```bash
# 1. Push to GitHub
git add -A
git commit -m "Lab 3: Complete deployment setup"
git push origin master

# 2. After Render gives you URL, create HF branch
git checkout -b hf-space
# Edit app.py with Render URL
git add app.py
git commit -m "Update API URL"
git push origin hf-space

# 3. Deploy to HuggingFace
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/mlops-lab3
git push hf hf-space:main
```

## ✅ Success Checklist

- [ ] Code pushed to GitHub
- [ ] GitHub Actions completed successfully
- [ ] Docker image on Docker Hub
- [ ] Render service created and deployed
- [ ] Render API responds to requests
- [ ] HuggingFace branch created
- [ ] HuggingFace Space deployed
- [ ] HuggingFace Space makes real predictions

## 🔗 Your Links (Update After Deployment)

- **GitHub**: https://github.com/ainhoupna/MLOPs-Lab3
- **GitHub Actions**: https://github.com/ainhoupna/MLOPs-Lab3/actions
- **Docker Hub**: https://hub.docker.com/r/YOUR_USERNAME/mlops-lab3
- **Render**: https://mlops-lab3-xxxx.onrender.com
- **HuggingFace**: https://huggingface.co/spaces/YOUR_USERNAME/mlops-lab3

## ⚠️ Important Notes

- First Render deployment takes 10-15 minutes (PyTorch installation)
- Make sure GitHub secrets are set before pushing
- Update app.py with actual Render URL before HuggingFace deployment
- The Dockerfile will download model files from your GitHub Release automatically

Ready to push to GitHub? Run the commands in Step 1!
