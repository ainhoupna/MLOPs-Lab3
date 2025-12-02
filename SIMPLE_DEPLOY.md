# Simple Deployment Guide (No Git LFS Needed!)

## Step 1: Prepare Repository

### Update .gitignore (already done!)
Model files are now allowed in git temporarily.

### Check what will be committed
```bash
git status
```

## Step 2: Set Up GitHub Secrets

1. Go to: https://github.com/ainhoupna/MLOPs-Lab3/settings/secrets/actions
2. Click "New repository secret"
3. Add these two secrets:

**Secret 1:**
- Name: `DOCKER_USERNAME`
- Value: Your Docker Hub username

**Secret 2:**
- Name: `DOCKER_PASSWORD`  
- Value: Your Docker Hub access token

**How to get Docker Hub token:**
- Go to https://hub.docker.com
- Click your profile → Account Settings → Security
- Click "New Access Token"
- Name it "GitHub Actions Lab3"
- Copy the token and use it as DOCKER_PASSWORD

## Step 3: Push to GitHub (Master Branch)

```bash
# Stage all changes
git add .

# Commit
git commit -m "Lab 3: Transfer learning with MLflow and ONNX

- ResNet50 model (90.90% accuracy)
- MLflow experiment tracking
- ONNX deployment
- GitHub Actions CI/CD
- All tests passing"

# Push
git push origin master
```

## Step 4: Create HuggingFace Branch

```bash
# Create and switch to hf-space branch
git checkout -b hf-space

# This branch will only have the Gradio app files
git push origin hf-space
```

## Step 5: Deploy to Render

1. Go to https://render.com/dashboard
2. Click "New +" → "Web Service"
3. Click "Connect account" if needed
4. Select your repository: `ainhoupna/MLOPs-Lab3`
5. Configure:
   - **Name**: mlops-lab3
   - **Branch**: master
   - **Runtime**: Docker
   - **Instance Type**: Free
6. Click "Create Web Service"
7. Wait 10-15 minutes for deployment

## Step 6: Get Your Render URL

After deployment completes, you'll get a URL like:
`https://mlops-lab3-xxxx.onrender.com`

Copy this URL!

## Step 7: Update HuggingFace Branch

```bash
# Make sure you're on hf-space branch
git checkout hf-space

# Edit app.py - change line 11 to your Render URL
# API_URL = "https://mlops-lab3-xxxx.onrender.com"
```

Update the file, then:
```bash
git add app.py
git commit -m "Update API URL for Render"
git push origin hf-space
```

## Step 8: Deploy to HuggingFace

1. Go to https://huggingface.co/new-space
2. Create new space:
   - **Name**: mlops-lab3
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free)
3. After creation, go to **Settings**
4. Under "Repository", find the git URL
5. Add HuggingFace as remote:

```bash
# Make sure you're on hf-space branch
git checkout hf-space

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/mlops-lab3

# Push to HuggingFace
git push hf hf-space:main
```

## Verification

### Check GitHub Actions
- Go to https://github.com/ainhoupna/MLOPs-Lab3/actions
- Should see workflow completed

### Check Docker Hub
- Go to https://hub.docker.com/r/YOUR_USERNAME/mlops-lab3
- Should see the image

### Check Render
- Status should be "Live"
- Test: `curl https://your-render-url.onrender.com/`

### Check HuggingFace
- Space should be running
- Upload a pet image
- Should get real prediction (not random!)

## Quick Commands

```bash
# Switch to master branch
git checkout master

# Switch to HuggingFace branch  
git checkout hf-space

# See current branch
git branch

# Push master
git push origin master

# Push HuggingFace
git push hf hf-space:main
```

## Troubleshooting

**Q: Git push fails with "file too large"**
A: The model.onnx.data is 90MB. GitHub allows up to 100MB. It should work.

**Q: GitHub Actions fails**
A: Check that secrets are set correctly in repository settings.

**Q: Render build fails**
A: Check logs. First build takes 10-15 minutes.

**Q: HuggingFace shows random predictions**
A: Make sure API_URL in app.py (hf-space branch) points to your Render URL.
