# 🚀 Deployment Guide - Render

Complete guide to deploy the Smart Forecasting application on Render.

---

## Prerequisites

- GitHub account
- Render account (free tier available)
- Git installed locally

---

## Step 1: Prepare Repository

### 1.1 Push to GitHub

```bash
# Add remote (if not already configured)
git remote add origin https://github.com/efrenbohorquez/smartforecasting.git

# Push code
git push -u origin master
```

### 1.2 Verify Files

Ensure these files exist in your repository:
- ✅ `requirements.txt` - Python dependencies
- ✅ `render.yaml` - Render configuration
- ✅ `app/main.py` - Application entry point
- ✅ `README.md` - Project documentation

---

## Step 2: Deploy on Render

### Option A: Using Render Dashboard (Recommended)

1. **Go to Render Dashboard**
   - Visit [https://dashboard.render.com/](https://dashboard.render.com/)
   - Sign in with your GitHub account

2. **Create New Web Service**
   - Click **"New +"** → **"Web Service"**
   - Select **"Connect a repository"**
   - Authorize Render to access your GitHub
   - Choose `efrenbohorquez/smartforecasting`

3. **Configure Service**
   - **Name**: `smartforecasting` (or your preference)
   - **Region**: Choose closest to your users
   - **Branch**: `master` (or `main`)
   - **Root Directory**: Leave blank
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app/main.py`

4. **Environment Variables** (Optional)
   - No manual env vars needed (PORT is auto-set)
   - If needed, add custom variables in the "Environment" tab

5. **Deploy**
   - Click **"Create Web Service"**
   - Render will build and deploy automatically
   - Wait 5-10 minutes for first deployment

### Option B: Using render.yaml (Blueprint)

1. **Go to Render Dashboard**
   - Visit [https://dashboard.render.com/](https://dashboard.render.com/)

2. **Create Blueprint**
   - Click **"New +"** → **"Blueprint"**
   - Select your repository
   - Render will detect `render.yaml` automatically

3. **Apply Blueprint**
   - Review configuration
   - Click **"Apply"**
   - Deployment starts automatically

---

## Step 3: Monitor Deployment

### Build Logs

```
==> Installing dependencies
Collecting gradio
Collecting tensorflow-cpu
...
==> Build successful

==> Starting service
Cargando datos...
Usando caché local: base_datos_cache.xlsx
* Running on http://0.0.0.0:10000
```

### Expected Output

- ✅ Dependencies installed
- ✅ Data cache loaded
- ✅ TensorFlow initialized
- ✅ Gradio server started
- ✅ App accessible at `https://smartforecasting.onrender.com`

---

## Step 4: Verify Deployment

### Test Functionality

1. **Open App URL**
   - Render provides: `https://your-service-name.onrender.com`

2. **Test Individual Analysis Tab**
   - Select Product: `P9933`
   - Select Warehouse: `BDG-19GN1`
   - Click **"Generar Predicción"**
   - Verify prediction appears

3. **Test Global Analysis Tab**
   - Check statistics load
   - Verify plots render

4. **Test Technical Details Tab**
   - Images should load correctly
   - Tables display properly

---

## Troubleshooting

### Issue: Build Fails

**Symptoms**: Render shows "Build failed"

**Solutions**:
```bash
# Check requirements.txt format
cat requirements.txt

# Ensure no version conflicts
pip install -r requirements.txt  # Test locally first
```

### Issue: App Crashes on Start

**Symptoms**: "Application failed to respond"

**Solutions**:
1. Check Render logs for Python errors
2. Verify `PORT` environment variable is used:
   ```python
   PORT = int(os.environ.get("PORT", 7860))
   ```
3. Ensure `server_name="0.0.0.0"` in `launch()`

### Issue: Models Not Found

**Symptoms**: "No hay modelo entrenado para la bodega X"

**Solutions**:
1. Verify `01_MODELOS/` directory exists in repository
2. Check `.gitignore` doesn't exclude model files
3. Ensure model files are committed:
   ```bash
   git add 01_MODELOS/
   git commit -m "Add trained models"
   git push
   ```

### Issue: Slow Performance

**Symptoms**: App takes 30+ seconds to load

**Solutions**:
1. Render free tier may sleep after inactivity (first request is slow)
2. Upgrade to paid tier for always-on instances
3. Optimize model caching in `app/models.py`

---

## Performance Optimization

### Free Tier Limitations
- ⚠️ Apps sleep after 15 minutes of inactivity
- ⚠️ 512 MB RAM limit
- ⚠️ Shared CPU

### Upgrade Options
- **Starter ($7/month)**: 512 MB RAM, always-on
- **Standard ($25/month)**: 2 GB RAM, faster CPU

### Optimization Tips
1. **Use tensorflow-cpu** (lighter than full TensorFlow)
2. **Enable model caching** (already implemented)
3. **Compress model files** if needed
4. **Use data cache** (`base_datos_cache.xlsx`)

---

## Custom Domain (Optional)

1. Go to Render service settings
2. Click **"Custom Domain"**
3. Add your domain (e.g., `forecast.yourdomain.com`)
4. Update DNS records as instructed by Render
5. Wait for SSL certificate provisioning

---

## Continuous Deployment

Render automatically redeploys on Git push:

```bash
# Make changes
git add .
git commit -m "Update model or code"
git push origin master

# Render detects push and redeploys automatically
```

---

## Monitoring

### Render Dashboard
- **Metrics**: CPU, Memory, Response time
- **Logs**: Real-time application logs
- **Events**: Deploy history

### Custom Monitoring (Advanced)
- Add error tracking (e.g., Sentry)
- Implement health check endpoint
- Set up uptime monitoring (e.g., UptimeRobot)

---

## Rollback

If deployment breaks:

1. Go to Render Dashboard
2. Select your service
3. Click **"Manual Deploy"** → **"Deploy from branch"**
4. Choose previous commit SHA
5. Click **"Deploy"**

---

## Cost Estimation

| Plan | Price | Use Case |
|------|-------|----------|
| Free | $0 | Testing, demos |
| Starter | $7/mo | Small projects |
| Standard | $25/mo | Production apps |

**Recommendation**: Start with Free tier, upgrade if needed.

---

## Security

### Best Practices
- ✅ Don't commit API keys (use environment variables)
- ✅ Use HTTPS (Render provides free SSL)
- ✅ Review Render access logs regularly
- ✅ Enable 2FA on GitHub and Render accounts

---

## Support

### Render Documentation
- https://render.com/docs

### GitHub Issues
- Open issue at: https://github.com/efrenbohorquez/smartforecasting/issues

---

## Next Steps After Deployment

1. ✅ Share app URL with stakeholders
2. ✅ Monitor user feedback
3. ✅ Iterate on model improvements
4. ✅ Add analytics (optional)
5. ✅ Document lessons learned

---

**Last Updated**: November 2024  
**Maintained by**: Efren Bohorquez
