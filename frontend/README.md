# Parking Intelligence Frontend

Modern web interface for the Parking Intelligence API.

## Features

- **Priority Prediction**: Predict enforcement priority for parking violations
- **Report Validation**: Validate parking violation reports
- **Real-time API Status**: Monitor API health and model loading status
- **Responsive Design**: Works on desktop and mobile devices

## Deployment on Vercel

### Option 1: Using Vercel CLI

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Navigate to frontend directory:
```bash
cd frontend
```

3. Deploy:
```bash
vercel
```

### Option 2: Using Vercel Dashboard

1. Push this frontend directory to a GitHub repository
2. Go to [vercel.com](https://vercel.com)
3. Click "New Project"
4. Import your GitHub repository
5. Vercel will auto-detect it as a static site
6. Click "Deploy"

## Configuration

The API URL is configured in `script.js`. To change it:

```javascript
const API_URL = 'https://your-api-url.com';
```

## Local Development

1. Serve the files using any static file server:
```bash
cd frontend
python -m http.server 8000
# or
npx serve
```

2. Open http://localhost:8000 in your browser

## API Endpoints

- `GET /` - Health check
- `GET /health` - Model status
- `POST /predict_priority` - Predict enforcement priority
- `POST /validate_report` - Validate parking reports

## Tech Stack

- HTML5
- CSS3 (with modern features)
- Vanilla JavaScript
- No build process required
