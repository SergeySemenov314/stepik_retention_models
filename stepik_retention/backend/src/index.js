const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3002;
const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || 'http://stepik-retention-model:8000';

// Load precomputed user features
let usersFeatures = {};
const featuresPath = path.join(__dirname, '..', 'data', 'users_features.json');

try {
  const data = fs.readFileSync(featuresPath, 'utf8');
  usersFeatures = JSON.parse(data);
  console.log(`Loaded features for ${Object.keys(usersFeatures).length} users`);
} catch (err) {
  console.error('Failed to load users_features.json:', err.message);
  console.error('Run precompute_features.py first.');
}

app.use(cors({
  origin: ['http://localhost:3002', 'http://localhost:3003', 'http://localhost:80', 'http://localhost'],
  credentials: true
}));
app.use(express.json());

// Get list of available user IDs for randomizer
app.get('/api/users', (req, res) => {
  const userIds = Object.keys(usersFeatures).map(Number);
  res.json({ userIds });
});

// Predict for a specific user
app.get('/api/predict/:userId', async (req, res) => {
  const userId = parseInt(req.params.userId, 10);
  if (isNaN(userId)) {
    return res.status(400).json({ error: 'Invalid user_id' });
  }

  const features = usersFeatures[userId];
  if (!features) {
    return res.status(404).json({
      error: 'User not found',
      userId,
      message: 'Пользователь не найден в датасете'
    });
  }

  try {
    const response = await axios.post(`${MODEL_SERVICE_URL}/predict`, {
      features
    }, { timeout: 5000 });

    res.json({
      userId,
      userData: features,
      prediction: response.data.prediction,
      willComplete: response.data.will_complete,
      probability: response.data.probability
    });
  } catch (err) {
    console.error('Model service error:', err.message);
    res.status(503).json({
      error: 'Model service unavailable',
      message: err.response?.data?.detail || err.message
    });
  }
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    usersLoaded: Object.keys(usersFeatures).length > 0
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Stepik Retention Backend running on port ${PORT}`);
});
