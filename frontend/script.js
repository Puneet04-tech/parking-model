const API_URL = 'https://parking-model.onrender.com';

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`${btn.dataset.tab}-tab`).classList.add('active');
    });
});

// Check API status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        const indicator = document.querySelector('.status-indicator');
        const statusText = document.getElementById('status-text');
        
        if (data.status === 'healthy') {
            indicator.classList.add('online');
            indicator.classList.remove('offline');
            statusText.textContent = 'API Online - Models Loaded';
        } else {
            indicator.classList.add('offline');
            indicator.classList.remove('online');
            statusText.textContent = 'API Offline';
        }
    } catch (error) {
        const indicator = document.querySelector('.status-indicator');
        const statusText = document.getElementById('status-text');
        indicator.classList.add('offline');
        indicator.classList.remove('online');
        statusText.textContent = 'API Offline - Connection Failed';
    }
}

// Priority prediction form
document.getElementById('priority-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = e.target.querySelector('button');
    btn.disabled = true;
    btn.textContent = 'Predicting...';
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric values
    Object.keys(data).forEach(key => {
        data[key] = parseFloat(data[key]);
    });
    
    try {
        const response = await fetch(`${API_URL}/predict_priority`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            const resultDiv = document.getElementById('priority-result');
            resultDiv.classList.remove('hidden');
            
            document.getElementById('priority-value').textContent = result.recommended_dispatch_priority;
            document.getElementById('priority-value').className = `value priority ${result.recommended_dispatch_priority}`;
            document.getElementById('confidence-value').textContent = (result.confidence * 100).toFixed(2) + '%';
            document.getElementById('action-value').textContent = result.enforcement_action;
        } else {
            alert('Prediction failed: ' + JSON.stringify(result));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict Priority';
    }
});

// Validation form
document.getElementById('validate-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = e.target.querySelector('button');
    btn.disabled = true;
    btn.textContent = 'Validating...';

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric values appropriately
    Object.keys(data).forEach(key => {
        if (['vehicle_type', 'police_station', 'junction_name', 'pincode'].includes(key)) {
            // Keep these as strings
        } else if (key.startsWith('violation_') || key.startsWith('loc_has_') || key === 'is_weekend') {
            data[key] = parseInt(data[key]);
        } else {
            data[key] = parseFloat(data[key]);
        }
    });

    try {
        const response = await fetch(`${API_URL}/validate_report`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            const resultDiv = document.getElementById('validate-result');
            resultDiv.classList.remove('hidden');

            document.getElementById('validation-status').textContent = result.validation_status;
            document.getElementById('validation-status').className = `value status ${result.validation_status}`;
            document.getElementById('validation-confidence').textContent = (result.confidence * 100).toFixed(2) + '%';
        } else {
            alert('Validation failed: ' + JSON.stringify(result));
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Validate Report';
    }
});

// Check status on load
checkApiStatus();
setInterval(checkApiStatus, 30000); // Check every 30 seconds
