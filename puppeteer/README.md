# Puppeteer Federated Learning Simulation

Automated testing and simulation framework for federated learning experiments using Puppeteer to launch multiple browser clients.

## 🎯 Purpose

The Puppeteer simulation allows you to:
- **Automate multi-client testing** without manual browser management
- **Consistent experiment conditions** across different runs
- **Scale testing** to dozens or hundreds of clients
- **Avoid browser throttling** issues that occur with background tabs
- **Collect centralized logs** from all simulated clients

## 🏗️ Architecture

### Core Components

- **fl_simulation.js**: Main simulation script
- **Browser Management**: Headless browser instance management
- **Client Orchestration**: Multiple page instances for FL clients
- **Logging System**: Centralized console output collection

### Key Features

- **Headless Mode**: No visible browser windows (configurable)
- **Configurable Client Count**: Easy adjustment of participant numbers
- **Centralized Logging**: All client logs in one terminal
- **Automatic Cleanup**: Browser cleanup after simulation
- **Error Handling**: Graceful handling of client failures

## 🚀 Quick Start

### Prerequisites

- Node.js (v16+)
- Federated learning server running on `http://localhost:3001`
- Federated learning client running on `http://localhost:3000`

### Installation

```bash
cd puppeteer
npm install
```

### Basic Usage

```bash
# Run simulation with default settings (5 clients, 2 minutes)
node fl_simulation.js
```

## ⚙️ Configuration

### Script Configuration

Edit the top of `fl_simulation.js`:

```javascript
// CONFIGURATION
const NUM_CLIENTS = 5;                    // Number of simulated clients
const CLIENT_URL = 'http://localhost:3000'; // URL of your React FL client
const RUN_DURATION_MS = 120000;           // How long to keep clients open (in ms)
const HEADLESS_MODE = true;               // Set to false to see browser windows
```

## 🔄 Simulation Flow

### 1. Browser Initialization
```javascript
const browser = await puppeteer.launch({ 
  headless: HEADLESS_MODE,
  args: ['--no-sandbox', '--disable-setuid-sandbox']
});
```

### 2. Client Page Creation
```javascript
for (let i = 0; i < NUM_CLIENTS; i++) {
  const page = await browser.newPage();
  await page.goto(CLIENT_URL);
  pages.push(page);
}
```

### 3. Log Collection
```javascript
// Capture console logs from each client
page.on('console', (msg) => {
  console.log(`Client ${i + 1}: ${msg.text()}`);
});
```

### 4. Cleanup
```javascript
// Close all pages and browser
await browser.close();
```

## 📈 Monitoring and Logs

### Console Output

Each client's logs are prefixed with `Client X:`:

```
Client 1: WebGPU backend initialized successfully
Client 2: Client registered and model initialized successfully
Client 3: Starting local training with non-IID data distribution...
Client 4: Local training completed in 2345.67ms
Client 5: Submitting weights to server...
```

### Log Analysis

```bash
# Filter logs by client
node fl_simulation.js | grep "Client 1:"

# Filter by specific events
node fl_simulation.js | grep "training completed"

# Count successful registrations
node fl_simulation.js | grep "registered successfully" | wc -l
```

### Network Throttling

```javascript
// Simulate slow network conditions
await page.emulateNetworkConditions({
  offline: false,
  downloadThroughput: (1024 * 1024) / 8, // 1 Mbps
  uploadThroughput: (1024 * 1024) / 8,   // 1 Mbps
  latency: 100 // 100ms latency
});
```

### Custom Client Behavior

```javascript
// Add custom client behavior
await page.evaluate(() => {
  // Inject custom JavaScript
  window.customClientBehavior = true;
});

// Wait for specific conditions
await page.waitForFunction(() => {
  return document.querySelector('.training-status') !== null;
});
```

## 🐛 Troubleshooting

### Common Issues

1. **Browser Launch Fails**
   ```bash
   # Install additional dependencies (Ubuntu/Debian)
   sudo apt-get install -y \
     gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 \
     libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 \
     libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 \
     libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 \
     libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 \
     libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation \
     libappindicator1 libnss3 lsb-release xdg-utils wget
   ```

2. **Client Registration Fails**
   ```bash
   # Ensure server is running and has capacity
   # Check server logs for registration errors
   # Verify CLIENT_URL is correct
   ```

3. **Memory Issues**
   ```bash
   # Reduce client count
   
   # Increase Node.js memory limit
   node --max-old-space-size=4096 fl_simulation.js
   ```

4. **Timeout Issues**
   ```bash
   # Increase duration for longer experiments
   
   # Check server timeout settings
   ```

### Performance Optimization

- **Headless Mode**: Faster execution without UI rendering
- **Client Count**: Balance between testing coverage and resource usage
- **Memory Management**: Monitor system memory during large simulations
- **Network Conditions**: Consider network throttling for realistic testing

## 🔄 Integration with CI/CD

### GitHub Actions Example

```yaml
name: Federated Learning Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '16'
      
      - name: Install dependencies
        run: |
          cd client && npm install
          cd ../server && npm install
          cd ../puppeteer && npm install
      
      - name: Start server
        run: |
          cd server
          npm start -- --min-clients=2 --fl-rounds=3 &
          sleep 10
      
      - name: Run simulation
        run: |
          cd puppeteer
          node fl_simulation.js --clients=2 --duration=120
```

## 🚀 Advanced Usage

### Custom Client Scripts

```javascript
// Create custom client behavior
const customClientScript = `
  // Custom client initialization
  window.customInit = async () => {
    console.log('Custom client initialization');
    // Add custom logic here
  };
  
  // Execute custom initialization
  window.customInit();
`;

await page.evaluate(customClientScript);
```

### Distributed Testing

```javascript
// Test across multiple machines
const MACHINES = [
  'http://machine1:3000',
  'http://machine2:3000',
  'http://machine3:3000'
];

for (const machine of MACHINES) {
  const page = await browser.newPage();
  await page.goto(machine);
  // ... rest of client setup
}
```

---

**Ready to automate your federated learning experiments! 🚀** 