# SecHeto-FL

A comprehensive federated learning system built with TensorFlow.js, featuring asynchronous client selection, deadline-based aggregation, and WebGPU acceleration.

## 🚀 Features

- **Asynchronous Federated Learning**: Deadline-based aggregation with straggler handling
- **Client Selection**: Model quality-based client selection for optimal aggregation
- **WebGPU Acceleration**: High-performance training with automatic fallback to WebGL/CPU
- **Multiple Datasets**: Support for Fashion-MNIST, MNIST and CIFAR-10
- **Non-IID Data Distribution**: Realistic federated learning scenarios
- **Comprehensive Logging**: Detailed metrics and timing information
- **Puppeteer Simulation**: Automated multi-client testing
- **Configurable Parameters**: Command-line configuration for experiments

## 📁 Project Structure

```
WebFL-tfjs/
├── client/                 # React frontend with TensorFlow.js
│   ├── src/
│   │   ├── services/       # FL client, neural network, dataset services
│   └── README.md           # Client-specific documentation
├── server/                 # Node.js backend server
│   ├── src/
│   │   ├── modelQuality.ts # Model evaluation and client selection
│   │   └── server.ts       # Main server with FL orchestration
│   └── README.md           # Server-specific documentation
├── puppeteer/              # Automated client simulation
│   ├── fl_simulation.js    # Multi-client testing script
│   └── README.md           # Puppeteer-specific documentation
└── README.md               # This file
```

## 🛠️ Quick Start

### Prerequisites

- **Node.js** (tested in v18.20.2)
- **npm** or **yarn**
- **Modern browser** with WebGPU support (Chrome 113+, Edge 113+)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd WebFL-tfjs
   ```

2. **Install dependencies**
   ```bash
   # Install client dependencies
   cd client
   npm install
   
   # Install server dependencies
   cd ../server
   npm install
   
   # Install puppeteer dependencies
   cd ../puppeteer
   npm install
   ```

3. **Start the server**
   ```bash
   cd ../server
   npm start -- --min-clients=3 --fl-rounds=10
   ```

4. **Start the client** (in a new terminal)
   ```bash
   cd client
   npm start
   ```

5. **Run experiments** (optional)
   ```bash
   cd puppeteer
   node fl_simulation.js
   ```

## 🧪 Running Experiments

### Basic Experiment
```bash
# Start server with 3 clients, 10 rounds
cd server
npm start -- --min-clients=3 --fl-rounds=10

# Open 3 browser tabs to http://localhost:3000
# Or use puppeteer simulation
cd ../puppeteer
node fl_simulation.js
```

### Advanced Experiment
```bash
# CIFAR-10 with more clients and rounds
cd server
npm start -- --dataset=cifar10 --min-clients=5 --fl-rounds=30 --sync-rounds=5

# Use puppeteer for automated testing
cd ../puppeteer
# Edit NUM_CLIENTS in fl_simulation.js to match server config
node fl_simulation.js
```

### Performance Testing
```bash
# Test with different client counts
npm start -- --min-clients=2 --fl-rounds=5    # Quick test
npm start -- --min-clients=10 --fl-rounds=50  # Full experiment
```

## 📈 Monitoring and Logs

### Server Logs
- Real-time console output
- JSON logs in `server/logs/<timestamp>.json`
- Client timing metrics
- Model quality scores

### Client Logs
- Training progress
- Backend selection
- Communication timing
- Error handling

### Admin Interface
- Access at `http://localhost:3001`
- Real-time client status
- Round progress
- Configuration management

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-clients` | 5 | Minimum clients required |
| `--fl-rounds` | 10 | Total federated learning rounds |
| `--dataset` | fashion-mnist | Dataset (fashion-mnist, cifar10) |
| `--sync-rounds` | 1 | Synchronous rounds before async |
| `--aggregation-interval` | 1000 | Aggregation interval (ms) |

## 📚 Documentation

- **[Client README](client/README.md)** - Frontend setup and usage
- **[Server README](server/README.md)** - Backend configuration and API
- **[Puppeteer README](puppeteer/README.md)** - Automated testing guide

## 🐛 Troubleshooting

### Common Issues

1. **WebGPU not supported**
   - System automatically falls back to WebGL/CPU
   - Check console for backend selection messages

2. **Server connection issues**
   - Ensure server is running on port 3001
   - Check CORS settings if using different ports

3. **Client registration fails**
   - Verify server has capacity for more clients
   - Check network connectivity

4. **Training hangs**
   - Check browser console for errors
   - Verify dataset files are accessible

### Performance Tips

- Use WebGPU-enabled browsers for best performance
- Adjust client count based on your hardware
- Monitor memory usage with large datasets
- Use puppeteer for consistent testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This software is released under an **Evaluation-Only License**.  
Use is limited to personal, non-commercial evaluation. Academic and research use is explicitly prohibited.

For licensing inquiries or special permissions, contact the developer.

See LICENSE for full terms.

## 🙏 Acknowledgments

- TensorFlow.js
- WebGPU working group for the new standard
- Federated learning research community

---

**Happy federated learning! 🚀** 
