# Federated Learning Client

A React-based federated learning client built with TensorFlow.js, featuring WebGPU acceleration, automatic backend fallback, and real-time training progress monitoring.

## 🏗️ Architecture

### Core Components

- **FederatedClient**: Main client orchestrator for FL participation
- **NeuralNetwork**: TensorFlow.js model management with backend optimization
- **DatasetService**: Non-IID data distribution and dataset loading

### Key Features

- **WebGPU Acceleration**: Automatic fallback to WebGL/CPU
- **Non-IID Data Distribution**: Realistic federated learning scenarios
- **Real-time Progress**: Live training status and round updates
- **Automatic Registration**: Seamless server connection
- **Memory Management**: Efficient tensor disposal

## 🚀 Quick Start

### Prerequisites

- Node.js (v18+)
- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- Server running on `http://localhost:3001`

### Installation

```bash
cd client
npm install
npm start
```

The client will automatically:
1. Connect to the federated learning server
2. Register as a participant
3. Download and prepare the dataset
4. Begin training when rounds start

## 🧪 Testing

### Manual Testing

1. Open multiple browser tabs
2. Navigate to `http://localhost:3000`
3. Monitor console for client behavior
4. Check server logs for registration

### Automated Testing

Use the puppeteer simulation for consistent testing:
```bash
cd ../puppeteer
node fl_simulation.js
```

## 📈 Performance Metrics

### Timing Information

- **Training Time**: Local model training duration
- **Communication Time**: Network request duration
- **Total Round Time**: Complete round duration

### Quality Metrics

- **Model Loss**: Training loss per round
- **Model Accuracy**: Training accuracy per round
- **Data Distribution**: Non-IID distribution statistics

---

**Ready to participate in federated learning! 🚀**
