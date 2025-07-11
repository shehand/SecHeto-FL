# Federated Learning Server

A Node.js backend server that orchestrates federated learning experiments with asynchronous client selection, deadline-based aggregation, and comprehensive monitoring.

## 🏗️ Architecture

### Core Components

- **Server Orchestrator**: Main FL round management
- **Model Quality Calculator**: Client selection based on model performance
- **Aggregation Engine**: Federated averaging with client selection
- **Timing Metrics**: Performance tracking and deadline management
- **Admin Interface**: Real-time monitoring dashboard

### Key Features

- **Asynchronous FL**: Deadline-based aggregation with straggler handling
- **Client Selection**: Quality-based participant selection
- **Hybrid Mode**: Synchronous → Asynchronous transition
- **Comprehensive Logging**: JSON-based experiment tracking
- **Command-line Configuration**: Flexible parameter tuning

## 🚀 Quick Start

### Prerequisites

- Node.js (v18+)
- npm or yarn
- dataset files (optional)

### Installation

```bash
cd server
npm install
npm start
```

### Basic Usage

```bash
# Default configuration (5 clients, 10 rounds)
npm start

# Custom configuration
npm start -- --min-clients=3 --fl-rounds=15 --dataset=cifar10

# Show help
npm start -- --help
```

## ⚙️ Configuration Options

### Command-line Arguments

| Argument | Default | Description | Example |
|----------|---------|-------------|---------|
| `--min-clients` | 5 | Minimum clients required | `--min-clients=3` |
| `--fl-rounds` | 10 | Total federated learning rounds | `--fl-rounds=20` |
| `--dataset` | fashion-mnist | Dataset name | `--dataset=cifar10` |
| `--sync-rounds` | 1 | Synchronous rounds before async | `--sync-rounds=5` |
| `--aggregation-interval` | 1000 | Aggregation interval (ms) | `--aggregation-interval=500` |

### Configuration Examples

```bash
# Quick test
npm start -- --min-clients=2 --fl-rounds=5

# Production experiment
npm start -- --min-clients=10 --fl-rounds=50 --sync-rounds=5

# CIFAR-10 experiment
npm start -- --dataset=cifar10 --min-clients=5 --fl-rounds=30

# Fast aggregation
npm start -- --aggregation-interval=500 --min-clients=3
```

## 📈 Logging and Monitoring

### Console Output

Real-time logging includes:
- Client registration events
- Round start/completion
- Aggregation progress
- Timing metrics
- Error messages

### JSON Logs

Structured logs in `logs/<timestamp>.json`:

```json
{
  "1": {
    "mode": "synchronous",
    "selectedClients": ["client_1", "client_2"],
    "submittedClients": ["client_1", "client_2", "client_3"],
    "meanTime": 1234.56,
    "globalModel": {
      "loss": 0.123,
      "accuracy": 0.95
    },
    "clientMetrics": { ... }
  }
}
```

### Admin Interface

Access at `http://localhost:3001`:
- Real-time client status
- Round progress visualization
- Configuration management
- Client disconnection tools

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port
   PORT=3002 npm start
   ```

2. **Dataset Loading Errors**
   ```bash
   # Ensure dataset files are in server/data/
   # System falls back to random data generation
   ```

3. **Memory Issues**
   ```bash
   # Monitor Node.js memory usage
   # Consider reducing client count for large experiments
   ```

4. **Client Connection Issues**
   ```bash
   # Check CORS settings
   # Verify client URLs match server configuration
   ```

### Performance Optimization

- **Client Count**: Adjust based on server capacity
- **Round Count**: Balance between convergence and time
- **Deadline Settings**: Tune based on network conditions
- **Memory Management**: Monitor tensor memory usage

**Ready to orchestrate federated learning experiments! 🚀** 