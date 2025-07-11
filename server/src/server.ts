/*!
 * Copyright (c) Shehan Edirimannage, 2025
 *
 * This file is part of the SecHeto-FL project.
 * 
 * Licensed for evaluation and personal testing purposes only.
 * Redistribution, modification, or commercial use of this file,
 * in whole or in part, is strictly prohibited without explicit 
 * written permission from the copyright holder.
 *
 * For license inquiries, contact developers.
 */

import express, { Request, Response } from 'express';
import cors from 'cors';
import { ClientInfo, FederatedRound, ModelWeights, TrainingConfig, LayerWeights } from './types';
import path from 'path';
import { ModelQualityCalculator } from './modelQuality';
import fs from 'fs';

// Parse command line arguments
function parseArgs(): ServerConfig {
  const args = process.argv.slice(2);
  const config: ServerConfig = {
    totalRounds: 10,
    requiredClients: 5,
    aggregationInterval: 1000,
    dataset: 'fashion-mnist',
    synchronousRounds: 1
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    
    if (arg.startsWith('--')) {
      const [key, value] = arg.slice(2).split('=');
      
      switch (key) {
        case 'min-clients':
        case 'required-clients':
          config.requiredClients = parseInt(value) || 5;
          break;
        case 'fl-rounds':
        case 'total-rounds':
          config.totalRounds = parseInt(value) || 10;
          break;
        case 'dataset':
          config.dataset = value || 'fashion-mnist';
          break;
        case 'sync-rounds':
        case 'synchronous-rounds':
          config.synchronousRounds = parseInt(value) || 3;
          break;
        case 'aggregation-interval':
          config.aggregationInterval = parseInt(value) || 1000;
          break;
        case 'help':
          console.log(`
Federated Learning Server Configuration Options:
  --min-clients=<number>        Minimum number of clients required (default: 5)
  --fl-rounds=<number>          Total number of federated learning rounds (default: 10)
  --dataset=<name>              Dataset to use: fashion-mnist, cifar10, mnist (default: fashion-mnist)
  --sync-rounds=<number>        Number of synchronous rounds before async (default: 3)
  --aggregation-interval=<ms>   Aggregation interval in milliseconds (default: 1000)
  --help                        Show this help message

Examples:
  npm start -- --min-clients=3 --fl-rounds=15
  npm start -- --dataset=cifar10 --sync-rounds=5
          `);
          process.exit(0);
          break;
      }
    }
  }

  return config;
}

const app = express();

// CORS and JSON middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve admin page at root
app.get('/', (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const POLLING_INTERVAL = 2000; // 2 seconds

interface ServerConfig {
  totalRounds: number;
  requiredClients: number;
  aggregationInterval: number;
  dataset: string;
  synchronousRounds: number;  // Number of rounds to run synchronously before switching to async
}

// Parse command line arguments for server configuration
let serverConfig: ServerConfig = parseArgs();

// Log the configuration
console.log('Server Configuration:', serverConfig);

// Initialize model quality calculator
const modelQualityCalculator = new ModelQualityCalculator(serverConfig.dataset);

// Update REQUIRED_CLIENTS and TOTAL_ROUNDS to use config
let REQUIRED_CLIENTS = serverConfig.requiredClients;
let TOTAL_ROUNDS = serverConfig.totalRounds;

const clients: Map<string, ClientInfo> = new Map();
const rounds: FederatedRound[] = [];
let currentRound: FederatedRound | null = null;
let globalModel: { [layer: string]: LayerWeights[] } | null = null;
let currentRoundNumber = 0;
let isAggregating = false;
let roundDeadline: number | null = null;
let roundStartTime: number | null = null;

// Add timing metrics to the types
interface ClientTimingMetrics {
  trainingTime: number;
  communicationTime: number;
  timestamp: Date;
}

// Add timing metrics map
const clientTimingMetrics: Map<string, ClientTimingMetrics[]> = new Map();

// In-memory storage for mean time per round
const roundMeanTimes: { [round: number]: number } = {};
let overallMeanTime: number = 0;

// Generate a timestamped log filename at startup
function getTimestampString() {
  const now = new Date();
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

const LOG_FILE = path.join(__dirname, `../logs/fl_rounds_${getTimestampString()}.json`);

function calculateAndStoreRoundMeanTime(roundNumber: number) {
  const times: number[] = [];
  clientTimingMetrics.forEach((timings) => {
    if (timings.length >= roundNumber) {
      const timing = timings[roundNumber - 1]; // 0-based index
      times.push(timing.trainingTime + timing.communicationTime);
    }
  });
  if (times.length > 0) {
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    roundMeanTimes[roundNumber] = mean;
    console.log(`Mean time for round ${roundNumber}: ${mean}`);
    
    // Calculate overall mean time across all rounds
    calculateOverallMeanTime();
  }
}

function calculateOverallMeanTime() {
  const roundTimes = Object.values(roundMeanTimes);
  if (roundTimes.length > 0) {
    overallMeanTime = roundTimes.reduce((a, b) => a + b, 0) / roundTimes.length;
    console.log(`Overall mean time across all rounds: ${overallMeanTime}`);
  }
}

function logRoundData(roundNumber: number, roundData: any) {
  let logs: { [key: string]: any } = {};
  if (fs.existsSync(LOG_FILE)) {
    logs = JSON.parse(fs.readFileSync(LOG_FILE, 'utf-8'));
  }
  logs[roundNumber] = roundData;
  fs.writeFileSync(LOG_FILE, JSON.stringify(logs, null, 2));
}

const getModelArchitecture = (dataset: string) => {
  switch (dataset.toLowerCase()) {
    case 'cifar10':
      return {
        inputShape: [32, 32, 3],
        layers: [
          { type: 'conv2d', filters: 32, kernelSize: 3, activation: 'relu', inputShape: [32, 32, 3] },
          { type: 'maxPooling2d', poolSize: 2 },
          { type: 'conv2d', filters: 64, kernelSize: 3, activation: 'relu' },
          { type: 'maxPooling2d', poolSize: 2 },
          { type: 'flatten' },
          { type: 'dense', units: 64, activation: 'relu' },
          { type: 'dense', units: 10, activation: 'softmax' }
        ]
      };
    case 'fashion-mnist':
      return {
        inputShape: [784],
        layers: [
          { type: 'dense', units: 128, activation: 'relu', inputShape: [784] },
          { type: 'dense', units: 64, activation: 'relu' },
          { type: 'dense', units: 10, activation: 'softmax' }
        ]
      };
    case 'mnist':
      return {
        inputShape: [784],
        layers: [
          { type: 'dense', units: 128, activation: 'relu', inputShape: [784] },
          { type: 'dense', units: 64, activation: 'relu' },
          { type: 'dense', units: 10, activation: 'softmax' }
        ]
      };
    default:
      return {
        inputShape: [10],
        layers: [
          { type: 'dense', units: 64, activation: 'relu', inputShape: [10] },
          { type: 'dense', units: 32, activation: 'relu' },
          { type: 'dense', units: 10, activation: 'softmax' }
        ]
      };
  }
};

const defaultTrainingConfig: TrainingConfig = {
  batchSize: 32,
  learningRate: 0.001,
  epochs: 1,
  modelArchitecture: getModelArchitecture(serverConfig.dataset)
};

// Helper function to check if we should use synchronous mode
function shouldUseSynchronousMode(): boolean {
  return currentRoundNumber <= serverConfig.synchronousRounds;
}

// Helper function to check if we should start aggregation (deadline passed or all clients submitted)
function shouldStartAggregation(): boolean {
  if (shouldUseSynchronousMode()) {
    // Synchronous mode: wait for all clients
    return haveAllClientsSubmitted();
  } else {
    // Asynchronous mode: wait for deadline or all clients
    return haveAllClientsSubmitted() || hasDeadlinePassed();
  }
}

// Modify the startNewRound function to handle synchronous/asynchronous transition
async function startNewRound() {
  if (currentRound || clients.size < REQUIRED_CLIENTS || currentRoundNumber >= TOTAL_ROUNDS || isAggregating) {
    return;
  }

  try {
    // Initialize model quality calculator with the current architecture
    await modelQualityCalculator.initializeModel(getModelArchitecture(serverConfig.dataset));
    modelQualityCalculator.clearMetrics();

    currentRoundNumber++;
    console.log(`Starting federated round ${currentRoundNumber}/${TOTAL_ROUNDS}`);

    currentRound = {
      roundId: currentRoundNumber,
      participants: [],
      modelWeights: {
        clients: {}
      },
      status: 'in_progress'
    };

    // Check if we should use synchronous or asynchronous mode
    if (shouldUseSynchronousMode()) {
      // Synchronous mode: no deadline, wait for all clients
      console.log(`Round ${currentRoundNumber}: SYNCHRONOUS mode (waiting for all clients)`);
      roundStartTime = Date.now();
      roundDeadline = null;
    } else {
      // Asynchronous mode: set deadline based on overallMeanTime
      roundStartTime = Date.now();
      const deadlineMs = Math.max(10000, Math.min(60000, overallMeanTime || 30000));
      roundDeadline = roundStartTime + deadlineMs;
      
      console.log(`Round ${currentRoundNumber}: ASYNCHRONOUS mode`);
      console.log(`Round started at ${new Date(roundStartTime).toISOString()}`);
      console.log(`Round deadline set to ${new Date(roundDeadline).toISOString()} (${deadlineMs}ms)`);
      
      // Set timeout to trigger aggregation when deadline is reached (only in async mode)
      setTimeout(() => {
        if (currentRound && !isAggregating) {
          console.log(`Deadline reached for round ${currentRoundNumber}, starting aggregation...`);
          performAggregation();
        }
      }, deadlineMs);
    }

    // Update all clients to training status
    clients.forEach((client, id) => {
      clients.set(id, {
        ...client,
        status: 'training'
      });
    });

    console.log(`Round ${currentRoundNumber} started with ${clients.size} clients`);
  } catch (error) {
    console.error('Error starting new round:', error);
    // Reset state if initialization fails
    currentRound = null;
    isAggregating = false;
    roundDeadline = null;
    roundStartTime = null;
  }
}

// New function to perform aggregation with client selection
async function performAggregation() {
  if (!currentRound || isAggregating) {
    return;
  }

  isAggregating = true;
  try {
    // Calculate and store mean time for this round before aggregation
    calculateAndStoreRoundMeanTime(currentRoundNumber);
    console.log('Starting weight aggregation...');
    
    // Get selected clients from those that submitted within deadline
    const selectedClients = modelQualityCalculator.getSelectedClients();
    console.log(`Selected clients for aggregation: ${selectedClients.join(', ')}`);
    
    // Filter weights to only include selected clients
    const selectedWeights: { [clientId: string]: { [layer: string]: LayerWeights[] } } = {};
    selectedClients.forEach(clientId => {
      if (currentRound!.modelWeights.clients[clientId]) {
        selectedWeights[clientId] = currentRound!.modelWeights.clients[clientId];
      }
    });
    
    // Aggregate weights from selected clients
    const aggregatedWeights = federatedAveragingAll(selectedWeights);
    globalModel = aggregatedWeights;
    currentRound.modelWeights.aggregated = aggregatedWeights;
    currentRound.status = 'completed';
    rounds.push(currentRound);
    
    console.log(`Round ${currentRoundNumber} completed with ${selectedClients.length} selected clients out of ${Object.keys(currentRound.modelWeights.clients).length} submitted`);
    
    let selectedClientIdsForLog = Object.keys(currentRound.modelWeights.clients);
    // Reset current round
    currentRound = null;
    roundDeadline = null;
    roundStartTime = null;

    // Reset all clients to idle status
    clients.forEach((client, id) => {
      clients.set(id, {
        ...client,
        status: 'idle'
      });
    });

    console.log(`Round ${currentRoundNumber} completed. ${TOTAL_ROUNDS - currentRoundNumber} rounds remaining.`);

    // Start next round if not finished
    isAggregating = false;
    if (currentRoundNumber < TOTAL_ROUNDS) {
      setTimeout(startNewRound, 1000);
    } else {
      console.log('All federated learning rounds completed!');
    }

    // Evaluate global model
    const globalEval = await modelQualityCalculator.evaluateGlobalModel(globalModel);
    console.log(`Global model evaluation - Loss: ${globalEval.loss}, Accuracy: ${globalEval.accuracy}`);

    // Log round data to JSON file
    const roundData = {
      mode: shouldUseSynchronousMode() ? 'synchronous' : 'asynchronous',
      selectedClients,
      submittedClients: selectedClientIdsForLog,
      meanTime: roundMeanTimes[currentRoundNumber],
      globalModel: globalEval,
      clientMetrics: modelQualityCalculator.getClientMetrics()
    };
    logRoundData(currentRoundNumber, roundData);
  } catch (error) {
    isAggregating = false;
    console.error('Error during weight aggregation:', error);
  }
}

// Helper function to check if deadline has passed
function hasDeadlinePassed(): boolean {
  return roundDeadline !== null && Date.now() >= roundDeadline;
}

// Helper function to check if all clients have submitted weights
function haveAllClientsSubmitted(): boolean {
  if (!currentRound) return false;
  const submittedClients = Object.keys(currentRound.modelWeights.clients).length;
  return submittedClients === clients.size;
}

// Modify the federatedAveraging function to use selected clients
function federatedAveraging(clientWeights: { [clientId: string]: { [layer: string]: LayerWeights[] } }): { [layer: string]: LayerWeights[] } {
  const selectedClients = modelQualityCalculator.getSelectedClients();
  if (selectedClients.length === 0) {
    console.warn('No clients selected for aggregation. Using all clients.');
    return federatedAveragingAll(clientWeights);
  }

  console.log(`Selected clients for aggregation: ${selectedClients.join(', ')}`);
  
  const aggregatedWeights: { [layer: string]: LayerWeights[] } = {};
  const numSelectedClients = selectedClients.length;

  // Get the first client's weights to determine the structure
  const firstClientId = selectedClients[0];
  const firstClientWeights = clientWeights[firstClientId];

  // Initialize aggregated weights with the structure from the first client
  for (const layerName in firstClientWeights) {
    aggregatedWeights[layerName] = firstClientWeights[layerName].map(weight => ({
      data: new Array(weight.data.length).fill(0),
      shape: weight.shape
    }));
  }

  // Aggregate weights from selected clients
  for (const clientId of selectedClients) {
    const clientWeight = clientWeights[clientId];
    for (const layerName in clientWeight) {
      const layerWeights = clientWeight[layerName];
      for (let i = 0; i < layerWeights.length; i++) {
        const weight = layerWeights[i];
        for (let j = 0; j < weight.data.length; j++) {
          aggregatedWeights[layerName][i].data[j] += weight.data[j] / numSelectedClients;
        }
      }
    }
  }

  return aggregatedWeights;
}

// Keep the original federated averaging function as a fallback
function federatedAveragingAll(clientWeights: { [clientId: string]: { [layer: string]: LayerWeights[] } }): { [layer: string]: LayerWeights[] } {
  const aggregatedWeights: { [layer: string]: LayerWeights[] } = {};
  const numClients = Object.keys(clientWeights).length;

  // Get the first client's weights to determine the structure
  const firstClientId = Object.keys(clientWeights)[0];
  const firstClientWeights = clientWeights[firstClientId];

  // Initialize aggregated weights with the structure from the first client
  for (const layerName in firstClientWeights) {
    aggregatedWeights[layerName] = firstClientWeights[layerName].map(weight => ({
      data: new Array(weight.data.length).fill(0),
      shape: weight.shape
    }));
  }

  // Aggregate weights from all clients
  for (const clientId in clientWeights) {
    const clientWeight = clientWeights[clientId];
    for (const layerName in clientWeight) {
      const layerWeights = clientWeight[layerName];
      for (let i = 0; i < layerWeights.length; i++) {
        const weight = layerWeights[i];
        for (let j = 0; j < weight.data.length; j++) {
          aggregatedWeights[layerName][i].data[j] += weight.data[j] / numClients;
        }
      }
    }
  }

  return aggregatedWeights;
}

// Register a new client
app.post('/register', (req: Request, res: Response) => {
  // Check if we already have enough clients
  if (clients.size >= REQUIRED_CLIENTS) {
    return res.status(400).json({ 
      error: 'Maximum number of clients reached',
      message: `Server already has ${REQUIRED_CLIENTS} clients registered`
    });
  }

  // Generate a unique client ID
  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  const clientId = `client_${timestamp}_${randomSuffix}`;

  // Add the new client to our map
  clients.set(clientId, {
    clientId,
    status: 'idle',
    lastUpdate: new Date()
  });

  console.log(`Client ${clientId} registered. Total clients: ${clients.size}/${REQUIRED_CLIENTS}`);

  // If we have reached the required number of clients, start the first round
  if (clients.size === REQUIRED_CLIENTS) {
    console.log('Required number of clients reached. Starting federated learning process...');
    startNewRound();
  }

  res.json({ 
    clientId, 
    config: defaultTrainingConfig,
    globalModel: globalModel,
    totalClients: clients.size,
    requiredClients: REQUIRED_CLIENTS,
    currentRound: currentRoundNumber,
    totalRounds: TOTAL_ROUNDS,
    datasetName: serverConfig.dataset
  });
});

// Get current round status
app.get('/round-status', (req: Request, res: Response) => {
  const { clientId } = req.query;
  const client = clients.get(clientId as string);

  if (!client) {
    return res.status(400).json({ error: 'Invalid client' });
  }

  if (!currentRound) {
    // Check if we should start a new round
    if (clients.size >= REQUIRED_CLIENTS && currentRoundNumber < TOTAL_ROUNDS && !isAggregating) {
      startNewRound();
    }

    return res.json({ 
      status: 'no_round',
      globalModel: globalModel,
      currentRound: currentRoundNumber,
      totalRounds: TOTAL_ROUNDS,
      message: currentRoundNumber >= TOTAL_ROUNDS ? 'All rounds completed' : 'Waiting for next round'
    });
  }

  // Check if this client has already submitted weights
  const hasSubmitted = currentRound.modelWeights.clients[clientId as string] !== undefined;

  return res.json({
    status: currentRound.status,
    roundId: currentRound.roundId,
    clientStatus: client.status,
    hasSubmitted,
    waitingForOthers: hasSubmitted && !shouldStartAggregation(),
    submittedClients: Object.keys(currentRound.modelWeights.clients).length,
    totalClients: clients.size,
    globalModel: globalModel,
    currentRound: currentRoundNumber,
    totalRounds: TOTAL_ROUNDS,
    deadline: roundDeadline ? new Date(roundDeadline).toISOString() : null,
    timeRemaining: roundDeadline ? Math.max(0, roundDeadline - Date.now()) : null,
    mode: shouldUseSynchronousMode() ? 'synchronous' : 'asynchronous'
  });
});

// Modify the submit-weights endpoint to handle asynchronous aggregation with client selection
app.post('/submit-weights', async (req: Request, res: Response) => {
  const { clientId, weights, numData, trainingTime, communicationStartTime } = req.body;
  const client = clients.get(clientId);
  
  if (!client || !currentRound) {
    return res.status(400).json({ error: 'Invalid client or no active round' });
  }

  // Check if client has already submitted weights for this round
  if (currentRound.modelWeights.clients[clientId]) {
    return res.status(400).json({ 
      error: 'Already submitted',
      message: 'You have already submitted weights for this round'
    });
  }

  try {
    // Calculate communication time
    const communicationEndTime = performance.now();
    const communicationTime = communicationEndTime - communicationStartTime;

    // Store timing metrics
    const timingMetrics: ClientTimingMetrics = {
      trainingTime,
      communicationTime,
      timestamp: new Date()
    };

    if (!clientTimingMetrics.has(clientId)) {
      clientTimingMetrics.set(clientId, []);
    }
    clientTimingMetrics.get(clientId)!.push(timingMetrics);

    console.log(`Client ${clientId} timing metrics:`, {
      trainingTime: `${trainingTime.toFixed(2)}ms`,
      communicationTime: `${communicationTime.toFixed(2)}ms`,
      round: currentRoundNumber,
      submittedAt: new Date().toISOString(),
      deadline: roundDeadline ? new Date(roundDeadline).toISOString() : 'N/A'
    });

    // Evaluate model quality
    const quality = await modelQualityCalculator.evaluateModel(clientId, weights, numData);
    console.log(`Client ${clientId} model quality: ${quality}`);

    // Update client status
    client.status = 'aggregating';
    client.lastUpdate = new Date();
    clients.set(clientId, client);

    // Store the weights in the current round
    currentRound.modelWeights.clients[clientId] = weights;

    // Check if we should start aggregation (all clients submitted or deadline passed)
    if (shouldStartAggregation()) {
      await performAggregation();
      
      return res.json({ 
        status: 'success',
        globalModel: globalModel,
        currentRound: currentRoundNumber,
        totalRounds: TOTAL_ROUNDS,
        message: currentRoundNumber >= TOTAL_ROUNDS ? 'All rounds completed' : 'Round completed, waiting for next round'
      });
    }

    res.json({ 
      status: 'success',
      message: 'Weights received, waiting for other clients or deadline',
      submittedClients: Object.keys(currentRound.modelWeights.clients).length,
      totalClients: clients.size,
      currentRound: currentRoundNumber,
      totalRounds: TOTAL_ROUNDS,
      deadline: roundDeadline ? new Date(roundDeadline).toISOString() : null
    });
  } catch (error) {
    console.error('Error processing weights:', error);
    return res.status(500).json({ error: 'Failed to process weights' });
  }
});

// Get global model weights
app.get('/global-model', (req: Request, res: Response) => {
  if (!globalModel) {
    return res.status(404).json({ 
      error: 'No aggregated model available',
      message: 'Please complete at least one training round'
    });
  }
  res.json({ 
    weights: globalModel,
    currentRound: currentRoundNumber,
    totalRounds: TOTAL_ROUNDS
  });
});

// Get training progress
app.get('/progress', (req: Request, res: Response) => {
  res.json({
    clients: {
      current: clients.size,
      required: REQUIRED_CLIENTS
    },
    rounds: {
      current: currentRoundNumber,
      total: TOTAL_ROUNDS
    },
    currentRound: currentRound ? {
      status: currentRound.status,
      submittedClients: Object.keys(currentRound.modelWeights.clients).length,
      totalClients: clients.size
    } : null
  });
});

// Add new endpoint to get and update server configuration
app.get('/config', (req: Request, res: Response) => {
  res.json(serverConfig);
});

app.post('/config', (req: Request, res: Response) => {
  const newConfig: Partial<ServerConfig> = req.body;
  
  // Validate the new configuration
  if (newConfig.totalRounds !== undefined && newConfig.totalRounds < 1) {
    return res.status(400).json({ error: 'Total rounds must be at least 1' });
  }
  if (newConfig.requiredClients !== undefined && newConfig.requiredClients < 1) {
    return res.status(400).json({ error: 'Required clients must be at least 1' });
  }
  if (newConfig.aggregationInterval !== undefined && newConfig.aggregationInterval < 100) {
    return res.status(400).json({ error: 'Aggregation interval must be at least 100ms' });
  }

  // Update the configuration
  serverConfig = { ...serverConfig, ...newConfig };
  
  // Update the global constants
  REQUIRED_CLIENTS = serverConfig.requiredClients;
  TOTAL_ROUNDS = serverConfig.totalRounds;

  console.log('Server configuration updated:', serverConfig);
  res.json(serverConfig);
});

// Add endpoint to get list of connected clients
app.get('/clients', (req: Request, res: Response) => {
  const clientList = Array.from(clients.entries()).map(([id, client]) => ({
    id: client.clientId,
    status: client.status,
    lastUpdate: client.lastUpdate,
    connectedAt: client.lastUpdate // Using lastUpdate as connection time
  }));
  
  res.json(clientList);
});

// Add endpoint to disconnect a client
app.delete('/clients/:clientId', (req: Request, res: Response) => {
  const { clientId } = req.params;
  
  if (!clients.has(clientId)) {
    return res.status(404).json({ error: 'Client not found' });
  }

  // Remove client from the map
  clients.delete(clientId);
  console.log(`Client ${clientId} disconnected by admin`);

  res.json({ message: 'Client disconnected successfully' });
});

// Add new endpoint to get client metrics
app.get('/client-metrics', (req: Request, res: Response) => {
  const metrics = modelQualityCalculator.getClientMetrics();
  res.json(metrics);
});

// Add new endpoint to get client timing metrics
app.get('/client-timing-metrics', (req: Request, res: Response) => {
  const metrics: { [clientId: string]: ClientTimingMetrics[] } = {};
  clientTimingMetrics.forEach((timings, clientId) => {
    metrics[clientId] = timings;
  });
  res.json(metrics);
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Federated Learning Server running on port ${PORT}`);
  console.log(`Waiting for ${REQUIRED_CLIENTS} clients to connect...`);
}); 