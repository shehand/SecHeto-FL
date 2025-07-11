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

import axios from 'axios';
import { NeuralNetwork } from './NeuralNetwork';
import * as tf from '@tensorflow/tfjs';
import { DatasetService } from './DatasetService';

const API_URL = 'http://localhost:3001';
const POLLING_INTERVAL = 2000; // 2 seconds

interface LayerWeights {
  data: number[];
  shape: number[];
}

interface ModelWeights {
  [layer: string]: LayerWeights[];
}

interface RoundStatus {
  status: 'no_round' | 'in_progress' | 'completed';
  roundId?: number;
  clientStatus?: string;
  hasSubmitted?: boolean;
  waitingForOthers?: boolean;
  submittedClients?: number;
  totalClients?: number;
  globalModel?: ModelWeights;
  currentRound?: number;
  totalRounds?: number;
  message?: string;
}

export class FederatedClient {
  private clientId: string | null = null;
  private neuralNetwork: NeuralNetwork;
  private trainingConfig: any = null;
  private pollingInterval: NodeJS.Timeout | null = null;
  private onProgressCallback: ((progress: any) => void) | null = null;
  private isTraining: boolean = false;
  private isRegistered: boolean = false;
  private datasetService: DatasetService;
  private datasetName: string = 'fashion-mnist'; // default

  constructor() {
    this.neuralNetwork = new NeuralNetwork();
    this.datasetService = DatasetService.getInstance();
  }

  setProgressCallback(callback: (progress: any) => void) {
    this.onProgressCallback = callback;
  }

  async register(): Promise<void> {
    if (this.isRegistered) {
      console.warn('Client already registered');
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/register`);
      this.clientId = response.data.clientId;
      this.trainingConfig = response.data.config;
      this.datasetName = response.data.datasetName || 'fashion-mnist';
      
      // Initialize the neural network with the received configuration
      await this.neuralNetwork.initialize(this.trainingConfig.modelArchitecture);

      // If there's a global model available, use it
      if (response.data.globalModel) {
        console.log('Initializing with global model weights');
        await this.neuralNetwork.setWeights(response.data.globalModel);
      }

      this.isRegistered = true;
      console.log('Client registered and model initialized successfully');
      console.log(`Clients connected: ${response.data.totalClients}/${response.data.requiredClients}`);

      // Update UI with initial status
      if (this.onProgressCallback) {
        this.onProgressCallback({
          status: 'waiting',
          message: `Waiting for other clients (${response.data.totalClients}/${response.data.requiredClients})`,
          currentRound: 0,
          totalRounds: response.data.totalRounds,
          submittedClients: response.data.totalClients,
          totalClients: response.data.requiredClients
        });
      }

      // Start polling for round status
      this.startPolling();
    } catch (error: any) {
      if (error.response?.status === 400 && error.response?.data?.error === 'Maximum number of clients reached') {
        console.log('Server has reached maximum number of clients');
        throw new Error('Server is full');
      }
      console.error('Failed to register client:', error);
      throw error;
    }
  }

  private startPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
    }

    this.pollingInterval = setInterval(async () => {
      if (this.isTraining) return; // Skip polling if currently training

      try {
        const status = await this.getRoundStatus();
        
        // Always update UI with status
        if (this.onProgressCallback) {
          this.onProgressCallback(status);
        }

        // Handle different round states
        if (status.status === 'in_progress') {
          if (status.clientStatus === 'training' && !status.hasSubmitted) {
            // Clear interval while we're training
            this.isTraining = true;
            clearInterval(this.pollingInterval!);
            this.pollingInterval = null;

            // Start training
            await this.participateInTraining();

            // Resume polling after training
            this.isTraining = false;
            this.startPolling();
          } else if (status.waitingForOthers) {
            console.log(`Waiting for other clients... (${status.submittedClients}/${status.totalClients})`);
          }
        } else if (status.status === 'completed' && status.globalModel) {
          console.log('Round completed, updating with global model');
          await this.neuralNetwork.setWeights(status.globalModel);
        }
      } catch (error) {
        console.error('Error during status polling:', error);
        // Resume polling even if there was an error
        this.isTraining = false;
      }
    }, POLLING_INTERVAL);
  }

  private getStatusMessage(status: RoundStatus): string {
    if (status.status === 'no_round') {
      return `Waiting for other clients (${status.submittedClients || 0}/${status.totalClients || 10})`;
    } else if (status.status === 'in_progress') {
      if (status.clientStatus === 'training') {
        return 'Training in progress...';
      } else if (status.waitingForOthers) {
        return `Waiting for other clients to complete training (${status.submittedClients}/${status.totalClients})`;
      }
      return 'Round in progress';
    } else if (status.status === 'completed') {
      return 'Round completed';
    }
    return 'Waiting for server...';
  }

  private async getRoundStatus(): Promise<RoundStatus> {
    if (!this.clientId) {
      throw new Error('Client not registered');
    }

    const response = await axios.get(`${API_URL}/round-status`, {
      params: { clientId: this.clientId }
    });

    // Update model with global weights if available
    if (response.data.globalModel) {
      console.log('Updating with latest global model weights');
      await this.neuralNetwork.setWeights(response.data.globalModel);
    }

    return response.data;
  }

  public async participateInTraining(): Promise<void> {
    if (!this.clientId || !this.trainingConfig) {
      throw new Error('Client not registered');
    }

    try {
      console.log('Starting training participation...');
      // Use the datasetName provided by the server
      const datasetName = this.datasetName;
      
      // First load the full dataset
      const fullDataset = await this.datasetService.loadDataset(datasetName);
      
      // Then get the client-specific non-IID subset
      const clientDataset = await this.datasetService.getClientDataset(fullDataset, this.clientId);
      
      console.log('Starting local training with non-IID data distribution...');

      // Start timing the training
      const trainingStartTime = performance.now();

      // Train the model locally with the client's subset
      const history = await this.neuralNetwork.train(clientDataset.data, clientDataset.labels, this.trainingConfig);
      
      // Calculate training time
      const trainingEndTime = performance.now();
      const trainingTime = trainingEndTime - trainingStartTime;
      console.log(`Local training completed in ${trainingTime.toFixed(2)}ms`, history);
      
      // Get the local model weights
      console.log('Getting local weights...');
      const weights = await this.neuralNetwork.getWeights();
      
      // Get the number of data samples
      const numData = clientDataset.data.shape[0];
      console.log(`Number of data samples: ${numData}`);
      
      // Log weights for debugging
      console.log('Local weights structure:', {
        layers: Object.keys(weights),
        shapes: Object.entries(weights).map(([layer, w]) => ({
          layer,
          shapes: w.map(weight => weight.shape)
        }))
      });

      // Submit the weights to the server
      console.log('Submitting weights to server...');
      // Add timestamp for communication time measurement
      const communicationStartTime = performance.now();

      const response = await axios.post(
        `${API_URL}/submit-weights`,
        {
          clientId: this.clientId,
          weights,
          numData,
          trainingTime,
          communicationStartTime
        },
        {
          headers: {
            'Content-Type': 'application/json'
          },
          maxContentLength: Infinity,
          maxBodyLength: Infinity
        }
      );
      console.log('Server response:', response.data);

      // Cleanup tensors
      fullDataset.data.dispose();
      fullDataset.labels.dispose();
      clientDataset.data.dispose();
      clientDataset.labels.dispose();
      
      console.log('Training participation completed successfully');
    } catch (error) {
      console.error('Error during training participation:', error);
      throw error;
    }
  }

  async predict(data: tf.Tensor): Promise<tf.Tensor> {
    if (!this.clientId) {
      throw new Error('Client not registered');
    }
    return this.neuralNetwork.predict(data);
  }

  disconnect() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.isRegistered = false;
    this.clientId = null;
    this.trainingConfig = null;
    this.isTraining = false;
  }
} 