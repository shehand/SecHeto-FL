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

export interface ClientInfo {
  clientId: string;
  status: 'idle' | 'training' | 'aggregating';
  lastUpdate: Date;
}

export interface LayerWeights {
  data: number[];
  shape: number[];
}

export interface ModelWeights {
  clients: { [clientId: string]: { [layer: string]: LayerWeights[] } };
  aggregated?: { [layer: string]: LayerWeights[] };
}

export interface FederatedRound {
  roundId: number;
  participants: string[];
  modelWeights: ModelWeights;
  status: 'in_progress' | 'completed';
}

type ActivationType = 'relu' | 'sigmoid' | 'softmax' | 'tanh' | 'linear';

export interface LayerConfig {
  type: string;
  units?: number;
  activation?: string;
  inputShape?: number[];
  filters?: number;
  kernelSize?: number;
  poolSize?: number;
}

export interface ModelArchitecture {
  inputShape: number[];
  layers: LayerConfig[];
}

export interface TrainingConfig {
  batchSize: number;
  learningRate: number;
  epochs: number;
  modelArchitecture: ModelArchitecture;
} 