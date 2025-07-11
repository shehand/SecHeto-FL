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

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

type ActivationType = 'relu' | 'sigmoid' | 'softmax' | 'tanh' | 'linear';

export interface LayerConfig {
  type: string;
  units?: number;
  activation?: string;
  inputShape?: number[];
  filters?: number;
  kernelSize?: number | [number, number];
  poolSize?: number | [number, number];
}

export interface ModelArchitecture {
  inputShape: number[];
  layers: LayerConfig[];
}

interface LayerWeights {
  data: number[];
  shape: number[];
}

interface ModelWeights {
  [layer: string]: LayerWeights[];
}

type WeightData = LayerWeights | number[];

export class NeuralNetwork {
  private model: tf.LayersModel | null = null;
  private architecture: ModelArchitecture | null = null;
  
  public isInitialized(): boolean {
    return this.model !== null;
  }

  async initialize(architecture: ModelArchitecture) {
    try {
      this.architecture = architecture;
      
      // Try to set WebGPU as the backend with fallback
      try {
        await tf.setBackend('webgpu');
        await tf.ready();
        console.log('WebGPU backend initialized successfully');
      } catch (webgpuError) {
        console.warn('WebGPU not available, trying WebGL backend:', webgpuError);
        
        try {
          await tf.setBackend('webgl');
          await tf.ready();
          console.log('WebGL backend initialized successfully');
        } catch (webglError) {
          console.warn('WebGL not available, falling back to CPU backend:', webglError);
          
          await tf.setBackend('cpu');
          await tf.ready();
          console.log('CPU backend initialized successfully (slower performance)');
        }
      }
      
      // Create the model based on the architecture
      const layers = architecture.layers.map((layer: LayerConfig, index: number) => {
        switch (layer.type) {
          case 'dense':
            return tf.layers.dense({
              units: layer.units!,
              activation: layer.activation as ActivationType,
              // Add input shape for the first layer only
              inputShape: index === 0 ? architecture.inputShape : undefined
            });
          case 'conv2d':
            return tf.layers.conv2d({
              filters: layer.filters!,
              kernelSize: layer.kernelSize!,
              activation: layer.activation as ActivationType,
              inputShape: index === 0 ? architecture.inputShape : undefined
            });
          case 'maxPooling2d':
            return tf.layers.maxPooling2d({
              poolSize: layer.poolSize!
            });
          case 'flatten':
            return tf.layers.flatten();
          default:
            throw new Error(`Unsupported layer type: ${layer.type}`);
        }
      });

      this.model = tf.sequential({ layers });

      // Log model summary
      this.model.summary();
      
      // Compile the model with default settings
      this.model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
    } catch (error) {
      console.error('Failed to initialize neural network:', error);
      throw error;
    }
  }

  async train(data: tf.Tensor, labels: tf.Tensor, config: any) {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    // Validate input shape
    const inputShape = data.shape.slice(1);
    if (!tf.util.arraysEqual(inputShape, this.architecture!.inputShape)) {
      throw new Error(`Input shape mismatch. Expected ${this.architecture!.inputShape}, got ${inputShape}`);
    }

    // Update optimizer settings if provided
    if (config.learningRate) {
      this.model.compile({
        optimizer: tf.train.adam(config.learningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
    }

    // Train the model
    const history = await this.model.fit(data, labels, {
      batchSize: config.batchSize || 32,
      epochs: config.epochs || 1,
      shuffle: true,
      verbose: 1
    });

    return history;
  }

  async getWeights(): Promise<ModelWeights> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    const weights: ModelWeights = {};
    
    // Use async data reading to avoid performance warning
    const layerWeightsPromises = this.model.layers.map(async (layer, index) => {
      const layerWeights = layer.getWeights();
      if (layerWeights.length > 0) {
        weights[`layer_${index}`] = await Promise.all(
          layerWeights.map(async w => {
            const data = await w.data();
            return {
              data: Array.from(data as Float32Array),
              shape: Array.from(w.shape)
            };
          })
        );
      }
    });

    await Promise.all(layerWeightsPromises);
    return weights;
  }

  async setWeights(weights: ModelWeights | { [layer: string]: number[][] }) {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    try {
      Object.entries(weights).forEach(([layerName, layerWeights], index) => {
        const layer = this.model!.layers[index];
        const originalWeights = layer.getWeights();
        
        if (originalWeights.length !== layerWeights.length) {
          throw new Error(`Weight count mismatch for layer ${layerName}. Expected ${originalWeights.length}, got ${layerWeights.length}`);
        }

        const tensors = layerWeights.map((w: WeightData, i: number) => {
          const originalShape = originalWeights[i].shape;
          
          // Handle both old and new weight formats
          if ('data' in w && 'shape' in w) {
            return tf.tensor(w.data, w.shape);
          } else {
            // Handle legacy format where w is just an array
            const flattenedData = Array.isArray(w) ? w.flat() : w;
            return tf.tensor(flattenedData, originalShape);
          }
        });

        // Log shapes for debugging
        console.log(`Setting weights for layer ${layerName}:`, {
          originalShapes: originalWeights.map(w => w.shape),
          newShapes: tensors.map((t: tf.Tensor) => t.shape)
        });
        
        layer.setWeights(tensors);
      });
    } catch (err) {
      const error = err as Error;
      console.error('Error setting weights:', error);
      throw new Error(`Failed to set weights: ${error.message}`);
    }
  }

  async predict(data: tf.Tensor): Promise<tf.Tensor> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }
    return this.model.predict(data) as tf.Tensor;
  }
} 