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
import * as fs from 'fs';
import * as path from 'path';

// Constants for model quality calculation
const THETA = 0.01;
const PHI = 0.5;

interface ModelQualityMetrics {
    clientId: string;
    loss: number;
    previousLoss: number | null;
    lossDifference: number | null;
    numData: number;
    quality: number;
}

export class ModelQualityCalculator {
    private testData!: tf.Tensor;  // Using definite assignment assertion
    private testLabels!: tf.Tensor;  // Using definite assignment assertion
    private model: tf.LayersModel | null = null;
    private clientMetrics: Map<string, ModelQualityMetrics> = new Map();
    private previousRoundMetrics: Map<string, number> = new Map();
    private dataset: string = 'fashion-mnist';

    constructor(dataset: string = 'fashion-mnist') {
        this.dataset = dataset;
        this.loadTestData();
    }

    private async loadTestData() {
        try {
            if (this.dataset.toLowerCase() === 'cifar10') {
                await this.loadCIFAR10TestData();
            } else if (this.dataset.toLowerCase() === 'mnist') {
                await this.loadMNISTTestData();
            } else {
                await this.loadFashionMNISTTestData();
            }
        } catch (error) {
            console.error('Error loading test data:', error);
            throw error;
        }
    }

    private async loadCIFAR10TestData() {
        console.log('Loading CIFAR-10 test data...');
        const testDataPath = path.join(__dirname, '../data/cifar10/test_batch.bin');
        const testDataBuffer = fs.readFileSync(testDataPath);
        
        // CIFAR-10 binary format: 3073 bytes per image (1 label + 3072 pixel values)
        const bytesPerImage = 3073;
        const numImages = testDataBuffer.length / bytesPerImage;
        
        // Create arrays for images and labels
        const imageArray = new Float32Array(numImages * 32 * 32 * 3);
        const labelArray = new Int32Array(numImages);
        
        for (let i = 0; i < numImages; i++) {
            const offset = i * bytesPerImage;
            
            // First byte is the label
            labelArray[i] = testDataBuffer[offset];
            
            // Next 3072 bytes are the image data (32x32x3)
            for (let j = 0; j < 3072; j++) {
                const pixelValue = testDataBuffer[offset + 1 + j] / 255.0; // Normalize to [0, 1]
                imageArray[i * 3072 + j] = pixelValue;
            }
        }
        
        // Create tensors
        this.testData = tf.tensor4d(imageArray, [numImages, 32, 32, 3]);
        this.testLabels = tf.oneHot(tf.tensor1d(labelArray, 'int32'), 10);
        
        console.log(`CIFAR-10 test data loaded: ${numImages} images`);
    }

    private async loadMNISTTestData() {
        console.log('Loading MNIST test data...');
        const testDataPath = path.join(__dirname, '../data/mnist/t10k-images-idx3-ubyte');
        const testLabelsPath = path.join(__dirname, '../data/mnist/t10k-labels-idx1-ubyte');

        const testDataBuffer = fs.readFileSync(testDataPath);
        const testLabelsBuffer = fs.readFileSync(testLabelsPath);

        // Parse test data
        const numImages = new DataView(testDataBuffer.buffer).getInt32(4, false);
        const numRows = new DataView(testDataBuffer.buffer).getInt32(8, false);
        const numCols = new DataView(testDataBuffer.buffer).getInt32(12, false);

        // Skip header (16 bytes)
        const imageArray = new Float32Array(numImages * numRows * numCols);
        for (let i = 0; i < numImages * numRows * numCols; i++) {
            imageArray[i] = testDataBuffer[i + 16] / 255.0;
        }

        // Reshape to [numImages, 784] for the neural network
        this.testData = tf.tensor2d(imageArray, [numImages, numRows * numCols]);

        // Parse test labels
        const numLabels = new DataView(testLabelsBuffer.buffer).getInt32(4, false);
        const labelArray = new Int32Array(numLabels);
        for (let i = 0; i < numLabels; i++) {
            labelArray[i] = testLabelsBuffer[i + 8];
        }

        this.testLabels = tf.oneHot(tf.tensor1d(labelArray, 'int32'), 10);
        console.log(`MNIST test data loaded: ${numImages} images`);
    }

    private async loadFashionMNISTTestData() {
        console.log('Loading Fashion MNIST test data...');
        const testDataPath = path.join(__dirname, '../data/fashion-mnist/t10k-images-idx3-ubyte');
        const testLabelsPath = path.join(__dirname, '../data/fashion-mnist/t10k-labels-idx1-ubyte');

        const testDataBuffer = fs.readFileSync(testDataPath);
        const testLabelsBuffer = fs.readFileSync(testLabelsPath);

        // Parse test data
        const numImages = new DataView(testDataBuffer.buffer).getInt32(4, false);
        const numRows = new DataView(testDataBuffer.buffer).getInt32(8, false);
        const numCols = new DataView(testDataBuffer.buffer).getInt32(12, false);

        // Skip header (16 bytes)
        const imageArray = new Float32Array(numImages * numRows * numCols);
        for (let i = 0; i < numImages * numRows * numCols; i++) {
            imageArray[i] = testDataBuffer[i + 16] / 255.0;
        }

        // Reshape to [numImages, 784] for the neural network
        this.testData = tf.tensor2d(imageArray, [numImages, numRows * numCols]);

        // Parse test labels
        const numLabels = new DataView(testLabelsBuffer.buffer).getInt32(4, false);
        const labelArray = new Int32Array(numLabels);
        for (let i = 0; i < numLabels; i++) {
            labelArray[i] = testLabelsBuffer[i + 8];
        }

        this.testLabels = tf.oneHot(tf.tensor1d(labelArray, 'int32'), 10);
        console.log(`Fashion MNIST test data loaded: ${numImages} images`);
    }

    public async initializeModel(architecture: any) {
        // Create layers array
        const layers = architecture.layers.map((layer: any) => {
            switch (layer.type) {
                case 'dense':
                    return tf.layers.dense({
                        units: layer.units,
                        activation: layer.activation,
                        inputShape: layer.inputShape
                    });
                case 'conv2d':
                    return tf.layers.conv2d({
                        filters: layer.filters,
                        kernelSize: layer.kernelSize,
                        activation: layer.activation,
                        inputShape: layer.inputShape
                    });
                case 'maxPooling2d':
                    return tf.layers.maxPooling2d({
                        poolSize: layer.poolSize
                    });
                case 'flatten':
                    return tf.layers.flatten();
                default:
                    throw new Error(`Unsupported layer type: ${layer.type}`);
            }
        });

        // Create model with layers
        this.model = tf.sequential({ layers });

        this.model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    public async evaluateModel(clientId: string, weights: any, numData: number): Promise<number> {
        if (!this.model) {
            throw new Error('Model not initialized');
        }

        try {
            // Convert weights to the correct format
            const formattedWeights = Object.entries(weights).reduce((acc: { [key: string]: tf.Tensor[] }, [layerName, layerWeights]) => {
                const weightsArray = (layerWeights as any[]).map(w => {
                    if ('data' in w && 'shape' in w) {
                        return tf.tensor(w.data, w.shape);
                    } else {
                        throw new Error(`Invalid weight format for layer ${layerName}`);
                    }
                });
                acc[layerName] = weightsArray;
                return acc;
            }, {});

            // Set model weights
            const layers = this.model.layers;
            for (let i = 0; i < layers.length; i++) {
                const layer = layers[i];
                const layerWeights = formattedWeights[`layer_${i}`];
                if (layerWeights) {
                    layer.setWeights(layerWeights);
                }
            }

            // Evaluate model on test data
            console.log('Evaluating model on test data...');
            console.log('Test data shape:', this.testData.shape);
            console.log('Test labels shape:', this.testLabels.shape);
            
            const evaluation = await this.model.evaluate(this.testData, this.testLabels) as tf.Scalar[];
            const currentLoss = evaluation[0].dataSync()[0];
            const accuracy = evaluation[1].dataSync()[0];
            
            console.log(`Model evaluation - Current Loss: ${currentLoss}, Accuracy: ${accuracy}`);

            if (isNaN(currentLoss)) {
                console.error('Loss is NaN, using default quality score');
                return 0.5;
            }

            // Get previous loss for this client
            const previousLoss = this.previousRoundMetrics.get(clientId) || null;
            const lossDifference = previousLoss !== null ? previousLoss - currentLoss : null;

            console.log(`Client ${clientId} - Previous Loss: ${previousLoss}, Loss Difference: ${lossDifference}`);

            // Calculate model quality using loss difference
            const quality = this.computeModelQuality(lossDifference || currentLoss, numData);
            console.log(`Computed quality score for client ${clientId}: ${quality}`);

            // Store metrics
            this.clientMetrics.set(clientId, {
                clientId,
                loss: currentLoss,
                previousLoss,
                lossDifference,
                numData,
                quality
            });

            // Store current loss for next round
            this.previousRoundMetrics.set(clientId, currentLoss);

            return quality;
        } catch (error) {
            console.error('Error evaluating model:', error);
            return 0.5;
        }
    }

    private computeModelQuality(loss: number, numData: number): number {
        try {
            // Ensure loss is a valid number
            if (isNaN(loss) || !isFinite(loss)) {
                console.warn('Invalid loss value:', loss);
                return 0.5;
            }

            // Ensure numData is a valid number
            if (isNaN(numData) || !isFinite(numData) || numData <= 0) {
                console.warn('Invalid numData value:', numData);
                return 0.5;
            }

            // For loss difference, we want to reward larger improvements
            const calculatedDataQualityRatio = Math.abs(loss) * numData;
            const poweredRatio = Math.pow(calculatedDataQualityRatio, PHI);

            let tmpPerformanceScore: number;
            if (loss > 0) {
                tmpPerformanceScore = 1 - Math.exp(THETA * poweredRatio);
            } else {
                tmpPerformanceScore = 1 - Math.exp(-THETA * poweredRatio);
            }

            // Ensure the final score is a valid number
            if (isNaN(tmpPerformanceScore) || !isFinite(tmpPerformanceScore)) {
                console.warn('Invalid performance score calculated:', tmpPerformanceScore);
                return 0.5;
            }

            return tmpPerformanceScore;
        } catch (error) {
            console.error('Error computing model quality:', error);
            return 0.5;
        }
    }

    public getSelectedClients(): string[] {
        const metrics = Array.from(this.clientMetrics.values());
        if (metrics.length === 0) return [];

        // Calculate mean quality
        const meanQuality = metrics.reduce((sum, m) => sum + m.quality, 0) / metrics.length;

        // Select clients with quality above mean
        return metrics
            .filter(m => m.quality > meanQuality)
            .map(m => m.clientId);
    }

    public getClientMetrics(): ModelQualityMetrics[] {
        return Array.from(this.clientMetrics.values());
    }

    public clearMetrics() {
        this.clientMetrics.clear();
        // Don't clear previousRoundMetrics as we need it for the next round
    }

    public clearAllMetrics() {
        this.clientMetrics.clear();
        this.previousRoundMetrics.clear();
    }

    // Evaluate global model
    public async evaluateGlobalModel(weights: any): Promise<{ loss: number, accuracy: number }> {
        if (!this.model) throw new Error('Model not initialized');
        // Set model weights
        const formattedWeights = Object.entries(weights).reduce((acc: { [key: string]: tf.Tensor[] }, [layerName, layerWeights]) => {
          const weightsArray = (layerWeights as any[]).map(w => tf.tensor(w.data, w.shape));
          acc[layerName] = weightsArray;
          return acc;
        }, {});
        for (let i = 0; i < this.model.layers.length; i++) {
          const layer = this.model.layers[i];
          const layerWeights = formattedWeights[`layer_${i}`];
          if (layerWeights) layer.setWeights(layerWeights);
        }
        const evaluation = await this.model.evaluate(this.testData, this.testLabels) as tf.Scalar[];
        return {
          loss: evaluation[0].dataSync()[0],
          accuracy: evaluation[1].dataSync()[0]
        };
      }
} 