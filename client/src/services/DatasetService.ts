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

export interface Dataset {
  data: tf.Tensor;
  labels: tf.Tensor;
  numClasses: number;
  inputShape: number[];
}

export interface ClientDataset {
  data: tf.Tensor;
  labels: tf.Tensor;
  distribution: number[];  // Distribution of classes in this client's dataset
}

export class DatasetService {
  private static instance: DatasetService;
  private dataset: Dataset | null = null;
  private readonly DATASET_DIR = '/datasets'; // This should be a public directory in your web server
  private readonly NUM_CLIENTS = 10;  // Number of clients to distribute data to
  private readonly MIN_SAMPLES_PER_CLIENT = 100;  // Minimum samples per client
  private readonly MAX_SAMPLES_PER_CLIENT = 1000;  // Maximum samples per client

  private constructor() {}

  public static getInstance(): DatasetService {
    if (!DatasetService.instance) {
      DatasetService.instance = new DatasetService();
    }
    return DatasetService.instance;
  }

  public async loadDataset(datasetName: string): Promise<Dataset> {
    try {
      console.log(`Attempting to load dataset: ${datasetName}`);
      switch (datasetName.toLowerCase()) {
        case 'cifar10':
          return await this.loadCIFAR10();
        case 'fashion-mnist':
          return await this.loadFashionMNIST();
        case 'mnist':
          return await this.loadMNIST();
        default:
          console.warn(`Unsupported dataset: ${datasetName}, falling back to random data`);
          return this.generateRandomData(datasetName);
      }
    } catch (error) {
      console.error(`Error loading dataset ${datasetName}:`, error);
      console.log('Falling back to random data generation');
      return this.generateRandomData(datasetName);
    }
  }

  private async loadCIFAR10(): Promise<Dataset> {
    console.log('Loading CIFAR10 dataset...');
    try {
      // Load training data from binary files
      const batchFiles = ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin', 'data_batch_4.bin', 'data_batch_5.bin'];
      const allImages: number[][][][] = [];
      const allLabels: number[] = [];
      
      for (const batchFile of batchFiles) {
        console.log(`Loading ${batchFile}...`);
        const response = await fetch(`${this.DATASET_DIR}/cifar10/${batchFile}`);
        if (!response.ok) {
          throw new Error(`Failed to load ${batchFile}: ${response.statusText}`);
        }
        
        const buffer = await response.arrayBuffer();
        const data = new Uint8Array(buffer);
        
        // CIFAR-10 binary format: 3073 bytes per image (1 label + 3072 pixel values)
        const bytesPerImage = 3073;
        const numImages = data.length / bytesPerImage;
        
        for (let i = 0; i < numImages; i++) {
          const offset = i * bytesPerImage;
          
          // First byte is the label
          const label = data[offset];
          allLabels.push(label);
          
          // Next 3072 bytes are the image data (32x32x3)
          const imageData = new Array(32 * 32 * 3);
          for (let j = 0; j < 3072; j++) {
            imageData[j] = data[offset + 1 + j] / 255.0; // Normalize to [0, 1]
          }
          
          // Reshape to 32x32x3
          const image = new Array(32);
          for (let y = 0; y < 32; y++) {
            image[y] = new Array(32);
            for (let x = 0; x < 32; x++) {
              image[y][x] = new Array(3);
              for (let c = 0; c < 3; c++) {
                const idx = (y * 32 + x) * 3 + c;
                image[y][x][c] = imageData[idx];
              }
            }
          }
          
          allImages.push(image);
        }
      }
      
      console.log(`Loaded ${allImages.length} CIFAR-10 images`);
      
      // Convert to tensors - flatten all images into a single 4D tensor
      const flattenedImages = new Float32Array(allImages.length * 32 * 32 * 3);
      for (let i = 0; i < allImages.length; i++) {
        const image: number[][][] = allImages[i];
        for (let y = 0; y < 32; y++) {
          for (let x = 0; x < 32; x++) {
            for (let c = 0; c < 3; c++) {
              const flatIndex = i * 32 * 32 * 3 + y * 32 * 3 + x * 3 + c;
              flattenedImages[flatIndex] = image[y][x][c];
            }
          }
        }
      }
      
      const images = tf.tensor4d(flattenedImages, [allImages.length, 32, 32, 3]);
      const labels = tf.oneHot(
        tf.tensor1d(allLabels, 'int32'),
        10
      );

      return {
        data: images,
        labels,
        numClasses: 10,
        inputShape: [32, 32, 3]
      };
    } catch (error) {
      console.error('Error loading CIFAR10:', error);
      return this.generateRandomData('cifar10');
    }
  }

  private async loadFashionMNIST(): Promise<Dataset> {
    console.log('Loading Fashion MNIST dataset...');
    try {
      // Load images
      console.log('Fetching Fashion MNIST images...');
      const imagesResponse = await fetch('/datasets/fashion-mnist/train-images-idx3-ubyte');
      if (!imagesResponse.ok) {
        throw new Error(`Failed to load Fashion MNIST images: ${imagesResponse.statusText}`);
      }
      console.log('Successfully fetched Fashion MNIST images');
      
      const imageBuffer = await imagesResponse.arrayBuffer();
      const imageData = new Uint8Array(imageBuffer);
      
      // Parse image data
      const magicNumber = new DataView(imageBuffer).getInt32(0, false);
      if (magicNumber !== 2051) {
        throw new Error(`Invalid magic number for Fashion MNIST images: ${magicNumber}`);
      }

      const numImages = new DataView(imageBuffer).getInt32(4, false);
      const numRows = new DataView(imageBuffer).getInt32(8, false);
      const numCols = new DataView(imageBuffer).getInt32(12, false);
      
      console.log(`Fashion MNIST: ${numImages} images, ${numRows}x${numCols} pixels`);
      
      if (numImages <= 0 || numRows !== 28 || numCols !== 28) {
        throw new Error(`Invalid Fashion MNIST dimensions: ${numImages} images, ${numRows}x${numCols} pixels`);
      }

      // Skip header (16 bytes)
      const imageArray = new Float32Array(numImages * numRows * numCols);
      for (let i = 0; i < numImages * numRows * numCols; i++) {
        imageArray[i] = imageData[i + 16] / 255.0;
      }
      
      // Reshape to [numImages, 784] for the neural network
      const images = tf.tensor2d(
        imageArray,
        [numImages, numRows * numCols]
      );

      // Load labels
      console.log('Fetching Fashion MNIST labels...');
      const labelsResponse = await fetch('/datasets/fashion-mnist/train-labels-idx1-ubyte');
      if (!labelsResponse.ok) {
        throw new Error(`Failed to load Fashion MNIST labels: ${labelsResponse.statusText}`);
      }
      console.log('Successfully fetched Fashion MNIST labels');
      
      const labelBuffer = await labelsResponse.arrayBuffer();
      const labelData = new Uint8Array(labelBuffer);
      
      // Parse label data
      const labelMagicNumber = new DataView(labelBuffer).getInt32(0, false);
      if (labelMagicNumber !== 2049) {
        throw new Error(`Invalid magic number for Fashion MNIST labels: ${labelMagicNumber}`);
      }

      const numLabels = new DataView(labelBuffer).getInt32(4, false);
      
      console.log(`Fashion MNIST: ${numLabels} labels`);
      
      if (numLabels !== numImages) {
        throw new Error(`Number of labels (${numLabels}) does not match number of images (${numImages})`);
      }

      // Skip header (8 bytes)
      const labelArray = new Int32Array(numLabels);
      for (let i = 0; i < numLabels; i++) {
        labelArray[i] = labelData[i + 8];
      }
      
      const labels = tf.oneHot(
        tf.tensor1d(labelArray, 'int32'),
        10
      );

      console.log('Fashion MNIST dataset loaded successfully');
      return {
        data: images,
        labels,
        numClasses: 10,
        inputShape: [784]  // Flattened 28x28 image
      };
    } catch (error) {
      console.error('Error loading Fashion MNIST:', error);
      console.log('Falling back to random data generation');
      return this.generateRandomData('fashion-mnist');
    }
  }

  private async loadMNIST(): Promise<Dataset> {
    console.log('Loading MNIST dataset...');
    try {
      // Load images
      console.log('Fetching MNIST images...');
      const imagesResponse = await fetch('/datasets/mnist/train-images-idx3-ubyte');
      if (!imagesResponse.ok) {
        throw new Error(`Failed to load MNIST images: ${imagesResponse.statusText}`);
      }
      console.log('Successfully fetched MNIST images');
      const imageBuffer = await imagesResponse.arrayBuffer();
      const imageData = new Uint8Array(imageBuffer);
      // Parse image data
      const magicNumber = new DataView(imageBuffer).getInt32(0, false);
      if (magicNumber !== 2051) {
        throw new Error(`Invalid magic number for MNIST images: ${magicNumber}`);
      }
      const numImages = new DataView(imageBuffer).getInt32(4, false);
      const numRows = new DataView(imageBuffer).getInt32(8, false);
      const numCols = new DataView(imageBuffer).getInt32(12, false);
      console.log(`MNIST: ${numImages} images, ${numRows}x${numCols} pixels`);
      if (numImages <= 0 || numRows !== 28 || numCols !== 28) {
        throw new Error(`Invalid MNIST dimensions: ${numImages} images, ${numRows}x${numCols} pixels`);
      }
      // Skip header (16 bytes)
      const imageArray = new Float32Array(numImages * numRows * numCols);
      for (let i = 0; i < numImages * numRows * numCols; i++) {
        imageArray[i] = imageData[i + 16] / 255.0;
      }
      // Reshape to [numImages, 784] for the neural network
      const images = tf.tensor2d(
        imageArray,
        [numImages, numRows * numCols]
      );
      // Load labels
      console.log('Fetching MNIST labels...');
      const labelsResponse = await fetch('/datasets/mnist/train-labels-idx1-ubyte');
      if (!labelsResponse.ok) {
        throw new Error(`Failed to load MNIST labels: ${labelsResponse.statusText}`);
      }
      console.log('Successfully fetched MNIST labels');
      const labelBuffer = await labelsResponse.arrayBuffer();
      const labelData = new Uint8Array(labelBuffer);
      // Parse label data
      const labelMagicNumber = new DataView(labelBuffer).getInt32(0, false);
      if (labelMagicNumber !== 2049) {
        throw new Error(`Invalid magic number for MNIST labels: ${labelMagicNumber}`);
      }
      const numLabels = new DataView(labelBuffer).getInt32(4, false);
      console.log(`MNIST: ${numLabels} labels`);
      if (numLabels !== numImages) {
        throw new Error(`Number of labels (${numLabels}) does not match number of images (${numImages})`);
      }
      // Skip header (8 bytes)
      const labelArray = new Int32Array(numLabels);
      for (let i = 0; i < numLabels; i++) {
        labelArray[i] = labelData[i + 8];
      }
      const labels = tf.oneHot(
        tf.tensor1d(labelArray, 'int32'),
        10
      );
      console.log('MNIST dataset loaded successfully');
      return {
        data: images,
        labels,
        numClasses: 10,
        inputShape: [784]  // Flattened 28x28 image
      };
    } catch (error) {
      console.error('Error loading MNIST:', error);
      console.log('Falling back to random data generation');
      return this.generateRandomData('mnist');
    }
  }

  private generateRandomData(datasetName: string): Dataset {
    console.log(`Generating random data for ${datasetName}`);
    const numExamples = 1000;
    
    if (datasetName.toLowerCase() === 'cifar10') {
      const data = tf.randomNormal([numExamples, 32, 32, 3]);
      const labels = tf.oneHot(
        tf.tensor1d(Array.from({ length: numExamples }, () => Math.floor(Math.random() * 10)), 'int32'),
        10
      );

      return {
        data,
        labels,
        numClasses: 10,
        inputShape: [32, 32, 3]
      };
    } else {
      const data = tf.randomNormal([numExamples, 784]);  // Flattened 28x28 image
      const labels = tf.oneHot(
        tf.tensor1d(Array.from({ length: numExamples }, () => Math.floor(Math.random() * 10)), 'int32'),
        10
      );

      return {
        data,
        labels,
        numClasses: 10,
        inputShape: [784]  // Flattened 28x28 image
      };
    }
  }

  public getDataset(): Dataset | null {
    return this.dataset;
  }

  public setDataset(dataset: Dataset) {
    this.dataset = dataset;
  }

  /**
   * Generates a unique distribution for each client
   * @param clientId The ID of the client
   * @returns An array representing the distribution of classes
   */
  private generateClientDistribution(clientId: string): number[] {
    const distribution = new Array(10).fill(0);
    
    // Create a simple hash of the client ID to get a consistent number
    const hash = this.hashString(clientId);
    console.log(`Client ID: ${clientId}, Hash: ${hash}`);
    
    // Assign higher probability to a subset of classes for each client
    const numPreferredClasses = Math.floor(Math.random() * 10);;  // Number of classes this client prefers
    const preferredClasses = new Set<number>();
    
    // Select preferred classes based on the hash
    while (preferredClasses.size < numPreferredClasses) {
      const classIdx = (hash + preferredClasses.size) % 10;
      preferredClasses.add(classIdx);
    }
    
    // Set probabilities for preferred classes
    const preferredProb = 0.7 / numPreferredClasses;  // 70% probability split among preferred classes
    const otherProb = 0.3 / (10 - numPreferredClasses);  // 30% probability split among other classes
    
    for (let i = 0; i < 10; i++) {
      distribution[i] = preferredClasses.has(i) ? preferredProb : otherProb;
    }
    
    return distribution;
  }

  /**
   * Simple string hash function
   * @param str The string to hash
   * @returns A number between 0 and 9
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash) % 10;  // Ensure we get a number between 0 and 9
  }

  /**
   * Distributes the dataset among clients in a non-IID manner
   * @param dataset The full dataset to distribute
   * @param clientId The ID of the client requesting data
   * @returns A subset of the dataset for the specified client
   */
  public async getClientDataset(dataset: Dataset, clientId: string): Promise<ClientDataset> {
    console.log(`Distributing data for client ${clientId}`);
    
    // Get the data and labels as arrays for easier manipulation
    const dataArray = await dataset.data.array() as any;
    const labelsArray = await dataset.labels.array() as number[][];
    
    // Create a distribution for this client
    const distribution = this.generateClientDistribution(clientId);
    console.log(`Client ${clientId} class distribution:`, distribution);
    
    // Calculate number of samples for this client
    const numSamples = Math.floor(
      Math.random() * (this.MAX_SAMPLES_PER_CLIENT - this.MIN_SAMPLES_PER_CLIENT) + 
      this.MIN_SAMPLES_PER_CLIENT
    );
    
    // Select samples based on the distribution
    const selectedIndices: number[] = [];
    const samplesPerClass = Math.floor(numSamples / dataset.numClasses);
    
    for (let classIdx = 0; classIdx < dataset.numClasses; classIdx++) {
      // Get all indices for this class
      const classIndices = labelsArray
        .map((label: number[], idx: number) => label[classIdx] === 1 ? idx : -1)
        .filter((idx: number) => idx !== -1);
      
      // Calculate how many samples to take for this class
      const numClassSamples = Math.floor(samplesPerClass * distribution[classIdx]);
      
      // Randomly select samples for this class
      for (let i = 0; i < numClassSamples && classIndices.length > 0; i++) {
        const randomIdx = Math.floor(Math.random() * classIndices.length);
        selectedIndices.push(classIndices[randomIdx]);
        classIndices.splice(randomIdx, 1);
      }
    }
    
    // Create tensors from selected samples based on input shape
    let clientData: tf.Tensor;
    if (dataset.inputShape.length === 3) {
      // CIFAR-10: 4D tensor [batch, height, width, channels]
      const selectedData = selectedIndices.map(idx => dataArray[idx]);
      clientData = tf.tensor4d(selectedData, [selectedIndices.length, dataset.inputShape[0], dataset.inputShape[1], dataset.inputShape[2]]);
    } else {
      // Fashion-MNIST: 2D tensor [batch, features]
      clientData = tf.tensor2d(
        selectedIndices.map(idx => dataArray[idx]),
        [selectedIndices.length, dataset.inputShape[0]]
      );
    }
    
    const clientLabels = tf.tensor2d(
      selectedIndices.map(idx => labelsArray[idx]),
      [selectedIndices.length, dataset.numClasses]
    );
    
    console.log(`Client ${clientId} received ${selectedIndices.length} samples`);
    
    return {
      data: clientData,
      labels: clientLabels,
      distribution
    };
  }
} 