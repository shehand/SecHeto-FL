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

import React, { useEffect, useState, useRef } from 'react';
import { FederatedClient } from './services/FederatedClient';
import './App.css';

interface TrainingProgress {
  status?: string;
  currentRound?: number;
  totalRounds?: number;
  message?: string;
  clientStatus?: string;
  submittedClients?: number;
  totalClients?: number;
}

function App() {
  const [client, setClient] = useState<FederatedClient | null>(null);
  const [status, setStatus] = useState<string>('Initializing...');
  const [prediction, setPrediction] = useState<number | null>(null);
  const [progress, setProgress] = useState<TrainingProgress>({});
  const [error, setError] = useState<string | null>(null);
  const initRef = useRef(false);

  useEffect(() => {
    let mounted = true;

    const init = async () => {
      // Prevent double initialization in strict mode
      if (initRef.current) return;
      initRef.current = true;

      try {
        const federatedClient = new FederatedClient();
        
        // Set up progress callback
        federatedClient.setProgressCallback((progress) => {
          if (mounted) {
            setProgress(progress);
            if (progress.status === 'no_round') {
              setStatus(`Waiting for clients (${progress.submittedClients || 0}/${progress.totalClients || 10})`);
            } else {
              setStatus(`Round ${progress.currentRound || 0}/${progress.totalRounds || 10}: ${progress.status}`);
            }
          }
        });

        // Register client
        await federatedClient.register();
        if (mounted) {
          setClient(federatedClient);
        }
      } catch (error: any) {
        if (!mounted) return;
        
        if (error.message === 'Server is full') {
          setError('Server has reached maximum number of clients. Please try again later.');
        } else {
          setError('Failed to initialize client. Please refresh the page to try again.');
        }
        setStatus('Failed to initialize');
        console.error(error);
      }
    };

    init();

    return () => {
      mounted = false;
      if (client) {
        client.disconnect();
        initRef.current = false;
      }
    };
  }, []); // Empty dependency array since we handle cleanup internally

  const initializeClient = async () => {
    // Remove the initialization logic since it's now in useEffect
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Federated Learning Demo</h1>
        
        {error ? (
          <div style={{ color: '#ff6b6b', marginBottom: '20px' }}>
            {error}
          </div>
        ) : (
          <>
            <div style={{ marginBottom: '20px' }}>
              <h2>Training Progress</h2>
              <p>Status: {status}</p>
              {progress.currentRound !== undefined && (
                <div style={{ marginTop: '10px' }}>
                  <div>Round: {progress.currentRound}/{progress.totalRounds}</div>
                  <div>Status: {progress.status}</div>
                  {progress.submittedClients !== undefined && progress.totalClients !== undefined && (
                    <div>Connected Clients: {progress.submittedClients}/{progress.totalClients}</div>
                  )}
                  {progress.clientStatus && <div>Client Status: {progress.clientStatus}</div>}
                </div>
              )}
            </div>

            {prediction !== null && (
              <div style={{ marginTop: '20px' }}>
                <h3>Last Prediction</h3>
                <p>Class {prediction}</p>
              </div>
            )}
          </>
        )}
      </header>
    </div>
  );
}

export default App;
