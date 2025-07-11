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

const puppeteer = require('puppeteer');

// CONFIGURATION
const NUM_CLIENTS = 3; // Number of simulated clients
const CLIENT_URL = 'http://localhost:3000'; // URL of your React FL client
const SERVER_URL = 'http://localhost:3001'; // URL of your FL server
const MAX_WAIT_TIME_MS = 300000; // Maximum time to wait (5 minutes as fallback)
const HEADLESS_MODE = true; // Set to false to see browser windows

// Anti-throttling configuration
const FOCUS_INTERVAL_MS = 10000; // Focus each page every 10 seconds
const KEEP_ALIVE_INTERVAL_MS = 5000; // Keep pages active every 5 seconds
const PROGRESS_CHECK_INTERVAL_MS = 5000; // Check FL progress every 5 seconds

// Function to check if server is running
async function checkServerStatus() {
  try {
    const response = await fetch(`${SERVER_URL}/progress`);
    return response.ok;
  } catch (error) {
    return false;
  }
}

// Function to check FL progress from server
async function checkFLProgress() {
  try {
    const response = await fetch(`${SERVER_URL}/progress`);
    const progress = await response.json();
    return progress;
  } catch (error) {
    console.error('Error checking FL progress:', error.message);
    return null;
  }
}

// Function to wait for FL completion
async function waitForFLCompletion() {
  const startTime = Date.now();
  
  while (Date.now() - startTime < MAX_WAIT_TIME_MS) {
    const progress = await checkFLProgress();
    
    if (progress) {
      const { rounds, clients } = progress;
      console.log(`📊 FL Progress: Round ${rounds.current}/${rounds.total}, Clients: ${clients.current}/${clients.required}`);
      
      // Check if all rounds are completed
      if (rounds.current >= rounds.total) {
        console.log('🎉 All FL rounds completed!');
        return true;
      }
      
      // Check if no clients are connected (something went wrong)
      if (clients.current === 0) {
        console.log('⚠️ No clients connected. FL may have failed.');
        return false;
      }
    }
    
    // Wait before next check
    await new Promise(resolve => setTimeout(resolve, PROGRESS_CHECK_INTERVAL_MS));
  }
  
  console.log('⏰ Maximum wait time reached. Closing simulation.');
  return false;
}

(async () => {
  // Check if server is running first
  console.log('🔍 Checking if FL server is running...');
  const serverRunning = await checkServerStatus();
  if (!serverRunning) {
    console.error('❌ FL server is not running. Please start the server first.');
    console.error(`   Expected server at: ${SERVER_URL}`);
    process.exit(1);
  }
  console.log('✅ FL server is running.');

  const browser = await puppeteer.launch({ headless: true }); // Set to false to see browser windows
  const pages = [];
  let focusIndex = 0;

  // Function to keep pages active and prevent throttling
  async function keepPagesActive() {
    for (let i = 0; i < pages.length; i++) {
      try {
        const page = pages[i];
        
        // Focus the page to prevent throttling
        await page.bringToFront();
        
        // Execute a small operation to keep the page active
        await page.evaluate(() => {
          // Keep the page active with a small operation
          if (window.performance && window.performance.now) {
            window.performance.now();
          }
          
          // Ensure any pending operations continue
          if (window.requestAnimationFrame) {
            window.requestAnimationFrame(() => {});
          }
        });
        
        // Small delay to prevent overwhelming the browser
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        console.error(`Error keeping page ${i + 1} active:`, error.message);
      }
    }
  }

  // Function to focus pages in rotation
  async function rotatePageFocus() {
    if (pages.length === 0) return;
    
    try {
      const page = pages[focusIndex % pages.length];
      await page.bringToFront();
      console.log(`🔄 Focused page ${focusIndex % pages.length + 1}`);
      focusIndex++;
    } catch (error) {
      console.error('Error rotating page focus:', error.message);
    }
  }

  console.log(`🚀 Launching ${NUM_CLIENTS} clients with anti-throttling measures...`);

  for (let i = 0; i < NUM_CLIENTS; i++) {
    const page = await browser.newPage();
    
    // Set viewport for consistent behavior
    await page.setViewport({ width: 1280, height: 720 });
    
    // Disable throttling for this page
    await page.evaluateOnNewDocument(() => {
      // Override throttling mechanisms
      if (window.requestIdleCallback) {
        window.requestIdleCallback = (callback) => {
          setTimeout(callback, 0);
        };
      }
      
      // Keep timers active
      const originalSetInterval = window.setInterval;
      window.setInterval = (callback, delay, ...args) => {
        return originalSetInterval(callback, Math.max(delay, 1), ...args);
      };
      
      // Prevent page from being throttled
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(registrations => {
          registrations.forEach(registration => registration.unregister());
        });
      }
    });

    // Navigate to client URL with better error handling
    try {
      await page.goto(CLIENT_URL, { 
        waitUntil: 'networkidle2',
        timeout: 30000 
      });
      
      // Ensure the page is fully loaded
      await page.waitForFunction(() => {
        return document.readyState === 'complete' && 
               document.querySelector('h1') !== null;
      }, { timeout: 30000 });
      
      pages.push(page);
      console.log(`✅ Client ${i + 1} launched successfully`);

    } catch (error) {
      console.error(`❌ Failed to launch client ${i + 1}:`, error.message);
      await page.close();
      continue;
    }

    // Log console output from each client
    page.on('console', msg => {
      const text = msg.text();
      
      // Filter out noisy logs
      if (text.includes('Warning') || text.includes('Deprecation') || text.includes('DevTools')) {
        return;
      }
      
      console.log(`Client ${i + 1}:`, text);
    });

    page.on('pageerror', err => {
      console.error(`Client ${i + 1} page error:`, err.message);
    });
    
    page.on('error', err => {
      console.error(`Client ${i + 1} error:`, err.message);
    });
  }

  if (pages.length === 0) {
    console.error('❌ No clients launched successfully. Exiting.');
    await browser.close();
    return;
  }

  console.log(`${pages.length} clients launched. Waiting for FL completion...`);
  console.log('🔄 Anti-throttling measures active...');

  // Start anti-throttling measures
  const keepAliveInterval = setInterval(keepPagesActive, KEEP_ALIVE_INTERVAL_MS);
  const focusInterval = setInterval(rotatePageFocus, FOCUS_INTERVAL_MS);

  // Main execution loop - wait for FL completion
  try {
    const completed = await waitForFLCompletion();
    
    if (completed) {
      console.log('✅ FL process completed successfully!');
    } else {
      console.log('❌ FL process did not complete within expected time.');
    }
  } catch (error) {
    console.error('Error during execution:', error);
  } finally {
    // Cleanup intervals
    clearInterval(keepAliveInterval);
    clearInterval(focusInterval);
    
    console.log('🧹 Cleaning up...');
    
    // Close all pages
    for (const page of pages) {
      try {
        await page.close();
      } catch (error) {
        console.error('Error closing page:', error.message);
      }
    }
    
    await browser.close();
    console.log('✅ Simulation complete. Browser closed.');
  }
})(); 