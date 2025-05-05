import html2canvas from "html2canvas";
const API_BASE_URL = 'http://localhost:8000';

/**
 * Process a navigation query with UI context
 * @param {string} query - Natural language user query
 * @param {object} uiContext - Information about current UI state
 * @returns {Promise<object>} - Response with suggested actions
 */
export const processNavigationQuery = async (query, screenMetadata) => {
  try {
    const payload = {
      query,
      screen_metadata: screenMetadata
    };
    
    // Send to backend
    const response = await fetch(`${API_BASE_URL}/process-query-visual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error processing query:', error);
    
    return {
      success: false,
      error: `Navigation error: ${error.message}`,
      actions: []
    };
  }
};

/**
 * Send feedback about an action (for learning)
 * @param {string} actionId - ID of the action
 * @param {string} elementId - ID of the element
 * @param {string} screenContext - Context identifier for the screen
 * @param {boolean} wasSuccessful - Whether the action was successful
 * @param {object} additionalData - Any additional feedback data
 * @returns {Promise<object>} - Response
 */
export const sendActionFeedback = async (element_id, screen_id, wasSuccessful, action_type = null) => {
  try {
    // Make sure element_id has a value
    if (!element_id) {
      console.warn("Missing element_id in feedback, using fallback");
      element_id = "unknown-element";
    }
    
    // Log what we're sending for debugging
    console.log("Sending feedback:", {
      element_id,
      screen_id,
      success: wasSuccessful,
      action_type
    });
    
    const response = await fetch(`${API_BASE_URL}/action-feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        element_id,
        screen_id,
        success: wasSuccessful,
        action_type
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending feedback:', error);
    return { success: false };
  }
};

/**
 * Get metrics about the navigation system's performance
 * @returns {Promise<object>} - Response with metrics
 */
export const getNavigationMetrics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting metrics:', error);
    return { success: false };
  }
};

/**
 * Capture a screenshot of the current page
 * @returns {Promise<string|null>} - Base64-encoded screenshot or null if failed
 */
export const captureScreenshot = async () => {
  try {
    // Import html2canvas dynamically
    const html2canvas = await import('html2canvas');
    
    // Capture the screenshot
    const canvas = await html2canvas.default(document.body, {
      useCORS: true,
      allowTaint: true,
      scale: 0.25,
      logging: false,
      backgroundColor: null,
      width: window.innerWidth,
      height: window.innerHeight,
      x: 0,
      y: 0,
      onclone: (documentClone) => {
        // Remove problematic CSS properties
        const styleElements = documentClone.querySelectorAll('style');
        styleElements.forEach((style) => {
          if (style.textContent.includes('oklch')) {
            style.textContent = style.textContent.replace(
              /oklch\([^)]+\)/g,
              'rgb(200, 200, 200)'
            );
          }
        });
        
        // Also remove problematic elements
        const problematicElements = documentClone.querySelectorAll('iframe, canvas, video');
        problematicElements.forEach(el => {
          el.remove();
        });
      },
      ignoreElements: (element) => {
        // Ignore problematic elements
        return element.tagName === 'IFRAME' || 
               element.tagName === 'CANVAS';
      },
    });

    // Convert to data URL with lower quality
    const dataUrl = canvas.toDataURL('image/jpeg', 0.3);
    return dataUrl;
  } catch (error) {
    console.error('Error capturing screenshot:', error);
    return null;
  }
};