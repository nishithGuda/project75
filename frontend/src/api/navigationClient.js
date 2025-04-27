const API_BASE_URL = 'http://localhost:8000';

/**
 * Process a natural language query with visual data using the LLM
 * @param {string} query - Natural language query from the user
 * @param {string} screenshot - Base64 encoded screenshot
 * @param {object} screenMetadata - Current screen metadata
 * @returns {Promise<object>} - Response with suggested actions
 */
export const processQueryWithVisual = async (query, screenshot, screenMetadata) => {
  try {
    // Create the payload with all available information
    const payload = {
      query,
      screen_metadata: screenMetadata
    };
    
    // Only include screenshot if available (it can be large)
    if (screenshot) {
      payload.screenshot = screenshot;
    }
    
    console.log('Sending query to LLM backend:', query);
    
    // Send request to the LLM backend
    const response = await fetch(`${API_BASE_URL}/api/process-query-visual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();
    console.log('Received response from LLM:', result);
    
    return result;
  } catch (error) {
    console.error('Error processing visual query:', error);
    
    // Return error information
    return {
      success: false,
      error: `Failed to connect to LLM backend: ${error.message}`,
      actions: []
    };
  }
};

/**
 * Execute a selected UI action
 * @param {string} actionId - ID of the action to execute
 * @param {string} screenId - Current screen ID
 * @param {object} actionDetails - Additional action details
 * @returns {Promise<object>} - Response with execution result
 */
export const executeAction = async (actionId, screenId, actionDetails = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/execute-action`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        action_id: actionId,
        screen_id: screenId,
        details: actionDetails
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error executing action:', error);
    
    return {
      success: false,
      error: `Failed to execute action: ${error.message}`
    };
  }
};

/**
 * Send feedback about an action (for LLM learning)
 * @param {string} actionId - ID of the action 
 * @param {boolean} wasSuccessful - Whether the action was successful
 * @param {string} feedback - Optional user feedback
 * @returns {Promise<object>} - Response
 */
export const sendActionFeedback = async (actionId, wasSuccessful, feedback = '') => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/action-feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        action_id: actionId,
        successful: wasSuccessful,
        feedback
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending feedback:', error);
    return { success: false };
  }
};