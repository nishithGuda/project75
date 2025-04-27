import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useBankData } from '../context/BankDataContext';
import html2canvas from 'html2canvas';
import { processQueryWithVisual } from '../api/navigationClient';

const NavigationAssistant = () => {
  const { accounts } = useBankData();
  const navigate = useNavigate();
  
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [screenshot, setScreenshot] = useState(null);
  const inputRef = useRef(null);
  const assistantRef = useRef(null);
  
  // Close assistant when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (assistantRef.current && !assistantRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // Handle keyboard shortcut (Ctrl+Space) to open assistant
  useEffect(() => {
    const handleKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.code === 'Space') {
        event.preventDefault();
        setIsOpen(true);
        setTimeout(() => {
          inputRef.current?.focus();
        }, 100);
      }
      
      // ESC to close
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);
  
  // Capture screenshot when assistant is opened
  useEffect(() => {
    if (isOpen) {
      captureScreenshot();
    }
  }, [isOpen]);
  
  
  const captureScreenshot = async () => {
    try {
      // First hide the assistant to avoid it appearing in the screenshot
      const assistantElement = assistantRef.current;
      if (assistantElement) {
        assistantElement.style.visibility = 'hidden';
      }
      
      // Wait longer for the UI to update
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // More forgiving options for html2canvas
      const canvas = await html2canvas(document.body, {
        useCORS: true,
        allowTaint: true, // Allow potentially tainted images
        scale: 0.3, // Even smaller scale for better performance
        logging: false,
        backgroundColor: null,
        ignoreElements: (element) => {
          // Ignore the assistant element itself
          return element === assistantRef.current;
        }
      });
      
      // Convert to data URL with lower quality
      const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
      setScreenshot(dataUrl);
      
      // Make the assistant visible again
      if (assistantElement) {
        assistantElement.style.visibility = 'visible';
      }
      
      return dataUrl;
    } catch (err) {
      console.error('Error capturing screenshot:', err);
      
      // Make sure assistant is visible again
      if (assistantRef.current) {
        assistantRef.current.style.visibility = 'visible';
      }
      
      // Continue with just metadata, without screenshot
      return null;
    }
  };

  // Get metadata about current screen elements
  const getCurrentScreenMetadata = () => {
    // Get current path to determine which screen we're on
    const path = window.location.pathname;
    
    // Create a map of all visible elements with data attributes
    const elements = [];
    const elementsWithDataId = document.querySelectorAll('[data-id]');
    
    elementsWithDataId.forEach(el => {
      // Check if the element is visible
      const rect = el.getBoundingClientRect();
      const isVisible = rect.width > 0 && rect.height > 0 && 
                        rect.top < window.innerHeight && 
                        rect.left < window.innerWidth;
      
      if (isVisible) {
        // Get element attributes
        const id = el.getAttribute('data-id');
        const type = el.getAttribute('data-type') || 'unknown';
        const label = el.getAttribute('data-label') || el.innerText || '';
        const isClickable = el.tagName === 'BUTTON' || 
                           el.tagName === 'A' || 
                           el.tagName === 'INPUT' ||
                           el.getAttribute('role') === 'button' ||
                           el.onclick !== null;
        
        // Get bounding box
        const bounds = [
          rect.left,
          rect.top,
          rect.right,
          rect.bottom
        ];
        
        elements.push({
          id,
          type,
          text: label,
          clickable: isClickable,
          bounds,
          tagName: el.tagName.toLowerCase()
        });
      }
    });
    
    // Add special handling for navigation elements (even if they don't have data attributes)
    const navLinks = document.querySelectorAll('nav a, nav button, .navbar a, .navbar button');
    navLinks.forEach(el => {
      // Check if the element is visible
      const rect = el.getBoundingClientRect();
      const isVisible = rect.width > 0 && rect.height > 0 && 
                        rect.top < window.innerHeight && 
                        rect.left < window.innerWidth;
      
      if (isVisible) {
        const text = el.innerText || '';
        const href = el.getAttribute('href') || '';
        const id = `nav-${text.toLowerCase().replace(/\s+/g, '-')}`;
        
        // Only add if not already in the elements array
        if (!elements.some(element => element.id === id)) {
          elements.push({
            id: id,
            type: 'navigation',
            text: text,
            clickable: true,
            bounds: [rect.left, rect.top, rect.right, rect.bottom],
            tagName: el.tagName.toLowerCase(),
            href: href
          });
        }
      }
    });
    
    // Additional context based on current page
    let screenId = 'unknown_screen';
    if (path === '/' || path === '/accounts') {
      screenId = 'accounts_screen';
    } else if (path === '/transactions') {
      screenId = 'transactions_screen';
    } else if (path === '/transfer') {
      screenId = 'transfer_screen';
    }
    
    return {
      screen_id: screenId,
      url: window.location.href,
      path: path,
      elements: elements
    };
  };
  
  const handleSubmit = async (e) => {
    e?.preventDefault();
    
    if (!query.trim()) return;
    
    try {
      setIsProcessing(true);
      setError(null);
      
      // Try to capture screenshot but continue even if it fails
      let screenshotDataUrl = null;
      try {
        screenshotDataUrl = await captureScreenshot();
      } catch (err) {
        console.error('Screenshot capture failed, continuing with metadata only');
      }
      
      // Get current screen metadata
      const screenMetadata = getCurrentScreenMetadata();
      
      // Process query using UI Navigation Assistant
      const response = await processQueryWithVisual(query, screenshotDataUrl, screenMetadata);
      
      setResult(response);
      
      // Auto-execute the highest confidence action if above threshold
      if (response.success && response.actions && response.actions.length > 0) {
        const topAction = response.actions[0];
        
        if (topAction.confidence >= 0.7) {
          executeAction(topAction, screenMetadata.screen_id);
        }
      }
      
      // Clear query after successful processing
      setQuery('');
    } catch (err) {
      console.error('Error processing query:', err);
      setError('Failed to process your request. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Execute an action
  const executeAction = (action, screenId) => {
    console.log('Executing action:', action);
    
    // Special handling for navigation intent - check both text and ID
    const lowerText = (action.text || '').toLowerCase();
    const actionId = action.id || '';
    
    // Handle navigation actions based on ID or text content
    if (action.type === 'navigation' || actionId.startsWith('nav-')) {
      // Navigation handling based on text matches
      if (lowerText.includes('transaction') || actionId.includes('transaction')) {
        navigate('/transactions');
        return;
      } else if (lowerText.includes('account') || actionId.includes('account') || lowerText.includes('home') || actionId.includes('home')) {
        navigate('/');
        return;
      } else if (lowerText.includes('transfer') || actionId.includes('transfer')) {
        navigate('/transfer');
        return;
      }
    }
    
    // Direct command handling - check query intent
    const queryLower = query.toLowerCase();
    if (queryLower.includes('view transaction') || queryLower.includes('see transaction') || 
        queryLower.includes('show transaction') || queryLower.includes('go to transaction')) {
      navigate('/transactions');
      return;
    } else if (queryLower.includes('transfer') || queryLower.includes('send money')) {
      navigate('/transfer');
      return;
    } else if (queryLower.includes('account') || queryLower.includes('overview') || 
               queryLower.includes('dashboard') || queryLower.includes('home')) {
      navigate('/');
      return;
    }
    
    // For element actions, find and interact with the element
    const element = document.querySelector(`[data-id="${action.id}"]`);
    if (element) {
      // Highlight the element before interacting
      const originalBg = element.style.backgroundColor;
      element.style.backgroundColor = 'rgba(59, 130, 246, 0.3)';
      element.style.transition = 'background-color 0.3s';
      
      // Scroll element into view if needed
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
      // Simulate interaction after a short delay
      setTimeout(() => {
        // Different interaction based on element type
        if (action.interaction === 'input' && element.tagName === 'INPUT') {
          // Focus and set value for input fields
          element.focus();
          element.value = action.value || '';
          // Trigger input event to notify React of the change
          const event = new Event('input', { bubbles: true });
          element.dispatchEvent(event);
        } else if (action.interaction === 'select' && (element.tagName === 'SELECT')) {
          // Handle dropdown selection
          element.value = action.value || '';
          // Trigger change event
          const event = new Event('change', { bubbles: true });
          element.dispatchEvent(event);
        } else {
          // Default to click for buttons and other elements
          element.click();
        }
        
        // Reset background after interaction
        setTimeout(() => {
          element.style.backgroundColor = originalBg;
        }, 300);
      }, 500);
    } else {
      console.log('Element not found:', action.id);
      
      // Fallback navigation if element is not found but has navigation text
      if (lowerText.includes('transaction')) {
        navigate('/transactions');
      } else if (lowerText.includes('transfer')) {
        navigate('/transfer');
      } else if (lowerText.includes('account') || lowerText.includes('home')) {
        navigate('/');
      }
    }
  };

  return (
    <>
      {/* Assistant Trigger Button */}
      <button
        onClick={() => {
          setIsOpen(true);
          setTimeout(() => inputRef.current?.focus(), 100);
        }}
        className="fixed bottom-4 right-4 bg-blue-600 text-white rounded-full p-3 shadow-lg hover:bg-blue-700 transition z-50"
        title="Open Navigation Assistant (Ctrl+Space)"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
      </button>
      
      {/* Assistant Dialog */}
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-25 flex items-center justify-center z-50">
          <div 
            ref={assistantRef}
            className="bg-white rounded-lg shadow-xl p-4 mx-4 w-full max-w-md"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Banking Assistant</h3>
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-500"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <form onSubmit={handleSubmit}>
              <div className="relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full border border-gray-300 rounded-full px-4 py-2 pl-10 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Tell me what you want to do..."
                  disabled={isProcessing}
                />
                <div className="absolute left-3 top-2.5 text-gray-400">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
              </div>
              
              {/* Hints */}
              {!result && !error && !isProcessing && (
                <div className="mt-3 text-sm text-gray-500">
                  <p className="mb-1">Try asking:</p>
                  <ul className="text-xs space-y-1">
                    <li className="hover:text-blue-600 cursor-pointer" onClick={() => setQuery("Show me my accounts")}>
                      "Show me my accounts"
                    </li>
                    <li className="hover:text-blue-600 cursor-pointer" onClick={() => setQuery("View my transactions")}>
                      "View my transactions"
                    </li>
                    <li className="hover:text-blue-600 cursor-pointer" onClick={() => setQuery("I want to transfer money")}>
                      "I want to transfer money"
                    </li>
                    <li className="hover:text-blue-600 cursor-pointer" onClick={() => setQuery("View my checking account statement")}>
                      "View my checking account statement"
                    </li>
                  </ul>
                </div>
              )}
              
              {/* Processing Indicator */}
              {isProcessing && (
                <div className="mt-3 flex justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                </div>
              )}
              
              {/* Results */}
              {result && (
                <div className="mt-3">
                  {result.success ? (
                    <div className="text-green-600 bg-green-50 p-2 rounded-md">
                      {result.actions.length > 0 ? (
                        <p>Executing: {result.actions[0].text}</p>
                      ) : (
                        <p>Command understood but no actions needed.</p>
                      )}
                    </div>
                  ) : (
                    <div className="text-red-600 bg-red-50 p-2 rounded-md">
                      <p>{result.error || "Couldn't process that request."}</p>
                    </div>
                  )}
                </div>
              )}
              
              {/* Error Message */}
              {error && (
                <div className="mt-3 text-red-600 bg-red-50 p-2 rounded-md">
                  <p>{error}</p>
                </div>
              )}
              
              <div className="mt-4 text-xs text-gray-400 flex items-center">
                <span className="mr-1">Press</span>
                <kbd className="px-1 bg-gray-100 border border-gray-300 rounded">Ctrl</kbd>
                <span className="mx-1">+</span>
                <kbd className="px-1 bg-gray-100 border border-gray-300 rounded">Space</kbd>
                <span className="ml-1">to open assistant</span>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default NavigationAssistant;