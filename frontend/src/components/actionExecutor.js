/**
 * Helper module for executing actions on UI elements
 */

// Generic action executor that can handle any action on any element
// In actionExecutor.js - Add detailed logging to executeAction function

export const executeAction = async (action, elementMetadata) => {
    console.log("Executing action:", action);
    
    // Find the element in DOM
    const element = findElementByMetadata(elementMetadata);
    
    if (!element) {
      // Try to infer the action from text for navigation
      if (action.text && (
          action.text.toLowerCase().includes("view") || 
          action.text.toLowerCase().includes("statement"))) {
        
        // Look for view buttons
        const viewButtons = document.querySelectorAll('[data-id^="btn-view-"]');
        if (viewButtons.length > 0) {
          console.log("Found view button as fallback");
          highlightElement(viewButtons[0]);
          await wait(300);
          viewButtons[0].click();
          return { success: true, message: "Clicked view button" };
        }
      }
      
      return { success: false, error: "Element not found" };
    }
    
    // Highlight the element
    highlightElement(element);
    
    // Scroll into view
    element.scrollIntoView({ behavior: "smooth", block: "center" });
    
    // Wait for scroll
    await wait(300);
    
    // Execute appropriate action
    try {
      // Click is the default action
      element.click();
      return { success: true };
    } catch (error) {
      console.error("Error executing action:", error);
      return { success: false, error: error.message };
    }
  };
// Simple function to perform basic actions
const performBasicAction = async (element, action) => {
  try {
    // Default to click for most elements
    element.click();
    return { success: true };
  } catch (error) {
    console.error("Error performing action:", error);
    return { success: false, error: error.message };
  }
};
  
  // Find element in DOM based on metadata
 // Improved element finding function
export const findElementByMetadata = (metadata) => {
    console.log("Looking for element with metadata:", metadata);
    
    // Try finding by ID first
    if (metadata.id) {
      console.log("Looking for element by ID:", metadata.id);
      const elementById = document.getElementById(metadata.id);
      if (elementById) {
        console.log("Found by ID");
        return elementById;
      }
      
      // Try by data-id
      const elementByDataId = document.querySelector(`[data-id="${metadata.id}"]`);
      if (elementByDataId) {
        console.log("Found by data-id");
        return elementByDataId;
      }
    }
    
    // Try by text content for buttons, links, etc.
    if (metadata.text) {
      console.log("Looking for element by text:", metadata.text);
      // Look for elements with exact text match
      const allElements = document.querySelectorAll('button, a, [role="button"], [data-id]');
      
      for (const element of allElements) {
        const elementText = element.innerText || element.textContent || '';
        if (elementText.trim() === metadata.text.trim()) {
          console.log("Found by exact text match");
          return element;
        }
      }
      
      // Try partial text match if exact match failed
      for (const element of allElements) {
        const elementText = element.innerText || element.textContent || '';
        if (elementText.toLowerCase().includes(metadata.text.toLowerCase())) {
          console.log("Found by partial text match");
          return element;
        }
      }
    }
    
    // Try by button or link text that contains "view" if looking for a view action
    if (metadata.text && metadata.text.toLowerCase().includes("view")) {
      console.log("Looking for view buttons");
      const viewButtons = Array.from(document.querySelectorAll('button, a'))
        .filter(el => (el.innerText || el.textContent || '').toLowerCase().includes('view'));
      
      if (viewButtons.length > 0) {
        console.log("Found view button:", viewButtons[0]);
        return viewButtons[0];
      }
    }
    
    console.log("Element not found");
    return null;
  };
  
  // Highlight element to provide visual feedback
  export const highlightElement = (element) => {
    const originalOutline = element.style.outline;
    const originalBoxShadow = element.style.boxShadow;
    
    element.style.outline = '2px solid rgba(0, 120, 255, 0.8)';
    element.style.boxShadow = '0 0 8px rgba(0, 120, 255, 0.5)';
    
    // Reset after animation
    setTimeout(() => {
      element.style.outline = originalOutline;
      element.style.boxShadow = originalBoxShadow;
    }, 1000);
  };
  
  // Action implementations
  export const performClickAction = (element) => {
    try {
      element.click();
      return { success: true };
    } catch (e) {
      console.error("Error clicking element:", e);
      return { success: false, error: e.message };
    }
  };
  
  export const performInputAction = (element, value) => {
    try {
      // Clear existing value
      element.value = '';
      
      // Set new value
      element.value = value || '';
      
      // Trigger input event
      element.dispatchEvent(new Event('input', { bubbles: true }));
      
      // Also trigger change event
      element.dispatchEvent(new Event('change', { bubbles: true }));
      
      return { success: true, value };
    } catch (e) {
      console.error("Error inputting value:", e);
      return { success: false, error: e.message };
    }
  };
  
  export const performSelectAction = (element, value) => {
    try {
      if (element.tagName !== 'SELECT') {
        throw new Error("Element is not a select");
      }
      
      let optionFound = false;
      
      // If we have a value, try to find matching option
      if (value) {
        // Try exact match
        for (let i = 0; i < element.options.length; i++) {
          if (element.options[i].value === value ||
              element.options[i].text.toLowerCase().includes(value.toLowerCase())) {
            element.selectedIndex = i;
            optionFound = true;
            break;
          }
        }
      }
      
      // If no match found, select first non-disabled option
      if (!optionFound) {
        for (let i = 0; i < element.options.length; i++) {
          if (!element.options[i].disabled) {
            element.selectedIndex = i;
            break;
          }
        }
      }
      
      // Trigger change event
      element.dispatchEvent(new Event('change', { bubbles: true }));
      
      return { success: true };
    } catch (e) {
      console.error("Error selecting option:", e);
      return { success: false, error: e.message };
    }
  };
  
  export const performNavigationAction = (element) => {
    try {
      // If element is a link, extract the href and use it
      if (element.tagName === 'A' && element.href) {
        const href = element.href;
        
        // Check if it's an internal link
        if (href.startsWith(window.location.origin) || href.startsWith('/')) {
          // Use history API for internal links to avoid page reload
          window.history.pushState({}, '', href);
          window.dispatchEvent(new Event('popstate'));
          return { success: true, href };
        } else {
          // External link, open in new tab
          window.open(href, '_blank');
          return { success: true, href, external: true };
        }
      }
      
      // If not a link, just click it
      return performClickAction(element);
    } catch (e) {
      console.error("Error navigating:", e);
      return { success: false, error: e.message };
    }
  };
  
  export const performToggleAction = (element) => {
    try {
      // For checkboxes and radios
      if (element.tagName === 'INPUT' && 
          (element.type === 'checkbox' || element.type === 'radio')) {
        element.checked = !element.checked;
        element.dispatchEvent(new Event('change', { bubbles: true }));
        return { success: true, checked: element.checked };
      }
      
      // For other elements, just click
      return performClickAction(element);
    } catch (e) {
      console.error("Error toggling element:", e);
      return { success: false, error: e.message };
    }
  };
  
  export const performScrollAction = (element, direction = 'down') => {
    try {
      const scrollAmount = direction === 'up' ? -100 : 100;
      
      // If element is scrollable, scroll it
      if (isElementScrollable(element)) {
        element.scrollBy({
          top: scrollAmount,
          behavior: 'smooth'
        });
      } else {
        // Otherwise scroll the page
        window.scrollBy({
          top: scrollAmount,
          behavior: 'smooth'
        });
      }
      
      return { success: true, direction };
    } catch (e) {
      console.error("Error scrolling:", e);
      return { success: false, error: e.message };
    }
  };
  
  // Helper to check if element is scrollable
  export const isElementScrollable = (element) => {
    const style = window.getComputedStyle(element);
    const overflowY = style.getPropertyValue('overflow-y');
    
    return (
      overflowY === 'scroll' ||
      overflowY === 'auto' ||
      element.scrollHeight > element.clientHeight
    );
  };
  
  // Utility function for waiting
  export const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));