export const discoverUIElements = () => {
    // Start with interactive elements
    const interactiveElements = [
      ...document.querySelectorAll('button, a, input, select, textarea, [role="button"]')
    ];
    
    // Add elements with data attributes that might be custom interactive components
    const dataElements = document.querySelectorAll('[data-id], [data-testid], [data-role]');
    
    // Combine and deduplicate
    const allElements = Array.from(new Set([...interactiveElements, ...dataElements]));
    
    // Extract metadata for each element
    return allElements.map(extractElementMetadata)
      .filter(elem => elem.isVisible); // Only include visible elements
  };
  
  // Extract detailed metadata from an element
  export const extractElementMetadata = (element) => {
    const rect = element.getBoundingClientRect();
    
    // Check if element is visible
    const isVisible = 
      rect.width > 0 && 
      rect.height > 0 && 
      rect.top < window.innerHeight && 
      rect.left < window.innerWidth &&
      window.getComputedStyle(element).display !== 'none' &&
      window.getComputedStyle(element).visibility !== 'hidden';
    
    // Get meaningful text content
    const text = element.innerText || element.textContent || '';
    const placeholder = element.placeholder || '';
    const ariaLabel = element.getAttribute('aria-label') || '';
    const title = element.getAttribute('title') || '';
    
    // Determine element type
    const tagName = element.tagName.toLowerCase();
    const role = element.getAttribute('role') || '';
    const type = element.getAttribute('type') || '';
    
    // Get data attributes
    const dataAttributes = {};
    Array.from(element.attributes)
      .filter(attr => attr.name.startsWith('data-'))
      .forEach(attr => {
        dataAttributes[attr.name] = attr.value;
      });
    
    // Determine if element is interactive
    const isInteractive = 
      tagName === 'button' || 
      tagName === 'a' || 
      tagName === 'input' || 
      tagName === 'select' || 
      role === 'button' || 
      element.onclick;
    
    return {
      id: element.id || element.getAttribute('data-id') || generateElementId(element),
      bounds: [rect.left, rect.top, rect.right, rect.bottom],
      tagName,
      type: type || role || tagName,
      text: [text, placeholder, ariaLabel, title].filter(Boolean).join(' ').trim(),
      isVisible,
      isInteractive,
      dataAttributes,
      path: getElementPath(element),
      semanticContext: extractSemanticContext(element)
    };
  };
  
  // Generate a unique ID for elements without ID
  export const generateElementId = (element) => {
    const text = element.innerText || element.textContent || '';
    const type = element.tagName.toLowerCase();
    const cleanText = text.trim().toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]/g, '');
    
    return cleanText ? `${type}-${cleanText}` : `${type}-${Math.random().toString(36).substr(2, 9)}`;
  };
  
  // Get DOM path to element for more stable identification
  export const getElementPath = (element) => {
    let path = [];
    let currentNode = element;
    
    while (currentNode && currentNode !== document.body) {
      let selector = currentNode.tagName.toLowerCase();
      
      if (currentNode.id) {
        selector += `#${currentNode.id}`;
      } else if (currentNode.className) {
        selector += `.${Array.from(currentNode.classList).join('.')}`;
      }
      
      path.unshift(selector);
      currentNode = currentNode.parentNode;
    }
    
    return path.join(' > ');
  };
  
  // Extract semantic context (nearby labels, headings, etc.)
  export const extractSemanticContext = (element) => {
    // Check for associated label
    let labelText = '';
    if (element.id) {
      const label = document.querySelector(`label[for="${element.id}"]`);
      if (label) {
        labelText = label.textContent.trim();
      }
    }
    
    // Check for parent fieldset/legend
    let fieldsetContext = '';
    let parent = element.parentNode;
    while (parent && parent !== document.body) {
      if (parent.tagName === 'FIELDSET') {
        const legend = parent.querySelector('legend');
        if (legend) {
          fieldsetContext = legend.textContent.trim();
        }
        break;
      }
      parent = parent.parentNode;
    }
    
    // Find nearby headings that might provide context
    let headingContext = '';
    const nearestHeading = findNearestHeading(element);
    if (nearestHeading) {
      headingContext = nearestHeading.textContent.trim();
    }
    
    return {
      label: labelText,
      fieldset: fieldsetContext,
      heading: headingContext
    };
  };
  
  // Find the nearest heading that might provide context for this element
  export const findNearestHeading = (element) => {
    let currentNode = element;
    
    // Look up the DOM tree for a section with a heading
    while (currentNode && currentNode !== document.body) {
      // Check if this section has a heading
      const headings = currentNode.querySelectorAll('h1, h2, h3, h4, h5, h6');
      if (headings.length > 0) {
        return headings[0]; // Return the first/highest level heading
      }
      
      currentNode = currentNode.parentNode;
    }
    
    return null;
  };