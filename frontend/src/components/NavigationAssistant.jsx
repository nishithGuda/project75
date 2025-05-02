import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useBankData } from "../context/BankDataContext";
import html2canvas from "html2canvas";
import {
  processQueryWithVisual,
  sendActionFeedback,
} from "../api/navigationClient";

const NavigationAssistant = () => {
  const { accounts } = useBankData();
  const navigate = useNavigate();

  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [screenshot, setScreenshot] = useState(null);
  const [feedback, setFeedback] = useState(""); // For user feedback
  const inputRef = useRef(null);
  const assistantRef = useRef(null);

  // Close assistant when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        assistantRef.current &&
        !assistantRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  // Handle keyboard shortcut (Ctrl+Space) to open assistant
  useEffect(() => {
    const handleKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.code === "Space") {
        event.preventDefault();
        setIsOpen(true);
        setTimeout(() => {
          inputRef.current?.focus();
        }, 100);
      }

      // ESC to close
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
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
        assistantElement.style.visibility = "hidden";
      }

      // Wait longer for the UI to update
      await new Promise((resolve) => setTimeout(resolve, 100));

      // More forgiving options for html2canvas
      const canvas = await html2canvas(document.body, {
        useCORS: true,
        allowTaint: true, // Allow potentially tainted images
        scale: 0.3, // Even smaller scale for better performance
        logging: false,
        backgroundColor: null,
        onclone: (documentClone) => {
          // Remove problematic CSS properties
          const styleElements = documentClone.querySelectorAll("style");
          styleElements.forEach((style) => {
            if (style.textContent.includes("oklch")) {
              style.textContent = style.textContent.replace(
                /oklch\([^)]+\)/g,
                "rgb(200, 200, 200)"
              );
            }
          });
        },
        ignoreElements: (element) => {
          // Ignore the assistant element itself
          return element === assistantRef.current;
        },
      });

      // Convert to data URL with lower quality
      const dataUrl = canvas.toDataURL("image/jpeg", 0.5);
      setScreenshot(dataUrl);

      // Make the assistant visible again
      if (assistantElement) {
        assistantElement.style.visibility = "visible";
      }

      return dataUrl;
    } catch (err) {
      console.error("Error capturing screenshot:", err);

      // Make sure assistant is visible again
      if (assistantRef.current) {
        assistantRef.current.style.visibility = "visible";
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
    const elementsWithDataId = document.querySelectorAll("[data-id]");

    elementsWithDataId.forEach((el) => {
      // Check if the element is visible
      const rect = el.getBoundingClientRect();
      const isVisible =
        rect.width > 0 &&
        rect.height > 0 &&
        rect.top < window.innerHeight &&
        rect.left < window.innerWidth;

      if (isVisible) {
        // Get element attributes
        const id = el.getAttribute("data-id");
        const type = el.getAttribute("data-type") || "unknown";
        const label = el.getAttribute("data-label") || el.innerText || "";
        const isClickable =
          el.tagName === "BUTTON" ||
          el.tagName === "A" ||
          el.tagName === "INPUT" ||
          el.getAttribute("role") === "button" ||
          el.onclick !== null;

        // Get bounding box
        const bounds = [rect.left, rect.top, rect.right, rect.bottom];

        elements.push({
          id,
          type,
          text: label,
          clickable: isClickable,
          bounds,
          tagName: el.tagName.toLowerCase(),
        });
      }
    });

    // Add special handling for navigation elements (even if they don't have data attributes)
    const navLinks = document.querySelectorAll(
      "nav a, nav button, .navbar a, .navbar button"
    );
    navLinks.forEach((el) => {
      // Check if the element is visible
      const rect = el.getBoundingClientRect();
      const isVisible =
        rect.width > 0 &&
        rect.height > 0 &&
        rect.top < window.innerHeight &&
        rect.left < window.innerWidth;

      if (isVisible) {
        const text = el.innerText || "";
        const href = el.getAttribute("href") || "";
        const id = `nav-${text.toLowerCase().replace(/\s+/g, "-")}`;

        // Only add if not already in the elements array
        if (!elements.some((element) => element.id === id)) {
          elements.push({
            id: id,
            type: "navigation",
            text: text,
            clickable: true,
            bounds: [rect.left, rect.top, rect.right, rect.bottom],
            tagName: el.tagName.toLowerCase(),
            href: href,
          });
        }
      }
    });

    // Additional context based on current page
    let screenId = "unknown_screen";
    if (path === "/" || path === "/accounts") {
      screenId = "accounts_screen";
    } else if (path === "/transactions") {
      screenId = "transactions_screen";
    } else if (path === "/transfer") {
      screenId = "transfer_screen";
    }

    return {
      screen_id: screenId,
      url: window.location.href,
      path: path,
      elements: elements,
    };
  };

  const handleSubmit = async (e) => {
    e?.preventDefault();

    if (!query.trim()) return;

    try {
      setIsProcessing(true);
      setError(null);
      setResult(null); // Clear previous results

      // Try to capture screenshot but continue even if it fails
      let screenshotDataUrl = null;
      try {
        screenshotDataUrl = await captureScreenshot();
      } catch (err) {
        console.error(
          "Screenshot capture failed, continuing with metadata only"
        );
      }

      // Get current screen metadata
      const screenMetadata = getCurrentScreenMetadata();

      // Process query using UI Navigation Assistant
      const response = await processQueryWithVisual(
        query,
        screenshotDataUrl,
        screenMetadata
      );

      setResult(response);

      // Auto-execute the highest confidence action if above threshold
      if (response.success && response.actions && response.actions.length > 0) {
        const topAction = response.actions[0];

        if (topAction.confidence >= 0.7) {
          setTimeout(() => {
            executeAction(topAction, screenMetadata.screen_id);

            // Send feedback after a delay to allow the action to complete
            setTimeout(() => {
              sendActionFeedback(topAction.id, screenMetadata.screen_id, true);
            }, 2000);
          }, 1000); // Delay execution slightly to let user see the result
        }
      }

      // Clear query after successful processing
      setQuery("");
    } catch (err) {
      console.error("Error processing query:", err);
      setError("Failed to process your request. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle clicking on an action
  const handleActionClick = (action) => {
    // Get current screen metadata
    const screenMetadata = getCurrentScreenMetadata();

    // Execute the action
    executeAction(action, screenMetadata.screen_id);

    // Send positive feedback
    sendActionFeedback(action.id, screenMetadata.screen_id, true);

    // Show feedback
    setFeedback(`Executing: ${action.text || action.id}`);
    setTimeout(() => setFeedback(""), 2000);

    // Close the assistant
    setIsOpen(false);
  };

  // Execute an action
  const executeAction = (action, screenId) => {
    console.log("Executing action:", action);

    // Get action type from the backend response, or default to "click"
    const actionType = action.action_type || "click";

    // Handle based on action type
    switch (actionType) {
      case "input":
        handleInputAction(action);
        break;

      case "select":
        handleSelectAction(action);
        break;

      case "view":
        handleViewAction(action);
        break;

      case "transfer":
        navigate("/transfer");
        break;

      case "navigate":
        handleNavigationAction(action);
        break;

      case "filter":
        handleFilterAction(action);
        break;

      case "click":
      default:
        handleClickAction(action);
        break;
    }
  };

  // Handle input actions (for text fields)
  const handleInputAction = (action) => {
    const element = document.querySelector(`[data-id="${action.id}"]`);
    if (!element) return;

    // Only process if it's an input element
    if (element.tagName === "INPUT" || element.tagName === "TEXTAREA") {
      // Get the value to input (if provided)
      const value = action.action_parameters?.value || "";

      // Highlight the element
      highlightElement(element);

      // Scroll into view
      element.scrollIntoView({ behavior: "smooth", block: "center" });

      // Focus and set value
      setTimeout(() => {
        element.focus();
        element.value = value;

        // Trigger input event
        const event = new Event("input", { bubbles: true });
        element.dispatchEvent(event);
      }, 500);
    } else {
      // Fallback to click if not an input element
      handleClickAction(action);
    }
  };

  // Handle select actions (for dropdowns)
  const handleSelectAction = (action) => {
    const element = document.querySelector(`[data-id="${action.id}"]`);
    if (!element || element.tagName !== "SELECT") {
      handleClickAction(action);
      return;
    }

    // Get the value to select (if provided)
    const value = action.action_parameters?.value || "";

    // Highlight the element
    highlightElement(element);

    // Scroll into view
    element.scrollIntoView({ behavior: "smooth", block: "center" });

    // Set value and trigger change event
    setTimeout(() => {
      element.value = value;
      const event = new Event("change", { bubbles: true });
      element.dispatchEvent(event);
    }, 500);
  };

  // Handle view actions
  const handleViewAction = (action) => {
    const text = (action.text || "").toLowerCase();

    // Check if it's a view transactions request
    if (text.includes("transaction")) {
      navigate("/transactions");
      return;
    }

    // Check if it's a view account statement
    if (action.id?.startsWith("btn-view-")) {
      const element = document.querySelector(`[data-id="${action.id}"]`);
      if (element) {
        highlightAndClick(element);
      }
      return;
    }

    // Default to regular click as fallback
    handleClickAction(action);
  };

  // Handle navigation actions
  const handleNavigationAction = (action) => {
    const text = (action.text || "").toLowerCase();

    if (text.includes("transaction") || action.id?.includes("transaction")) {
      navigate("/transactions");
    } else if (text.includes("transfer") || action.id?.includes("transfer")) {
      navigate("/transfer");
    } else if (
      text.includes("account") ||
      action.id?.includes("account") ||
      text.includes("home") ||
      action.id?.includes("home")
    ) {
      navigate("/");
    } else {
      // Try clicking the element as fallback
      const element = document.querySelector(`[data-id="${action.id}"]`);
      if (element) {
        highlightAndClick(element);
      }
    }
  };

  // Handle filter actions
  const handleFilterAction = (action) => {
    const element = document.querySelector(`[data-id="${action.id}"]`);
    if (!element) return;

    // Handle different filter elements
    if (element.tagName === "SELECT") {
      handleSelectAction(action);
    } else if (action.id.includes("dropdown") || action.id.includes("filter")) {
      highlightAndClick(element);
    } else {
      handleClickAction(action);
    }
  };

  // Handle click actions
  const handleClickAction = (action) => {
    const element = document.querySelector(`[data-id="${action.id}"]`);
    if (!element) {
      // Try navigation based on text as fallback
      handleNavigationAction(action);
      return;
    }

    // Click the element
    highlightAndClick(element);
  };

  // Helper to highlight and click an element
  const highlightAndClick = (element) => {
    highlightElement(element);
    element.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => element.click(), 500);
  };

  // Helper to highlight an element
  const highlightElement = (element) => {
    const originalBg = element.style.backgroundColor;
    element.style.backgroundColor = "rgba(59, 130, 246, 0.3)";
    element.style.transition = "background-color 0.3s";
    setTimeout(() => (element.style.backgroundColor = originalBg), 800);
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
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
          />
        </svg>
      </button>

      {/* Feedback Message */}
      {feedback && (
        <div className="fixed bottom-20 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white px-4 py-2 rounded-full shadow-lg z-50">
          {feedback}
        </div>
      )}

      {/* Assistant Dialog */}
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-25 flex items-center justify-center z-50">
          <div
            ref={assistantRef}
            className="bg-white rounded-lg shadow-xl p-4 mx-4 w-full max-w-md"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">
                Banking Assistant
              </h3>
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-500"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
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
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>
              </div>

              {/* Hints */}
              {!result && !error && !isProcessing && (
                <div className="mt-3 text-sm text-gray-500">
                  <p className="mb-1">Try asking:</p>
                  <ul className="text-xs space-y-1">
                    <li
                      className="hover:text-blue-600 cursor-pointer"
                      onClick={() => setQuery("Show me my accounts")}
                    >
                      "Show me my accounts"
                    </li>
                    <li
                      className="hover:text-blue-600 cursor-pointer"
                      onClick={() => setQuery("View my transactions")}
                    >
                      "View my transactions"
                    </li>
                    <li
                      className="hover:text-blue-600 cursor-pointer"
                      onClick={() => setQuery("I want to transfer money")}
                    >
                      "I want to transfer money"
                    </li>
                    <li
                      className="hover:text-blue-600 cursor-pointer"
                      onClick={() =>
                        setQuery("View my checking account statement")
                      }
                    >
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
              {/* Results */}
              {result && (
                <div className="mt-3">
                  {result.success ? (
                    result.actions.length > 0 ? (
                      <div>
                        <div className="text-green-600 bg-green-50 p-2 rounded-md mb-2">
                          <p>Found these possible actions:</p>
                        </div>
                        <div className="space-y-2">
                          {result.actions.slice(0, 2).map((action, index) => (
                            <div
                              key={index}
                              className="p-2 border rounded-md cursor-pointer hover:bg-blue-50"
                              onClick={() => handleActionClick(action)}
                            >
                              <div className="flex justify-between">
                                <span className="font-medium">
                                  {action.text || action.id}
                                </span>
                                <span className="text-sm text-blue-600">
                                  {Math.round(action.confidence * 100)}% match
                                </span>
                              </div>
                              {action.bert_confidence &&
                                action.rag_confidence && (
                                  <div className="flex space-x-4 text-xs text-gray-500 mt-1">
                                    <span>
                                      BERT:{" "}
                                      {Math.round(action.bert_confidence * 100)}
                                      %
                                    </span>
                                    <span>
                                      RAG:{" "}
                                      {Math.round(action.rag_confidence * 100)}%
                                    </span>
                                  </div>
                                )}
                              {action.reasoning && (
                                <p className="text-xs text-gray-500 mt-1">
                                  {action.reasoning}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="text-orange-600 bg-orange-50 p-2 rounded-md">
                        <p>
                          Command understood but no matching elements found.
                        </p>
                      </div>
                    )
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
                <kbd className="px-1 bg-gray-100 border border-gray-300 rounded">
                  Ctrl
                </kbd>
                <span className="mx-1">+</span>
                <kbd className="px-1 bg-gray-100 border border-gray-300 rounded">
                  Space
                </kbd>
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
