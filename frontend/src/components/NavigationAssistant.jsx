// NavigationAssistant.jsx
import React, { useState, useEffect, useRef } from "react";
import {
  processNavigationQuery,
  sendActionFeedback,
  captureScreenshot,
} from "../api/navigationClient";
import { discoverUIElements } from "./elementDiscovery";
import { executeAction, wait, highlightElement } from "./actionExecutor";

const NavigationAssistant = () => {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [feedback, setFeedback] = useState("");

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

  // Handle form submission
  const handleSubmit = async (e) => {
    e?.preventDefault();

    if (!query.trim()) return;

    try {
      setIsProcessing(true);
      setError(null);
      setResult(null);

      // Capture UI context
      const screenshot = await captureScreenshot();

      // Discover UI elements
      const elements = discoverUIElements();

      // Build UI context
      const uiContext = {
        screen_id: window.location.pathname,
        url: window.location.href,
        title: document.title,
        elements,
        timestamp: new Date().toISOString(),
      };

      // Process query
      const response = await processNavigationQuery(query, uiContext);
      setResult(response);

      // Auto-execute if high confidence
      if (response.success && response.actions && response.actions.length > 0) {
        const topAction = response.actions[0];

        if (topAction.confidence >= 0.7) {
          setTimeout(() => {
            executeActionAndSendFeedback(topAction, uiContext.screen_id);
          }, 800);
        }
      }

      // Clear query
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
    if (!action) {
      setFeedback("Error: No action selected");
      setTimeout(() => setFeedback(""), 2000);
      return;
    }

    // Execute action and send feedback
    executeActionAndSendFeedback(action, window.location.pathname);

    // Close the assistant
    setIsOpen(false);
  };
  // Execute action and send feedback
  const executeActionAndSendFeedback = async (action, screenId) => {
    try {
      if (!action) {
        setFeedback("Error: No action selected");
        setTimeout(() => setFeedback(""), 2000);
        return;
      }

      // Create an element object from the action properties
      // This matches what your actionExecutor expects
      const elementData = {
        id: action.id,
        tagName: action.type,
        text: action.text,
        bounds: action.bounds,
        type: action.type,
      };

      // Execute the action
      const result = await executeAction(action, elementData);

      // Show feedback
      setFeedback(
        result.success
          ? `Executed: ${action.text || action.id}`
          : `Failed: ${result.error}`
      );

      // Clear feedback after 2 seconds
      setTimeout(() => setFeedback(""), 2000);

      // Send feedback to server with proper ID handling
      await sendActionFeedback(
        action.id, // Element ID
        screenId, // Screen ID
        result.success, // Whether the action was successful
        action.action_type || "click" // Action type
      );
    } catch (error) {
      console.error("Error executing action:", error);
      setFeedback(`Error: ${error.message}`);
      setTimeout(() => setFeedback(""), 2000);
    }
  };

  // Show suggested commands based on current page
  const getSuggestedCommands = () => {
    const path = window.location.pathname;

    // Default suggestions
    const defaultSuggestions = [
      "Help me navigate this page",
      "What can I do here?",
      "Show me the main features",
    ];

    // Path-specific suggestions
    const pathSuggestions = {
      "/": [
        "Show me my account balance",
        "View my recent transactions",
        "I want to transfer money",
      ],
      "/transactions": [
        "Filter by this month",
        "Show deposits only",
        "Sort by amount",
      ],
      "/transfer": [
        "Transfer $100 to my savings account",
        "Move money between accounts",
        "Schedule a future payment",
      ],
    };

    return pathSuggestions[path] || defaultSuggestions;
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
                Navigation Assistant
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
                    {getSuggestedCommands().map((suggestion, index) => (
                      <li
                        key={index}
                        className="hover:text-blue-600 cursor-pointer"
                        onClick={() => setQuery(suggestion)}
                      >
                        "{suggestion}"
                      </li>
                    ))}
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
                    result.actions.length > 0 ? (
                      <div>
                        <div className="text-green-600 bg-green-50 p-2 rounded-md mb-2">
                          <p>Found these possible actions:</p>
                        </div>
                        <div className="space-y-2">
                          {/* Show more recommendations (up to 4 instead of 3) */}
                          {result.actions.slice(0, 4).map((action, index) => (
                            <div
                              key={index}
                              className={`p-2 border rounded-md cursor-pointer hover:bg-blue-50 ${
                                index === 0 ? "border-blue-500" : ""
                              }`}
                              onClick={() => handleActionClick(action)}
                            >
                              <div className="flex justify-between">
                                <span className="font-medium">
                                  {action.description ||
                                    action.text ||
                                    action.id}
                                </span>
                                <span className="text-sm text-blue-600">
                                  {Math.round(action.confidence * 100)}% match
                                </span>
                              </div>
                              <div className="flex flex-wrap space-x-2 mt-1 text-xs text-gray-500">
                                {action.diversity_bonus > 0 && (
                                  <span className="bg-blue-100 text-blue-800 px-1 rounded">
                                    New
                                  </span>
                                )}
                                {action.action_type && (
                                  <span>{action.action_type}</span>
                                )}
                              </div>
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
