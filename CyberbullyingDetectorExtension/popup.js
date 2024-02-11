// Handle messages from the background script
chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  const highlightedTextElement = document.getElementById('highlightedText');
  const feedbackElement = document.getElementById('feedback');

  // Display highlighted text
  highlightedTextElement.textContent = message.text;

  // Optionally, update the feedbackElement with additional feedback from Streamlit
  // feedbackElement.textContent = message.feedback;
});

// Add event listener for the Test Communication button
document.getElementById('testButton').addEventListener('click', function () {
  // Send a test message to the background script
  chrome.runtime.sendMessage({ test: 'Test message from popup' }, function (response) {
    if (chrome.runtime.lastError) {
      console.error(chrome.runtime.lastError);
    } else {
      console.log(response);
    }
  });
});
