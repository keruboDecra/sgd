chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  const highlightedTextElement = document.getElementById('highlightedText');
  const feedbackElement = document.getElementById('feedback');

  // Display highlighted text
  highlightedTextElement.textContent = message.text;

  // Optionally, you can update the feedbackElement with additional feedback from Streamlit
  // feedbackElement.textContent = message.feedback;
});
