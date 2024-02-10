chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  const highlightedTextElement = document.getElementById('highlightedText');
  const feedbackElement = document.getElementById('feedback');

  // Display highlighted text
  highlightedTextElement.textContent = message.text;

  // Send highlighted text to your Streamlit app for feedback
  chrome.runtime.sendMessage({ text: message.text }, function (response) {
    // Display feedback in the popup
    feedbackElement.textContent = response.feedback;
  });
});
