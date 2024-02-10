document.addEventListener("mouseup", function () {
  const selectedText = window.getSelection().toString().trim();

  if (selectedText) {
    // Send the selected text to your Streamlit app
    chrome.runtime.sendMessage({ text: selectedText }, function (response) {
      // Handle the response from your Streamlit app
      console.log(response);
    });
  }
});
document.addEventListener('DOMContentLoaded', function () {
  // Get the button element
  var testButton = document.getElementById('testButton');

  // Add click event listener to the button
  testButton.addEventListener('click', function () {
    // Send a message to the background script
    chrome.runtime.sendMessage({ testMessage: 'Testing communication from popup' });
  });
});
