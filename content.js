chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === 'detectCyberbullying') {
    var selectedText = window.getSelection().toString();
    
    // Send the selected text to your Streamlit app for classification
    chrome.runtime.sendMessage({ action: 'classifyText', text: selectedText });
  }
});

