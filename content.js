

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log('Content Script Received Message:', request);
  
  if (request.action === 'detectCyberbullying') {
    var selectedText = window.getSelection().toString();
    console.log('Selected Text:', selectedText);
    
    // Send the selected text to your Streamlit app for classification
    chrome.runtime.sendMessage({ action: 'classifyText', text: selectedText });
  }
});
