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
