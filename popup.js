// Detect Cyberbullying button
document.getElementById('detectButton').addEventListener('click', function () {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    var activeTab = tabs[0];
    chrome.tabs.sendMessage(activeTab.id, { action: 'detectCyberbullying' });
  });
});

// Placeholder for "Posts Manager" button
document.getElementById('postsManagerButton').addEventListener('click', function () {
  // Add functionality for "Posts Manager" here
  console.log('Posts Manager button clicked - Placeholder');
});

// Placeholder for "Assess Profiles" button
document.getElementById('assessProfilesButton').addEventListener('click', function () {
  // Add functionality for "Assess Profiles" here
  console.log('Assess Profiles button clicked - Placeholder');
});
