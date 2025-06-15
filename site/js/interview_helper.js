function switchTab(tabId) {
    // Remove 'active' class from all tab buttons
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));

    // Remove 'active' class from all content sections
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    // Add 'active' class to the selected tab button
    const button = Array.from(document.querySelectorAll('.tab')).find(btn => btn.textContent.includes(tabIdToEmoji(tabId)));
    if (button) button.classList.add('active');

    // Add 'active' class to the selected content section
    const selectedContent = document.getElementById(tabId);
    if (selectedContent) selectedContent.classList.add('active');
}

function tabIdToEmoji(tabId) {
    const map = {
        practice: '🎯',
        dashboard: '📊',
        analytics: '📈',
        questions: '❓',
        settings: '⚙️'
    };
    return map[tabId] || '';
}