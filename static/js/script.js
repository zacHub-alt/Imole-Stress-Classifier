// Handles language selection and dynamic survey question display
// Also dynamically loads the correct CSS theme for the selected language
function setLanguageTheme(lang) {
    // Remove any previous language-specific CSS
    const head = document.head;
    const existing = document.getElementById('lang-css');
    if (existing) head.removeChild(existing);
    let cssFile = '';
    if (lang === 'yo') cssFile = '/static/css/style_yoruba.css';
    else if (lang === 'ig') cssFile = '/static/css/style_igbo.css';
    else if (lang === 'ha') cssFile = '/static/css/style_hausa.css';
    else cssFile = '/static/css/style.css';
    // Only add if not English (default is already loaded)
    if (lang !== 'en') {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = cssFile;
        link.id = 'lang-css';
        head.appendChild(link);
    }
}

// Progress bar for survey completion (now counts all selects in the form)
function updateProgressBar() {
    const selects = document.querySelectorAll('form select:not([name="language"])');
    let answered = 0;
    selects.forEach(sel => { if (sel.value) answered++; });
    const progress = document.getElementById('survey-progress');
    if (progress) {
        progress.value = answered;
        progress.max = selects.length;
    }
    const label = document.getElementById('progress-label');
    if (label) {
        label.textContent = `${answered} / ${selects.length}`;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const langSelect = document.getElementById('language');
    if (langSelect) {
        langSelect.addEventListener('change', function() {
            setLanguageTheme(this.value);
            updateProgressBar();
        });
        setLanguageTheme(langSelect.value);
    }
    // Add progress bar update on change
    if (langSelect) {
        langSelect.addEventListener('change', updateProgressBar);
    }
    document.querySelectorAll('form select:not([name="language"])').forEach(sel => {
        sel.addEventListener('change', updateProgressBar);
    });
    updateProgressBar();
});
