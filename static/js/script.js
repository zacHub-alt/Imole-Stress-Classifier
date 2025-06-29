// Handles language selection and dynamic survey question display
// Also dynamically loads the correct CSS theme for the selected language
function setLanguageTheme(lang) {
    // Remove any previous language-specific CSS
    const head = document.head;
    const existing = document.getElementById('lang-css');
    if (existing) head.removeChild(existing);
    let cssFile = '';
    if (lang === 'pidgin') cssFile= 'static/css/style_pidgin.css';
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