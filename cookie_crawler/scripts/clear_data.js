setTimeout(() => {
    document.querySelector('#clearSiteDataButton').click();
    setTimeout(() => {
        frame = window.frames[1].document;
        dialog = frame.querySelector('#SanitizeDialog > dialog:nth-child(1)');
        let checkbox1 = frame.querySelector('#historyFormDataAndDownloads');
        if (!checkbox1.checked) {
            checkbox1.click();
        }
        // checkbox 2 and 3 are checked by default
        let checkbox4 = frame.querySelector('#siteSettings');
        if (!checkbox4.checked) {
            checkbox4.click();
        }
        setTimeout(() => {
            dialog._buttons["accept"].click();
        }, 500);
    }, 500);
}, 500);