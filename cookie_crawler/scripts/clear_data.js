document.querySelector('#clearSiteDataButton').click();
setTimeout(() => {
    dialog = window.frames[1].document.querySelector(
        '#ClearSiteDataDialog > dialog:nth-child(1)'
    );
    // checkbox = window.frames[1].document.querySelector(
    //     '#clearCache'
    // )
    // checkbox.click();
    // setTimeout(() => {dialog._buttons["accept"].click();}, 1000);
    dialog._buttons["accept"].click();
}, 2000);
