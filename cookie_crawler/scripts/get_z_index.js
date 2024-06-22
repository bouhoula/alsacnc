window.getZIndex = function (e) {
    var z = document.defaultView.getComputedStyle(e).getPropertyValue('z-index');
    if (isNaN(z)) {return window.getZIndex(e.parentNode);}
    else {return z;}
};

return parseInt(window.getZIndex(arguments[0]))