function element_is_hidden(e) {
    var is_hidden = true;
    var height = e.offsetHeight;
    var width = e.offsetWidth;
    if (height === undefined || width === undefined) {
        return true;
    }
    try {
        var cur = e;
        while (cur) {
            if (window.getComputedStyle(cur).getPropertyValue("opacity") === "0") {
              return true;
            }
            cur = cur.parentElement;
      }
    } catch(error) {
    }
    try {
        is_hidden = (
            window.getComputedStyle(e).display === "none"
            || window.getComputedStyle(e).visibility === "hidden"
            || height === 0
            || width === 0
        );
    } catch (error) {
    }
    e.childNodes.forEach(function (item) {
        is_hidden = is_hidden && element_is_hidden(item);
    });
    return is_hidden;
}

return element_is_hidden(arguments[0])
