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

var elements = document.querySelectorAll("body *");
var visible_elements = [];
var return_links_only = arguments[0]
var element_to_exclude = arguments[1]
Array.from(elements).forEach(function(element) {
    if (
        !element_is_hidden(element)
        && (!return_links_only || element.nodeName === "A")
        && (element_to_exclude === null || !element_to_exclude.contains(element))
    ) {
        visible_elements = visible_elements.concat(element);
    }
});
return visible_elements;

