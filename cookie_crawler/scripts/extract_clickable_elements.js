function element_is_hidden(e) {
    var is_hidden = true;
    var height = e.offsetHeight;
    var width = e.offsetWidth;
    if (height === undefined || width === undefined) {
        return true;
    }
    try {
        var cur = e;
        var tries = 3
        while (cur && tries > 0) {
            if (window.getComputedStyle(cur).getPropertyValue("opacity") === "0") {
              return true;
            }
            cur = cur.parentElement;
            tries -= 1
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

function get_clickable_elements(parent) {
  var elements = [];
  for (let element of parent.getElementsByTagName("*")) {
    if (
        !element_is_hidden(element)
        && ["DIV", "SPAN", "A", "BUTTON", "INPUT"].includes(element.tagName)
        && (
            element.tabIndex >= 0
            || element.getAttribute("role") === "button"
            || element.getAttribute('onclick') !== null
        )
    ) {
     elements.push(element);
    }
  }
  var filtered_elements = [];
  for (let element of elements) {
      var parent_found = false;
      for (let parent of elements) {
          if (element !== parent && parent.contains(element)) {
              parent_found = true;
          }
      }
      if (parent_found === false) {
          filtered_elements.push(element)
      }
  }
  return filtered_elements;
}

return get_clickable_elements(arguments[0]);
