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

function extract_text_from_element(e, exclude_links = false) {
    var text = [];
    if (element_is_hidden(e) || (exclude_links && (e.nodeName === "A" || e.nodeName === "BUTTON"))) {
        return text;
    }
    var cur_text = "";
    var prv_item_type = "";
    var children = e.childNodes;
    children.forEach(function(item) {
        if (item.textContent.trim() === "" || item.nodeName === "#comment") {
            return;
        }
        if (item.nodeName === "BUTTON" && exclude_links === true) {
            return;
        } else if (item.nodeName === "A") {
            if (exclude_links === true) {
                return;
            }
            var link_text = extract_text_from_element(item, exclude_links);
            if (link_text.length > 1 || prv_item_type === "A") {
                if (cur_text.trim() !== "") {
                    text.push(cur_text.trim());
                    cur_text = "";
                }
                text = text.concat(link_text);
            } else if (link_text.length === 1) {
                cur_text += " " + link_text[0].trim();
            }
        } else if (["#text", "EM", "STRONG", "I", "MARK"].includes(item.nodeName)) {
            cur_text = cur_text + " " + item.textContent.trim();
        } else if (["UL", "OL"].includes(item.nodeName)) {
            var list_items = extract_text_from_element(item, exclude_links);
            if (cur_text.trim() !== "") {
                cur_text = cur_text.trim() + " ";
            }
            text = text.concat(Array.from(list_items).map(x => cur_text + x));
            cur_text = "";
        } else {
            if (cur_text.trim() !== "") {
                text.push(cur_text.trim());
                cur_text = "";
            }
            text = text.concat(extract_text_from_element(item, exclude_links));
        }
        prv_item_type = item.nodeName;
    });
    if (cur_text.trim() !== "") {
        text.push(cur_text.trim());
        cur_text = "";
    }
    return text.filter(x => {return x !== undefined;});
}


var text = extract_text_from_element(arguments[0], arguments[1]);
return text.join("\n");
