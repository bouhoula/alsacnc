function elementIsHidden(e) {
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
        is_hidden = is_hidden && elementIsHidden(item);
    });
    return is_hidden;
}

function extractTextFromElement(e, exclude_links = false) {
    var text = [];
    if (elementIsHidden(e) || (exclude_links && (e.nodeName === "A" || e.nodeName === "BUTTON"))) {
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
            var link_text = extractTextFromElement(item, exclude_links);
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
            var list_items = extractTextFromElement(item, exclude_links);
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
            text = text.concat(extractTextFromElement(item, exclude_links));
        }
        prv_item_type = item.nodeName;
    });
    if (cur_text.trim() !== "") {
        text.push(cur_text.trim());
        cur_text = "";
    }
    return text.filter(x => {return x !== undefined;});
}


function getNeighborsRecursive(element, root) {
    const neighbors = [];

    let currentElement = element
    
    while (currentElement && currentElement !== root) {
        const parent = currentElement.parentElement;
        
        if (parent) {
            const siblingElements = Array.from(parent.children).filter(child => child !== currentElement);
            neighbors.push.apply(neighbors, siblingElements);
        }

        currentElement = parent;
    }
    return neighbors;
}

function shouldBeSplit(currentElement, button) {
    const currentRect = currentElement.getBoundingClientRect();
    const otherRect = button.getBoundingClientRect();

    return (currentElement.children.length > 0) && currentRect.top < otherRect.top && currentRect.bottom > otherRect.bottom;
}

function analyzeButtonPlacement(button, cookieNotice) {
	const neighbors = getNeighborsRecursive(button, cookieNotice)
	const updatedNeighbors = [];
    
    for (const element of neighbors) {
	    const tagName = element.tagName.toLowerCase()
	    if (tagName === 'script' || tagName === 'style' || extractTextFromElement(element).join('').length == 0 || elementIsHidden(element)) {
		    continue;
	    }
        if (shouldBeSplit(element, button)) {
            const children = Array.from(element.children);
            updatedNeighbors.push(...children);
        } else {
            updatedNeighbors.push(element);
        }
    }

	let lenAbove = 0, lenBelow = 0;
	for (const element of updatedNeighbors) {
		const currentRect = element.getBoundingClientRect();
	    const buttonRect = button.getBoundingClientRect();
	    if (currentRect.right <= buttonRect.left || buttonRect.right <= currentRect.left) {
		    continue;
	    }
	    elementLength = extractTextFromElement(element).join('').length
		if (currentRect.top < buttonRect.top) {
			lenAbove += elementLength;
		}
		if (currentRect.bottom > buttonRect.bottom) {
			lenBelow += elementLength;
		}

	}
	const noticeLength = extractTextFromElement(cookieNotice).join('').length
    return [lenBelow, lenAbove, noticeLength];
}

return analyzeButtonPlacement(arguments[0], arguments[1]);