var elements_with_assigned_text = arguments[1];

function assign_text_to_element(element, selector) {
    if (selector === null) {
        selector = get_selector(element);
    }
    if (!elements_with_assigned_text.has(selector)) {
        if (element.innerText !== "") {
            elements_with_assigned_text.add(selector);
            return element.innerText;
        }
    }
    for (let neighbor)


}


return [assign_text_to_element(arguments[0], arguments[2]), elements_with_assigned_text]


