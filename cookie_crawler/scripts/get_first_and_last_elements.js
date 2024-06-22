var body = document.getElementsByTagName('body')[0]
var children = Array.from(body.childNodes);
var elements = [];

for (var i = 0; i < children.length; i++) {
    var element = children[i];
    console.log(element.tagName);
    if (element.tagName === 'DIV' || element.tagName === 'FORM') {
        elements = elements.concat(element);
        break;
    }
}

children.reverse()

for (var i = 0; i < children.length; i++) {
    var element = children[i];
    if (element.tagName === 'DIV' || element.tagName === 'FORM') {
        if (!elements.includes(element)){
            elements = elements.concat(element);
        }
        break;
    }
}
return elements