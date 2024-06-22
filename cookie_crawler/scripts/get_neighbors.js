var neighbors = arguments[0].parentNode.childNodes;
neighbors = Array.from(neighbors).filter(x => x !== arguments[0]);
var text = neighbors.map(x => x.innerText);
return [neighbors, text];
