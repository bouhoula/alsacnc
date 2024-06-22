let consents = [];
let detected_selectors = [];
var selectors = arguments[0]
for (let selector of selectors) {
  const tags = document.querySelector(selector);
  console.log(tags)
  if (tags) {
    consents = consents.concat(tags);
    detected_selectors = detected_selectors.concat(selector);
  }
}
return [consents, detected_selectors]