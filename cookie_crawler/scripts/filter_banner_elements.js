var elements = arguments[0];
var selectors = arguments[1];

var empty_elements = new Set();

elements.forEach((e, i) => {
  if (e.innerText === "" || e.offsetWidth === 0 || e.offsetHeight === 0) {
    empty_elements.add(i);
  }
});

var filtered_elements = [];
var filtered_selectors = selectors === null ? null : [];

for (let [i_1, e_1] of elements.entries()) {
  if (empty_elements.has(i_1) || filtered_elements.includes(e_1)) {
      continue;
  }
  var s_1 = selectors === null ? null : selectors[i_1];
  var contained_in_another_element = false;
  for (let [i_2, e_2] of elements.entries()) {
      if (e_1 !== e_2 && !empty_elements.has(i_2) && e_2.contains(e_1)) {
          contained_in_another_element = true;
      }
  }
  if (contained_in_another_element === false) {
      filtered_elements=filtered_elements.concat(e_1);
      if (selectors !== null) {
          filtered_selectors = filtered_selectors.concat(s_1);
      }
  }
}
return [filtered_elements, filtered_selectors]