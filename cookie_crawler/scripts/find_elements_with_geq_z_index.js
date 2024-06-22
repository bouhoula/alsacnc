window.getZIndex = function (e) {
    var z = document.defaultView.getComputedStyle(e).getPropertyValue('z-index');
    if (isNaN(z)) {return window.getZIndex(e.parentNode);}
    else {return z;}
};

var cur_z_index = arguments[0];

var elements = Array.from(document.querySelectorAll('body *'));
var filtered_elements = [];
Array.from(elements).forEach(function(element) {
  	try {
          if (parseInt(window.getZIndex(element)) >= parseInt(cur_z_index) && element.offsetWidth > 0 && element.offsetHeight > 0) {
              console.log(element.innerText, parseInt(window.getZIndex(element)), parseInt(cur_z_index), element.offsetWidth, element.offsetHeight);
              filtered_elements = filtered_elements.concat(element);
          }
    }
    catch (e) {}
});

var results = [];
for (let e_1 of filtered_elements) {
    var contained_in_another_element = false;
    for (let e_2 of filtered_elements) {
        if (e_1 !== e_2 && e_2.contains(e_1)) {
            contained_in_another_element = true;
        }
    }
    if (contained_in_another_element === false) {
        results=results.concat(e_1);
    }
}
return results