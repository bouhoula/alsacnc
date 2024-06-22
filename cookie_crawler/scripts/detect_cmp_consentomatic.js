class Tools {
    static setBase(base) {
        Tools.base = base;
    }

    static getBase() {
        return Tools.base;
    }

    static findElement(options, parent = null, multiple = false) {
        let possibleTargets = null;

        if(options.selector.trim() === ":scope") {
            //Select current root
            if(parent != null) {
                possibleTargets = [parent];
            } else {
                if (Tools.base != null) {
                    possibleTargets = [Tools.base];
                } else {
                    possibleTargets = [document];
                }

            }

        } else {
            if (parent != null) {
                possibleTargets = Array.from(parent.querySelectorAll(options.selector));
            } else {
                if (Tools.base != null) {
                    possibleTargets = Array.from(Tools.base.querySelectorAll(options.selector));
                } else {
                    possibleTargets = Array.from(document.querySelectorAll(options.selector));
                }
            }
        }

        const clonedPossibleTargets = possibleTargets.slice();

        if (options.textFilter != null) {
            let filterMultipleSpacesRegex = /\s{2,}/gm;

            possibleTargets = possibleTargets.filter((possibleTarget) => {
                let textContent = possibleTarget.textContent.toLowerCase().replace(filterMultipleSpacesRegex, " ");

                if (Array.isArray(options.textFilter)) {
                    let foundText = false;

                    for (let text of options.textFilter) {
                        if (textContent.indexOf(text.toLowerCase().replace(filterMultipleSpacesRegex, " ")) !== -1) {
                            foundText = true;
                            break;
                        }
                    }

                    return foundText;
                } else if (options.textFilter != null) {
                    return textContent.indexOf(options.textFilter.toLowerCase()) !== -1;
                }
            });
        }

        if (options.styleFilters != null) {
            possibleTargets = possibleTargets.filter((possibleTarget) => {
                let styles = window.getComputedStyle(possibleTarget);

                let keep = true;

                for (let styleFilter of options.styleFilters) {
                    let option = styles[styleFilter.option]

                    if (styleFilter.negated) {
                        keep = keep && (option !== styleFilter.value);
                    } else {
                        keep = keep && (option === styleFilter.value);
                    }
                }

                return keep;
            });
        }

        if (options.displayFilter != null) {
            possibleTargets = possibleTargets.filter((possibleTarget) => {
                if(possibleTarget.matches(".ConsentOMatic-CMP-NoDetect")) {
                    return !options.displayFilter;
                }

                if(options.displayFilter) {
                    //We should be displayed
                    return possibleTarget.offsetHeight !== 0;
                } else {
                    //We should not be displayed
                    return possibleTarget.offsetHeight === 0;
                }
            });
        }

        if (options.iframeFilter != null) {
            possibleTargets = possibleTargets.filter((possibleTarget) => {
                if(options.iframeFilter) {
                    //We should be inside an iframe
                    return window.location !== window.parent.location;
                } else {
                    //We should not be inside an iframe
                    return window.location === window.parent.location;
                }
            });
        }

        if(options.childFilter != null) {
            possibleTargets = possibleTargets.filter((possibleTarget) => {
                let oldBase = Tools.base;
                Tools.setBase(possibleTarget);
                let childResults = Tools.find(options.childFilter);
                Tools.setBase(oldBase);
                return childResults.target != null;
            });
        }

        if (multiple) {
            return possibleTargets;
        } else {
            return possibleTargets[0];
        }
    }

    static find(options, multiple = false) {
        let results = [];
        if (options.parent != null) {
            let parent = Tools.findElement(options.parent, null, multiple);
            if (parent != null) {
                if (parent instanceof Array) {
                    parent.forEach((p) => {
                        let targets = Tools.findElement(options.target, p, multiple);
                        if (targets instanceof Array) {
                            targets.forEach((target) => {
                                results.push({
                                    "parent": p,
                                    "target": target
                                });
                            });
                        } else {
                            results.push({
                                "parent": p,
                                "target": targets
                            });
                        }
                    });

                    return results;
                } else {
                    let targets = Tools.findElement(options.target, parent, multiple);
                    if (targets instanceof Array) {
                        targets.forEach((target) => {
                            results.push({
                                "parent": parent,
                                "target": target
                            });
                        });
                    } else {
                        results.push({
                            "parent": parent,
                            "target": targets
                        });
                    }
                }
            }
        } else {
            let targets = Tools.findElement(options.target, null, multiple);
            if (targets instanceof Array) {
                targets.forEach((target) => {
                    results.push({
                        "parent": null,
                        "target": target
                    });
                });
            } else {
                results.push({
                    "parent": null,
                    "target": targets
                });
            }
        }

        if (results.length === 0) {
            results.push({
                "parent": null,
                "target": null
            });
        }

        if (multiple) {
            return results;
        } else {
            if (results.length !== 1) {
                console.warn("Multiple results found, even though multiple false", results);
            }

            return results[0];
        }
    }
}

Tools.base = null;

function find(option) {
    if (option.constructor.name === "Array") {
        return option.some(o => find(o))
    }
    if (option.target === undefined)
        return false
    let target = Tools.find(option).target
    return target !== undefined && target !== null
}


function detectCMP(rules, cmps) {
    let potentialCMPs = []
    let detectedCMPs = []
    for (let [cmp, rule] of Object.entries(rules)) {
        cmp = cmp.toLowerCase()
        cmp = cmps.find((element) => cmp.startsWith(element))
        if (cmp === undefined) {
            continue
        }
        let detector_matched = false
        for (let option of rule.detectors) {
            if (option !== undefined && find(option)) {
                detector_matched = true
                break
            }
        }
        let action_target_matched = false
        if (detector_matched) {
            for (let option of rule.actions_targets) {
                if (option !== undefined && find(option)) {
                    action_target_matched = true
                    break
                }
            }
        }

        if (detector_matched) {
            potentialCMPs = potentialCMPs.concat(cmp)
            if (action_target_matched) {
                detectedCMPs = detectedCMPs.concat(cmp)
            }
        }
    }
    potentialCMPs = [...new Set(potentialCMPs)]
    detectedCMPs = [...new Set(detectedCMPs)]
    console.log("Potential: ", potentialCMPs)
    console.log("Detected: ", detectedCMPs)
    if (detectedCMPs.length > 0) {
        return detectedCMPs;
    } else {
        return potentialCMPs;
    }
}

return detectCMP(arguments[0], arguments[1])

