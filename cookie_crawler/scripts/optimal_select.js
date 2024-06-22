// Adapted from https://github.com/autarc/optimal-select
/**
 * Modify the context based on the environment
 *
 * @param  {HTMLELement} element - [description]
 * @param  {Object}      options - [description]
 * @return {boolean}             - [description]
 */
function adapt (element, options) {

  // detect environment setup
  if (global.document) {
    return false
  } else {
    global.document = options.context || (() => {
      var root = element
      while (root.parent) {
        root = root.parent
      }
      return root
    })()
  }

  // https://github.com/fb55/domhandler/blob/master/index.js#L75
  const ElementPrototype = Object.getPrototypeOf(global.document)

  // alternative descriptor to access elements with filtering invalid elements (e.g. textnodes)
  if (!Object.getOwnPropertyDescriptor(ElementPrototype, 'childTags')) {
    Object.defineProperty(ElementPrototype, 'childTags', {
      enumerable: true,
      get () {
        return this.children.filter((node) => {
          // https://github.com/fb55/domelementtype/blob/master/index.js#L12
          return node.type === 'tag' || node.type === 'script' || node.type === 'style'
        })
      }
    })
  }

  if (!Object.getOwnPropertyDescriptor(ElementPrototype, 'attributes')) {
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/attributes
    // https://developer.mozilla.org/en-US/docs/Web/API/NamedNodeMap
    Object.defineProperty(ElementPrototype, 'attributes', {
      enumerable: true,
      get () {
        const { attribs } = this
        const attributesNames = Object.keys(attribs)
        const NamedNodeMap = attributesNames.reduce((attributes, attributeName, index) => {
          attributes[index] = {
            name: attributeName,
            value: attribs[attributeName]
          }
          return attributes
        }, { })
        Object.defineProperty(NamedNodeMap, 'length', {
          enumerable: false,
          configurable: false,
          value: attributesNames.length
        })
        return NamedNodeMap
      }
    })
  }

  if (!ElementPrototype.getAttribute) {
    // https://docs.webplatform.org/wiki/dom/Element/getAttribute
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttribute
    ElementPrototype.getAttribute = function (name) {
      return this.attribs[name] || null
    }
  }

  if (!ElementPrototype.getElementsByTagName) {
    // https://docs.webplatform.org/wiki/dom/Document/getElementsByTagName
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/getElementsByTagName
    ElementPrototype.getElementsByTagName = function (tagName) {
      const HTMLCollection = []
      traverseDescendants(this.childTags, (descendant) => {
        if (descendant.name === tagName || tagName === '*') {
          HTMLCollection.push(descendant)
        }
      })
      return HTMLCollection
    }
  }

  if (!ElementPrototype.getElementsByClassName) {
    // https://docs.webplatform.org/wiki/dom/Document/getElementsByClassName
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/getElementsByClassName
    ElementPrototype.getElementsByClassName = function (className) {
      const names = className.trim().replace(/\s+/g, ' ').split(' ')
      const HTMLCollection = []
      traverseDescendants([this], (descendant) => {
        const descendantClassName = descendant.attribs.class
        if (descendantClassName && names.every((name) => descendantClassName.indexOf(name) > -1)) {
          HTMLCollection.push(descendant)
        }
      })
      return HTMLCollection
    }
  }

  if (!ElementPrototype.querySelectorAll) {
    // https://docs.webplatform.org/wiki/css/selectors_api/querySelectorAll
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/querySelectorAll
    ElementPrototype.querySelectorAll = function (selectors) {
      selectors = selectors.replace(/(>)(\S)/g, '$1 $2').trim() // add space for '>' selector

      // using right to left execution => https://github.com/fb55/css-select#how-does-it-work
      const instructions = getInstructions(selectors)
      const discover = instructions.shift()

      const total = instructions.length
      return discover(this).filter((node) => {
        var step = 0
        while (step < total) {
          node = instructions[step](node, this)
          if (!node) { // hierarchy doesn't match
            return false
          }
          step += 1
        }
        return true
      })
    }
  }

  if (!ElementPrototype.contains) {
    // https://developer.mozilla.org/en-US/docs/Web/API/Node/contains
    ElementPrototype.contains = function (element) {
      var inclusive = false
      traverseDescendants([this], (descendant, done) => {
        if (descendant === element) {
          inclusive = true
          done()
        }
      })
      return inclusive
    }
  }

  return true
}

/**
 * Retrieve transformation steps
 *
 * @param  {Array.<string>}   selectors - [description]
 * @return {Array.<Function>}           - [description]
 */
function getInstructions (selectors) {
  return selectors.split(' ').reverse().map((selector, step) => {
    const discover = step === 0
    const [type, pseudo] = selector.split(':')

    var validate = null
    var instruction = null

    switch (true) {

      // child: '>'
      case />/.test(type):
        instruction = function checkParent (node) {
          return (validate) => validate(node.parent) && node.parent
        }
        break

      // class: '.'
      case /^\./.test(type):
        const names = type.substr(1).split('.')
        validate = (node) => {
          const nodeClassName = node.attribs.class
          return nodeClassName && names.every((name) => nodeClassName.indexOf(name) > -1)
        }
        instruction = function checkClass (node, root) {
          if (discover) {
            return node.getElementsByClassName(names.join(' '))
          }
          return (typeof node === 'function') ? node(validate) : getAncestor(node, root, validate)
        }
        break

      // attribute: '[key="value"]'
      case /^\[/.test(type):
        const [attributeKey, attributeValue] = type.replace(/\[|\]|"/g, '').split('=')
        validate = (node) => {
          const hasAttribute = Object.keys(node.attribs).indexOf(attributeKey) > -1
          if (hasAttribute) { // regard optional attributeValue
            if (!attributeValue || (node.attribs[attributeKey] === attributeValue)) {
              return true
            }
          }
          return false
        }
        instruction = function checkAttribute (node, root) {
          if (discover) {
            const NodeList = []
            traverseDescendants([node], (descendant) => {
              if (validate(descendant)) {
                NodeList.push(descendant)
              }
            })
            return NodeList
          }
          return (typeof node === 'function') ? node(validate) : getAncestor(node, root, validate)
        }
        break

      // id: '#'
      case /^#/.test(type):
        const id = type.substr(1)
        validate = (node) => {
          return node.attribs.id === id
        }
        instruction = function checkId (node, root) {
          if (discover) {
            const NodeList = []
            traverseDescendants([node], (descendant, done) => {
              if (validate(descendant)) {
                NodeList.push(descendant)
                done()
              }
            })
            return NodeList
          }
          return (typeof node === 'function') ? node(validate) : getAncestor(node, root, validate)
        }
        break

      // universal: '*'
      case /\*/.test(type):
        validate = (node) => true
        instruction = function checkUniversal (node, root) {
          if (discover) {
            const NodeList = []
            traverseDescendants([node], (descendant) => NodeList.push(descendant))
            return NodeList
          }
          return (typeof node === 'function') ? node(validate) : getAncestor(node, root, validate)
        }
        break

      // tag: '...'
      default:
        validate = (node) => {
          return node.name === type
        }
        instruction = function checkTag (node, root) {
          if (discover) {
            const NodeList = []
            traverseDescendants([node], (descendant) => {
              if (validate(descendant)) {
                NodeList.push(descendant)
              }
            })
            return NodeList
          }
          return (typeof node === 'function') ? node(validate) : getAncestor(node, root, validate)
        }
    }

    if (!pseudo) {
      return instruction
    }

    const rule = pseudo.match(/-(child|type)\((\d+)\)$/)
    const kind = rule[1]
    const index = parseInt(rule[2], 10) - 1

    const validatePseudo = (node) => {
      if (node) {
        var compareSet = node.parent.childTags
        if (kind === 'type') {
          compareSet = compareSet.filter(validate)
        }
        const nodeIndex = compareSet.findIndex((child) => child === node)
        if (nodeIndex === index) {
          return true
        }
      }
      return false
    }

    return function enhanceInstruction (node) {
      const match = instruction(node)
      if (discover) {
        return match.reduce((NodeList, matchedNode) => {
          if (validatePseudo(matchedNode)) {
            NodeList.push(matchedNode)
          }
          return NodeList
        }, [])
      }
      return validatePseudo(match) && match
    }
  })
}

/**
 * Walking recursive to invoke callbacks
 *
 * @param {Array.<HTMLElement>} nodes   - [description]
 * @param {Function}            handler - [description]
 */
function traverseDescendants (nodes, handler) {
  nodes.forEach((node) => {
    var progress = true
    handler(node, () => progress = false)
    if (node.childTags && progress) {
      traverseDescendants(node.childTags, handler)
    }
  })
}

/**
 * Bubble up from bottom to top
 *
 * @param  {HTMLELement} node     - [description]
 * @param  {HTMLELement} root     - [description]
 * @param  {Function}    validate - [description]
 * @return {HTMLELement}          - [description]
 */
function getAncestor (node, root, validate) {
  while (node.parent) {
    node = node.parent
    if (validate(node)) {
      return node
    }
    if (node === root) {
      break
    }
  }
  return null
}


/**
 * Find the last common ancestor of elements
 *
 * @param  {Array.<HTMLElements>} elements - [description]
 * @return {HTMLElement}                   - [description]
 */
function getCommonAncestor (elements, options = {}) {

  const {
    root = document
  } = options

  const ancestors = []

  elements.forEach((element, index) => {
    const parents = []
    while (element !== root) {
      element = element.parentNode
      parents.unshift(element)
    }
    ancestors[index] = parents
  })

  ancestors.sort((curr, next) => curr.length - next.length)

  const shallowAncestor = ancestors.shift()

  var ancestor = null

  for (var i = 0, l = shallowAncestor.length; i < l; i++) {
    const parent = shallowAncestor[i]
    const missing = ancestors.some((otherParents) => {
      return !otherParents.some((otherParent) => otherParent === parent)
    })

    if (missing) {
      // TODO: find similar sub-parents, not the top root, e.g. sharing a class selector
      break
    }

    ancestor = parent
  }

  return ancestor
}

/**
 * Get a set of common properties of elements
 *
 * @param  {Array.<HTMLElement>} elements - [description]
 * @return {Object}                       - [description]
 */
function getCommonProperties (elements) {

  const commonProperties = {
    classes: [],
    attributes: {},
    tag: null
  }

  elements.forEach((element) => {

    var {
      classes: commonClasses,
      attributes: commonAttributes,
      tag: commonTag
    } = commonProperties

    // ~ classes
    if (commonClasses !== undefined) {
      var classes = element.getAttribute('class')
      if (classes) {
        classes = classes.trim().split(' ')
        if (!commonClasses.length) {
          commonProperties.classes = classes
        } else {
          commonClasses = commonClasses.filter((entry) => classes.some((name) => name === entry))
          if (commonClasses.length) {
            commonProperties.classes = commonClasses
          } else {
            delete commonProperties.classes
          }
        }
      } else {
        // TODO: restructure removal as 2x set / 2x delete, instead of modify always replacing with new collection
        delete commonProperties.classes
      }
    }

    // ~ attributes
    if (commonAttributes !== undefined) {
      const elementAttributes = element.attributes
      const attributes = Object.keys(elementAttributes).reduce((attributes, key) => {
        const attribute = elementAttributes[key]
        const attributeName = attribute.name
        // NOTE: workaround detection for non-standard phantomjs NamedNodeMap behaviour
        // (issue: https://github.com/ariya/phantomjs/issues/14634)
        if (attribute && attributeName !== 'class') {
          attributes[attributeName] = attribute.value
        }
        return attributes
      }, {})

      const attributesNames = Object.keys(attributes)
      const commonAttributesNames = Object.keys(commonAttributes)

      if (attributesNames.length) {
        if (!commonAttributesNames.length) {
          commonProperties.attributes = attributes
        } else {
          commonAttributes = commonAttributesNames.reduce((nextCommonAttributes, name) => {
            const value = commonAttributes[name]
            if (value === attributes[name]) {
              nextCommonAttributes[name] = value
            }
            return nextCommonAttributes
          }, {})
          if (Object.keys(commonAttributes).length) {
            commonProperties.attributes = commonAttributes
          } else {
            delete commonProperties.attributes
          }
        }
      } else {
        delete commonProperties.attributes
      }
    }

    // ~ tag
    if (commonTag !== undefined) {
      const tag = element.tagName.toLowerCase()
      if (!commonTag) {
        commonProperties.tag = tag
      } else if (tag !== commonTag) {
        delete commonProperties.tag
      }
    }
  })

  return commonProperties
}


const defaultIgnore = {
  attribute (attributeName) {
    return [
      'style',
      'data-reactid',
      'data-react-checksum'
    ].indexOf(attributeName) > -1
  }
}

/**
 * Get the path of the element
 *
 * @param  {HTMLElement} node    - [description]
 * @param  {Object}      options - [description]
 * @return {string}              - [description]
 */
function match (node, options) {

  const {
    root = document,
    skip = null,
    priority = ['href', 'class', 'src', 'title', 'id'],
    ignore = {}
  } = options

  const path = []
  var element = node
  var length = path.length
  var ignoreClass = false

  const skipCompare = skip && (Array.isArray(skip) ? skip : [skip]).map((entry) => {
    if (typeof entry !== 'function') {
      return (element) => element === entry
    }
    return entry
  })

  const skipChecks = (element) => {
    return skip && skipCompare.some((compare) => compare(element))
  }

  Object.keys(ignore).forEach((type) => {
    if (type === 'class') {
      ignoreClass = true
    }
    var predicate = ignore[type]
    if (typeof predicate === 'function') return
    if (typeof predicate === 'number') {
      predicate = predicate.toString()
    }
    if (typeof predicate === 'string') {
      predicate = new RegExp(escapeValue(predicate).replace(/\\/g, '\\\\'))
    }
    if (typeof predicate === 'boolean') {
      predicate = predicate ? /(?:)/ : /.^/
    }
    // check class-/attributename for regex
    ignore[type] = (name, value) => predicate.test(value)
  })

  if (ignoreClass) {
    const ignoreAttribute = ignore.attribute
    ignore.attribute = (name, value, defaultPredicate) => {
      return ignore.class(value) || ignoreAttribute && ignoreAttribute(name, value, defaultPredicate)
    }
  }

  while (element !== root) {
    if (skipChecks(element) !== true) {
      // ~ global
      if (checkAttributes(priority, element, ignore, path, root)) break
      if (checkTag(element, ignore, path, root)) break

      // ~ local
      checkAttributes(priority, element, ignore, path)
      if (path.length === length) {
        checkTag(element, ignore, path)
      }

      // define only one part each iteration
      if (path.length === length) {
        checkChilds(priority, element, ignore, path)
      }
    }

    element = element.parentNode
    length = path.length
  }

  if (element === root) {
    const pattern = findPattern(priority, element, ignore)
    path.unshift(pattern)
  }

  return path.join(' ')
}

/**
 * Extend path with attribute identifier
 *
 * @param  {Array.<string>} priority - [description]
 * @param  {HTMLElement}    element  - [description]
 * @param  {Object}         ignore   - [description]
 * @param  {Array.<string>} path     - [description]
 * @param  {HTMLElement}    parent   - [description]
 * @return {boolean}                 - [description]
 */
function checkAttributes (priority, element, ignore, path, parent = element.parentNode) {
  const pattern = findAttributesPattern(priority, element, ignore)
  if (pattern) {
    const matches = parent.querySelectorAll(pattern)
    if (matches.length === 1) {
      path.unshift(pattern)
      return true
    }
  }
  return false
}

/**
 * Lookup attribute identifier
 *
 * @param  {Array.<string>} priority - [description]
 * @param  {HTMLElement}    element  - [description]
 * @param  {Object}         ignore   - [description]
 * @return {string?}                 - [description]
 */
function findAttributesPattern (priority, element, ignore) {
  const attributes = element.attributes
  const sortedKeys = Object.keys(attributes).sort((curr, next) => {
    const currPos = priority.indexOf(attributes[curr].name)
    const nextPos = priority.indexOf(attributes[next].name)
    if (nextPos === -1) {
      if (currPos === -1) {
        return 0
      }
      return -1
    }
    return currPos - nextPos
  })

  for (var i = 0, l = sortedKeys.length; i < l; i++) {
    const key = sortedKeys[i]
    const attribute = attributes[key]
    const attributeName = attribute.name
    const attributeValue = escapeValue(attribute.value)

    // Check whether the value contains a UUID
    if (!!attributeValue.match(/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/)) {
      continue
    }

    const currentIgnore = ignore[attributeName] || ignore.attribute
    const currentDefaultIgnore = defaultIgnore[attributeName] || defaultIgnore.attribute
    if (checkIgnore(currentIgnore, attributeName, attributeValue, currentDefaultIgnore)) {
      continue
    }


    var pattern = `[${attributeName}="${attributeValue}"]`

    if ((/\b\d/).test(attributeValue) === false) {
      if (attributeName === 'id') {
        pattern = `#${attributeValue}`
      }

      if (attributeName === 'class') {
        const className = attributeValue.trim().replace(/\s+/g, '.')
        pattern = `.${className}`
      }
    }

    return pattern
  }
  return null
}

/**
 * Extend path with tag identifier
 *
 * @param  {HTMLElement}    element - [description]
 * @param  {Object}         ignore  - [description]
 * @param  {Array.<string>} path    - [description]
 * @param  {HTMLElement}    parent  - [description]
 * @return {boolean}                - [description]
 */
function checkTag (element, ignore, path, parent = element.parentNode) {
  const pattern = findTagPattern(element, ignore)
  if (pattern) {
    const matches = parent.getElementsByTagName(pattern)
    if (matches.length === 1) {
      path.unshift(pattern)
      return true
    }
  }
  return false
}

/**
 * Lookup tag identifier
 *
 * @param  {HTMLElement} element - [description]
 * @param  {Object}      ignore  - [description]
 * @return {boolean}             - [description]
 */
function findTagPattern (element, ignore) {
  const tagName = element.tagName.toLowerCase()
  if (checkIgnore(ignore.tag, null, tagName)) {
    return null
  }
  return tagName
}

/**
 * Extend path with specific child identifier
 *
 * NOTE: 'childTags' is a custom property to use as a view filter for tags using 'adapter.js'
 *
 * @param  {Array.<string>} priority - [description]
 * @param  {HTMLElement}    element  - [description]
 * @param  {Object}         ignore   - [description]
 * @param  {Array.<string>} path     - [description]
 * @return {boolean}                 - [description]
 */
function checkChilds (priority, element, ignore, path) {
  const parent = element.parentNode
  const children = parent.childTags || parent.children
  for (var i = 0, l = children.length; i < l; i++) {
    const child = children[i]
    if (child === element) {
      const childPattern = findPattern(priority, child, ignore)
      if (!childPattern) {
        return console.warn(`
          Element couldn\'t be matched through strict ignore pattern!
        `, child, ignore, childPattern)
      }
      const pattern = `> ${childPattern}:nth-child(${i+1})`
      path.unshift(pattern)
      return true
    }
  }
  return false
}

/**
 * Lookup identifier
 *
 * @param  {Array.<string>} priority - [description]
 * @param  {HTMLElement}    element  - [description]
 * @param  {Object}         ignore   - [description]
 * @return {string}                  - [description]
 */
function findPattern (priority, element, ignore) {
  var pattern = findAttributesPattern(priority, element, ignore)
  if (!pattern) {
    pattern = findTagPattern(element, ignore)
  }
  return pattern
}

/**
 * Validate with custom and default functions
 *
 * @param  {Function} predicate        - [description]
 * @param  {string?}  name             - [description]
 * @param  {string}   value            - [description]
 * @param  {Function} defaultPredicate - [description]
 * @return {boolean}                   - [description]
 */
function checkIgnore (predicate, name, value, defaultPredicate) {
  if (!value) {
    return true
  }
  const check = predicate || defaultPredicate
  if (!check) {
    return false
  }
  return check(name, value, defaultPredicate)
}

/**
 * Apply different optimization techniques
 *
 * @param  {string}                          selector - [description]
 * @param  {HTMLElement|Array.<HTMLElement>} element  - [description]
 * @param  {Object}                          options  - [description]
 * @return {string}                                   - [description]
 */
function optimize (selector, elements, options = {}) {

  // convert single entry and NodeList
  if (!Array.isArray(elements)) {
    elements = !elements.length ? [elements] : convertNodeList(elements)
  }

  if (!elements.length || elements.some((element) => element.nodeType !== 1)) {
    throw new Error(`Invalid input - to compare HTMLElements its necessary to provide a reference of the selected node(s)! (missing "elements")`)
  }

  const globalModified = adapt(elements[0], options)

  // chunk parts outside of quotes (http://stackoverflow.com/a/25663729)
  var path = selector.replace(/> /g, '>').split(/\s+(?=(?:(?:[^"]*"){2})*[^"]*$)/)

  if (path.length < 2) {
    return optimizePart('', selector, '', elements)
  }

  const shortened = [path.pop()]
  while (path.length > 1)  {
    const current = path.pop()
    const prePart = path.join(' ')
    const postPart = shortened.join(' ')

    const pattern = `${prePart} ${postPart}`
    const matches = document.querySelectorAll(pattern)
    if (matches.length !== elements.length) {
      shortened.unshift(optimizePart(prePart, current, postPart, elements))
    }
  }
  shortened.unshift(path[0])
  path = shortened

  // optimize start + end
  path[0] = optimizePart('', path[0], path.slice(1).join(' '), elements)
  path[path.length-1] = optimizePart(path.slice(0, -1).join(' '), path[path.length-1], '', elements)

  if (globalModified) {
    delete global.document
  }

  return path.join(' ').replace(/>/g, '> ').trim()
}

/**
 * Improve a chunk of the selector
 *
 * @param  {string}              prePart  - [description]
 * @param  {string}              current  - [description]
 * @param  {string}              postPart - [description]
 * @param  {Array.<HTMLElement>} elements - [description]
 * @return {string}                       - [description]
 */
function optimizePart (prePart, current, postPart, elements) {
  if (prePart.length) prePart = `${prePart} `
  if (postPart.length) postPart = ` ${postPart}`

  // robustness: attribute without value (generalization)
  if (/\[*\]/.test(current)) {
    const key = current.replace(/=.*$/, ']')
    var pattern = `${prePart}${key}${postPart}`
    var matches = document.querySelectorAll(pattern)
    if (compareResults(matches, elements)) {
      current = key
    } else {
      // robustness: replace specific key-value with base tag (heuristic)
      const references = document.querySelectorAll(`${prePart}${key}`)
      for (var i = 0, l = references.length; i < l; i++) {
        const reference = references[i]
        if (elements.some((element) => reference.contains(element))) {
          const description = reference.tagName.toLowerCase()
          var pattern = `${prePart}${description}${postPart}`
          var matches = document.querySelectorAll(pattern)
          if (compareResults(matches, elements)) {
            current = description
          }
          break
        }
      }
    }
  }

  // robustness: descendant instead child (heuristic)
  if (/>/.test(current)) {
    const descendant = current.replace(/>/, '')
    var pattern = `${prePart}${descendant}${postPart}`
    var matches = document.querySelectorAll(pattern)
    if (compareResults(matches, elements)) {
      current = descendant
    }
  }

  // robustness: 'nth-of-type' instead 'nth-child' (heuristic)
  if (/:nth-child/.test(current)) {
    // TODO: consider complete coverage of 'nth-of-type' replacement
    const type = current.replace(/nth-child/g, 'nth-of-type')
    var pattern = `${prePart}${type}${postPart}`
    var matches = document.querySelectorAll(pattern)
    if (compareResults(matches, elements)) {
      current = type
    }
  }

  // efficiency: combinations of classname (partial permutations)
  if (/\.\S+\.\S+/.test(current)) {
    var names = current.trim().split('.').slice(1)
                                         .map((name) => `.${name}`)
                                         .sort((curr, next) => curr.length - next.length)
    while (names.length) {
      const partial = current.replace(names.shift(), '').trim()
      var pattern = `${prePart}${partial}${postPart}`.trim()
      if (!pattern.length || pattern.charAt(0) === '>' || pattern.charAt(pattern.length-1) === '>') {
        break
      }
      var matches = document.querySelectorAll(pattern)
      if (compareResults(matches, elements)) {
        current = partial
      }
    }

    // robustness: degrade complex classname (heuristic)
    names = current && current.match(/\./g)
    if (names && names.length > 2) {
      const references = document.querySelectorAll(`${prePart}${current}`)
      for (var i = 0, l = references.length; i < l; i++) {
        const reference = references[i]
        if (elements.some((element) => reference.contains(element) )) {
          // TODO:
          // - check using attributes + regard excludes
          const description = reference.tagName.toLowerCase()
          var pattern = `${prePart}${description}${postPart}`
          var matches = document.querySelectorAll(pattern)
          if (compareResults(matches, elements)) {
            current = description
          }
          break
        }
      }
    }
  }

  return current
}

/**
 * Evaluate matches with expected elements
 *
 * @param  {Array.<HTMLElement>} matches  - [description]
 * @param  {Array.<HTMLElement>} elements - [description]
 * @return {Boolean}                      - [description]
 */
function compareResults (matches, elements) {
  const { length } = matches
  return length === elements.length && elements.every((element) => {
    for (var i = 0; i < length; i++) {
      if (matches[i] === element) {
        return true
      }
    }
    return false
  })
}


/**
 * # Utilities
 *
 * Convenience helpers.
 */

/**
 * Create an array with the DOM nodes of the list
 *
 * @param  {NodeList}             nodes - [description]
 * @return {Array.<HTMLElement>}        - [description]
 */
function convertNodeList (nodes) {
  const { length } = nodes
  const arr = new Array(length)
  for (var i = 0; i < length; i++) {
    arr[i] = nodes[i]
  }
  return arr
}

/**
 * Escape special characters and line breaks as a simplified version of 'CSS.escape()'
 *
 * Description of valid characters: https://mathiasbynens.be/notes/css-escapes
 *
 * @param  {String?} value - [description]
 * @return {String}        - [description]
 */
function escapeValue (value) {
  return value && value.replace(/['"`\\/:\?&!#$%^()[\]{|}*+;,.<=>@~]/g, '\\$&')
                       .replace(/\n/g, '\A')
}


/**
 * Get a selector for the provided element
 *
 * @param  {HTMLElement} element - [description]
 * @param  {Object}      options - [description]
 * @return {string}              - [description]
 */
function getSingleSelector (element, options = {}) {

  if (element.nodeType === 3) {
    element = element.parentNode
  }

  if (element.nodeType !== 1) {
    throw new Error(`Invalid input - only HTMLElements or representations of them are supported! (not "${typeof element}")`)
  }

  const globalModified = adapt(element, options)

  const selector = match(element, options)
  const optimized = optimize(selector, element, options)

  // debug
  // console.log(`
  //   selector:  ${selector}
  //   optimized: ${optimized}
  // `)

  if (globalModified) {
    delete global.document
  }

  return optimized
}

/**
 * Get a selector to match multiple descendants from an ancestor
 *
 * @param  {Array.<HTMLElement>|NodeList} elements - [description]
 * @param  {Object}                       options  - [description]
 * @return {string}                                - [description]
 */
function getMultiSelector (elements, options = {}) {

  if (!Array.isArray(elements)) {
    elements = convertNodeList(elements)
  }

  if (elements.some((element) => element.nodeType !== 1)) {
    throw new Error(`Invalid input - only an Array of HTMLElements or representations of them is supported!`)
  }

  const globalModified = adapt(elements[0], options)

  const ancestor = getCommonAncestor(elements, options)
  const ancestorSelector = getSingleSelector(ancestor, options)

  // TODO: consider usage of multiple selectors + parent-child relation + check for part redundancy
  const commonSelectors = getCommonSelectors(elements)
  const descendantSelector = commonSelectors[0]

  const selector = optimize(`${ancestorSelector} ${descendantSelector}`, elements, options)
  const selectorMatches = convertNodeList(document.querySelectorAll(selector))

  if (!elements.every((element) => selectorMatches.some((entry) => entry === element) )) {
    // TODO: cluster matches to split into similar groups for sub selections
    return console.warn(`
      The selected elements can\'t be efficiently mapped.
      Its probably best to use multiple single selectors instead!
    `, elements)
  }

  if (globalModified) {
    delete global.document
  }

  return selector
}

/**
 * Get selectors to describe a set of elements
 *
 * @param  {Array.<HTMLElements>} elements - [description]
 * @return {string}                        - [description]
 */
function getCommonSelectors (elements) {

  const { classes, attributes, tag } = getCommonProperties(elements)

  const selectorPath = []

  if (tag) {
    selectorPath.push(tag)
  }

  if (classes) {
    const classSelector = classes.map((name) => `.${name}`).join('')
    selectorPath.push(classSelector)
  }

  if (attributes) {
    const attributeSelector = Object.keys(attributes).reduce((parts, name) => {
      parts.push(`[${name}="${attributes[name]}"]`)
      return parts
    }, []).join('')
    selectorPath.push(attributeSelector)
  }

  if (selectorPath.length) {
    // TODO: check for parent-child relation
  }

  return [
    selectorPath.join('')
  ]
}

/**
 * Choose action depending on the input (multiple/single)
 *
 * NOTE: extended detection is used for special cases like the <select> element with <options>
 *
 * @param  {HTMLElement|NodeList|Array.<HTMLElement>} input   - [description]
 * @param  {Object}                                   options - [description]
 * @return {string}                                           - [description]
 */
function getQuerySelector (input, options = {}) {
  if (input.length && !input.name) {
    return getMultiSelector(input, options)
  }
  return getSingleSelector(input, options)
}


global = window
let selectors = [];
let elements_with_selectors = [];
var elements = arguments[0];
var keep_text_elements_only = arguments[1];
for (let element of elements) {
    if (keep_text_elements_only && element.innerText === "") {
       continue;
    }
    try {
      var selector = getSingleSelector(element)
      if (selector !== null) {
        selectors = selectors.concat(selector);
        elements_with_selectors = elements_with_selectors.concat(element)
      }
    } catch (error) {}
}
return [selectors, elements_with_selectors]