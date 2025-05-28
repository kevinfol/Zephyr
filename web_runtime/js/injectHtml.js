'use strict'

/*
    Module: injectHtml.js
    Author: Kevin Foley
    Description:
        Provides a function to inject common HTML elements
        into pages, allowing for a more modular application 
        design.
*/

function injectHtml(sourcePath, destElement) {
    /**
        * Function that injects specificed html file ('sourcePath')
        * into the specified DOM node ('destElement').
        *
        * @param sourcePath - a path to a source HTML file
        * @param destElement - an html element into which this 
        *     content will be injected.
    */

    try {

        // Retreive the html to be injected
        const html = getHtmlSource(sourcePath);

        // inject the html into destination node
        destElement.innerHTML = html;

        // Some janky stuff to re-inject any script tags
        destElement.querySelectorAll('script').forEach((script) => {

            // Create new script tag
            const newScript = document.createElement('script');

            // Copy attributes of script to new script tag
            Array.from(script.attributes).forEach(attr =>
                newScript.setAttribute(attr.name, attr.value)
            );
            newScript.setAttribute('async', 'false');

            // Inject content of existing script into new tag
            newScript.appendChild(
                document.createTextNode(script.innerHTML)
            )

            // Put the script below the inject script to be run immediatly
            script.parentNode.removeChild(script);
            insertNodeAfter(document.getElementById('inject-script'), newScript);
        });

    } catch (err) {
        console.error(err.message);
    }
}

/**
 * Inserts a new node immediately after the 'referenceNode'
 * 
 * @param {*} referenceNode - node that the new node will be inserted after
 * @param {*} newNode - new node to insert 
 */
function insertNodeAfter(referenceNode, newNode) {
    referenceNode.parentNode.insertBefore(newNode, referenceNode.nextSibling);
}



let divs = document.querySelectorAll("div[include]");
for (let i = 0; i < divs.length; i++) {
    let elem = divs[i];
    injectHtml(elem.getAttribute("include"), elem)
}
