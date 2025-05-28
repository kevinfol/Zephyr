'use strict'

/*
    Module: index.js
    Author: Kevin Foley
    Description:
        Zephyr Web Runtime application starting point
*/

// Application initialization
const express = require('express');
const app = express();
const router = express.Router();
const port = 3210;

// Set up routes
router.get('/', (request, response) => {
    return response.render("")
})
