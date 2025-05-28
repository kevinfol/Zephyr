'use strict'

/*
    Module: onnx.js
    Author: Kevin Foley
    Description:
        Provides functionality for working with onnx files in the 
        browser runtime environment.
*/

import * as ort from 'onnxruntime-web';

class ONNXFile {
    /**
     * ONNXFile is a class that stores metadata and custom methods
     * associated with an ONNX file. 
     * 
     * Properties:
     * 
     * Methods:
     * 
     * 
     */
    constructor(filename) {
        // Creates the ONNX web runtime session using the filename and
        // stores some metadata about the model session
        this.createSession(filename).then(() => this.createMetadata());

    }

    async createSession(filename) {
        this.session = await ort.InferenceSession.create(filename);
    }

    createMetadata() {
        this.inputNames = this.session.inputNames;
        this.outputNames = this.session.outputNames;
        this.inputMetadata = this.session.inputMetadata;
        this.outputMetadata = this.session.outputMetadata;
    }
}
