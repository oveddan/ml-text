/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Use a trained next-character prediction model to generate some text. 
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as argparse from 'argparse';

import * as tf from '@tensorflow/tfjs';

const puppeteer = require('puppeteer');

import {TextData} from './data';
import {generateText} from './model';

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Use a pre-trained lstm model to post to facebook.'
  });
  parser.addArgument('textFile', {
    type: 'string',
    // choices: Object.keys(TEXT_DATA_URLS),
    help: 'Path of corpus file'
  });
  parser.addArgument('modelJSONPath', {
    type: 'string',
    help: 'Path to the trained next-char prediction model saved on disk ' +
    '(e.g., ./my-model/model.json)'
  });
  parser.addArgument('--genLength', {
    type: 'int',
    defaultValue: 200,
    help: 'Length of the text to generate.'
  });
  parser.addArgument('--temperature', {
    type: 'float',
    defaultValue: 0.5,
    help: 'Temperature value to use for text generation. Higher values ' +
    'lead to more random-looking generation results.'
  });
  parser.addArgument('--sampleStep', {
    type: 'int',
    defaultValue: 3,
    help: 'Step length: how many characters to skip between one example ' +
    'extracted from the text data to the next.'
  });
  return parser.parseArgs();
}

function loadTextData(textFile, sampleStep, model) {
  // Create the text data object.
  // Create the text data object.
  const localTextDataPath = path.join(__dirname, 'data', textFile);
  const text = fs.readFileSync(localTextDataPath, {encoding: 'utf-8'});

  const sampleLen = model.inputs[0].shape[1];

  const textData = new TextData('text-data', text, sampleLen, sampleStep);

  return textData;
}

async function loadModelAndGenerateText(args) {
  if (args.gpu) {
    require('./tfjs-backend-nodegl');
    const gl = tf.backend().getGPGPUContext().gl;
    console.log(`  - gl.VERSION: ${gl.getParameter(gl.VERSION)}`);
    console.log(`  - gl.RENDERER: ${gl.getParameter(gl.RENDERER)}`)
  } else {
    console.log('Using CPU');
    require('@tensorflow/tfjs-node');
  }

  // Load the lstm model.
  const fullModelPath = path.join(__dirname, 'models', args.modelJSONPath, '/model.json')
  const model = await tf.loadLayersModel(`file://${fullModelPath}`);

  // load the text data
  const textData = loadTextData(args.textFile, args.sampleStep, model);

  // Get a seed text from the text data object.
  const [seed, seedIndices] = textData.getRandomSlice();
  
  console.log(`Seed text:\n"${seed}"\n`);

  const generated = await generateText(
      model, textData, seedIndices, args.genLength, args.temperature);

  return generated;

}

async function promptForFacebookLogin(page) {
  await page.goto('https://www.facebook.com');

  // enter email address and password
  // give page time to load
  console.log("ENTER YOUR USERNAME AND PASSWORD");

  await page.waitFor('textarea');
  // click on page to get rid of message
  await page.mouse.click(1000, 1000);
}

async function main() {
  const args = parseArgs();

  const browser = await puppeteer.launch({
    headless: false,
    slowMo: 10
  });

  const context = await browser.createIncognitoBrowserContext();
  const page = await context.newPage();

  await page.setViewport({
    width: 1280,
    height: 720
  });

  await promptForFacebookLogin(page);

  console.log('generating text')
  const generatedText = await loadModelAndGenerateText(args);
  console.log('generated text', generatedText);

  // console.log('waiting for post box');
  const postBoxSelector = 'textarea';
  await page.waitFor(postBoxSelector);

  // open the post box
  console.log('clicking to open post box');
  await page.click(postBoxSelector);

  await page.keyboard.type(generatedText);

  console.log('clicking to post');
  // click the post button
  await page.waitFor('[data-testid="react-composer-post-button"]');
  await page.click('[data-testid="react-composer-post-button"]');
}

main();
