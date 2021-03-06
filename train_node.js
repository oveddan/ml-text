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
 * Training of a next-char prediction model.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs';

import * as argparse from 'argparse';

import {TextData} from './data';
import {createModel, compileModel, fitModel, generateText} from './model';

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Train an lstm-text-generation model.'
  });
  parser.addArgument('textFile', {
    type: 'string',
    // choices: Object.keys(TEXT_DATA_URLS),
    help: "Path of corpus file, located within the './data' folder"
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Use GPU with node.gl for training.  This offers a slight ' + 
      'increase in performance over cpu, but is experimental ' + 
      'and can often fail.'
  });
  parser.addArgument('--cuda', {
    action: 'storeTrue',
    help: 'Use Cuda GPU for training.'
  });
  parser.addArgument('--sampleLen', {
    type: 'int',
    defaultValue: 60,
    help: 'Sample length: Length of each input sequence to the model, in ' +
    'number of characters.'
  });
  parser.addArgument('--sampleStep', {
    type: 'int',
    defaultValue: 3,
    help: 'Step length: how many characters to skip between one example ' +
    'extracted from the text data to the next.'
  });
  parser.addArgument('--learningRate', {
    type: 'float',
    defaultValue: 1e-2,
    help: 'Learning rate to be used during training'
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 150,
    help: 'Number of training epochs'
  });
  parser.addArgument('--examplesPerEpoch', {
    type: 'int',
    defaultValue: 10000,
    help: 'Number of examples to sample from the text in each training epoch.'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size for training.'
  });
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.0625,
    help: 'Validation split for training.'
  });
  parser.addArgument('--displayLength', {
    type: 'int',
    defaultValue: 120,
    help: 'Length of the sampled text to display after each epoch of training.'
  });
  parser.addArgument('--save', {
    type: 'string',
    help: "Name of model to save; will be saved within the './models' folder"
  });
  parser.addArgument('--debugAtEpoch', {
    type: 'int',
    defaultValue: 5,
    help: 'Test inference every x epochs'
  });
  parser.addArgument('--lstmLayerSize', {
    type: 'string',
    defaultValue: '128,128',
    help: 'LSTM layer size. Can be a single number or an array of numbers ' +
    'separated by commas (E.g., "256", "256,128")'
  });  // TODO(cais): Support
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();
  if (args.gpu) {
    require('./tfjs-backend-nodegl');
    const gl = tf.backend().getGPGPUContext().gl;
    console.log(`  - gl.VERSION: ${gl.getParameter(gl.VERSION)}`);
    console.log(`  - gl.RENDERER: ${gl.getParameter(gl.RENDERER)}`)
  } else if (args.cuda) {
    console.log('Using CUDA GPU');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    require('@tensorflow/tfjs-node');
  }
  console.log('backend:', tf.getBackend());

  // Create the text data object.
  const localTextDataPath = path.join(__dirname, 'data', args.textFile);
  const text = fs.readFileSync(localTextDataPath, {encoding: 'utf-8'});
  const textData =
      new TextData('text-data', text, args.sampleLen, args.sampleStep);

  // Convert lstmLayerSize from string to number array before handing it
  // to `createModel()`.
  const lstmLayerSize = args.lstmLayerSize.indexOf(',') === -1 ?
      Number.parseInt(args.lstmLayerSize) :
      args.lstmLayerSize.split(',').map(x => Number.parseInt(x));

  const model = createModel(
      textData.sampleLen(), textData.charSetSize(), lstmLayerSize);
  compileModel(model, args.learningRate);

  console.log('text size', text.length);

  // Get a seed text for display in the course of model training.
  const [seed, seedIndices] = textData.getRandomSlice();
  const { debugAtEpoch } = args;
  console.log(`Seed text:\n"${seed}"\n`);

  const DISPLAY_TEMPERATURES = [0.25, 0.5, 0.75];

  const fullSavePath = args.save ? path.join(__dirname, 'models', args.save) : null;

  let epochCount = 0;
  let startTime = 0;
  await fitModel(
      model, textData, args.epochs, args.examplesPerEpoch, args.batchSize,
      args.validationSplit, {
        onTrainBegin: async () => {
          epochCount++;
          console.log(`Epoch ${epochCount} of ${args.epochs}:`);
        },
        onEpochBegin: async () => {
          startTime = new Date().getTime();
        },
        onEpochEnd: async () => {
          console.log('time to complete epoch:', (new Date().getTime() - startTime) / 1000);
          if (fullSavePath) {
            console.log(`saving to ${fullSavePath}`);
            model.save(`file://${fullSavePath}`);
          }
        },
        onTrainEnd: async () => {
          if ((epochCount - 1) % debugAtEpoch === 0) {
            DISPLAY_TEMPERATURES.forEach(async temperature => {
              const generated = await generateText(
                  model, textData, seedIndices, args.displayLength, temperature);
              console.log(
                  `Generated text (temperature=${temperature}):\n` +
                  `"${generated}"\n`);
            });
          }
        }
      });

  if (fullSavePath) {
    await model.save(`file://${fullSavePath}`);
    console.log(`Saved model to ${fullSavePath}`);
  }
}

main();
