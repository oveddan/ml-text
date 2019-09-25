const fs = require('fs');
const path = require('path');
const argparse = require('argparse');
const { mkdirp } = require('./util');

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Conbines files in a folder into a text corpus, and saves them to a file in ./corpus.'
  });
  parser.addArgument('inputFolder', {
    type: 'string',
    help: 'Path of the folder to grab text files from'
  });
  parser.addArgument('outputFile', {
    type: 'string',
    help: 'Output text file name, to be saved in "data"'
  });
  parser.addArgument('delimiter', {
    type: 'string',
    default: '|',
    help: 'Delimiter to use when concattenating the files'
  });

  return parser.parseArgs();
}

function loadTextFilesInFolder(inputFolder) {
  // list files in directory
  const files = fs.readdirSync(inputFolder);

  console.log('the files', files);
  const fullPaths = files.map(file => path.join(inputFolder, file));

  const fileContents = fullPaths.map(fullPath => fs.readFileSync(fullPath, 'utf-8'));

  return fileContents;
}

function getNumberOfCharacters(arrayOfStrings) {
  return arrayOfStrings.reduce(function(sum, inputString) {
    return sum + inputString.length;
  }, 0)
}

function convertPostsToCorpus(posts, delimiter) {
  const asString = posts.join(delimiter);

  return asString;
}

async function main() {
  const args = parseArgs();

  const { inputFolder, outputFile, delimiter } = args;

  const files = loadTextFilesInFolder(inputFolder);

  console.log(`creating corpus from ${files.length} files with ${getNumberOfCharacters(files)} characters`);

  const asCorpus = convertPostsToCorpus(files, delimiter);

  const fullOutputPath = path.join(__dirname, 'data', outputFile);

  console.log('saving corpus to ' + fullOutputPath);

  fs.writeFileSync(fullOutputPath, asCorpus);
}

main();
