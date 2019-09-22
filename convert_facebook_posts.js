import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import * as argparse from 'argparse';

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Convert posts json to corpus that can be trained on.'
  });
  parser.addArgument('inputJsonFile', {
    type: 'string',
    help: 'Path of the posts json file to convert'
  });
  parser.addArgument('outputFile', {
    type: 'string',
    help: 'Output text file name'
  });
  return parser.parseArgs();
}

function loadJsonPosts(path) {
  const postsText = fs.readFileSync(path, 'utf-8');

  return JSON.parse(postsText);
}

function extractPostsText(facebookPosts) {
  const postsWithData = facebookPosts.filter(post => post.data);

  const dataOfPosts = postsWithData.map(post => post.data[0]);

  const dataOfPostsWithPosts = dataOfPosts.filter(data => data && data.post);

  const posts = dataOfPostsWithPosts.map(data => data.post);

  return posts;
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

  const { inputJsonFile, outputFile } = args;

  const fullInputPath = path.join(__dirname, inputJsonFile);

  const posts = loadJsonPosts(fullInputPath);


  const postsOfText = extractPostsText(posts);

  console.log(`creating corpus from ${postsOfText.length} with ${getNumberOfCharacters(postsOfText)} characters`);

  const asCorpus = convertPostsToCorpus(postsOfText, '|');

  const fullOutputPath = path.join(__dirname, outputFile);

  console.log('saving corpus to ' + fullOutputPath);

  fs.writeFileSync(fullOutputPath, asCorpus);
}

main();