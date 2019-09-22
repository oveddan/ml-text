import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import * as argparse from 'argparse';

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Convert posts json to corpus that can be trained on.'
  });
  parser.addArgument('inputFolder', {
    type: 'string',
    help: 'Path of the folder the blog posts exist'
  });
  parser.addArgument('outputFile', {
    type: 'string',
    help: 'Output text file name'
  });
  return parser.parseArgs();
}

function loadBlogPosts(postsFolder) {
  const blogPostFiles = fs.readdirSync(postsFolder);

  const postsText = blogPostFiles.map(fileName => fs.readFileSync(path.join(postsFolder, fileName), 'utf-8'));

  return postsText;
}

function toTokenizableText(posts) {

  return posts.map(x => 
    x.replace(/\+\+\+/g, '---')
    .replace("title =", "title:")
    .replace("description =", "description:")
    .replace("tags =", "tags:")
    .replace("date =", "date:")
    .replace("classes =", "classes:")
    .replace("menu =", "menu:")
  );
  
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

  const { inputFolder, outputFile } = args;

  const fullInputFolderPath = path.join(__dirname, 'data', inputFolder);

  const posts = loadBlogPosts(fullInputFolderPath);

  const postsOfText = toTokenizableText(posts);

  console.log(`creating corpus from ${postsOfText.length} with ${getNumberOfCharacters(postsOfText)} characters`);

  const asCorpus = convertPostsToCorpus(postsOfText, 'Ÿ');

  const fullOutputPath = path.join(__dirname, 'data', outputFile);

  console.log('saving corpus to ' + fullOutputPath);

  fs.writeFileSync(fullOutputPath, asCorpus);
}

main();