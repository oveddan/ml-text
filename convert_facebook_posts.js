const fs = require('fs');
const path = require('path');
const { mkdirp} = require('./util');

const argparse = require('argparse');

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Convert posts json to corpus that can be trained on.'
  });
  parser.addArgument('inputJsonFile', {
    type: 'string',
    help: 'Path of the posts json file to convert'
  });
  parser.addArgument('outputFolder', {
    type: 'string',
    help: 'Output folder to save files to'
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

async function main() {
  const args = parseArgs();

  const { inputJsonFile, outputFolder } = args;

  const fullInputPath = path.join(__dirname, inputJsonFile);

  const posts = loadJsonPosts(fullInputPath);

  const postsTexts = extractPostsText(posts);

  const fullOutputFolderPath = path.join(__dirname, 'data', outputFolder);
  console.log('saving posts to ' + fullOutputFolderPath );
  await mkdirp(fullOutputFolderPath);

  for(let i = 0; i < postsTexts.length; i++) {
    const fullOutputPath = path.join(fullOutputFolderPath , i + '.txt');
    fs.writeFileSync(fullOutputPath, postsTexts[i]);
  }
}

main();