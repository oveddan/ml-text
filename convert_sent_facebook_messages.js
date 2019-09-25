import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import * as argparse from 'argparse';

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Extract sent messages from the facebook messages inbox json to corpus that can be trained on.'
  });
  parser.addArgument('inputFolder', {
    type: 'string',
    help: 'Path of the folder where the inbox messages exist'
  });
  parser.addArgument('outputFile', {
    type: 'string',
    help: 'Output text file name'
  });
  return parser.parseArgs();
}

function loadConversations(conversationsFolder) {
  const conversationsFolders = fs.readdirSync(conversationsFolder);

  const messageFilePaths = conversationsFolders.map(folder => path.join(conversationsFolder, folder, 'message_1.json'));

  const messagesJsons = messageFilePaths.map(filePath=> JSON.parse(fs.readFileSync(filePath, 'utf-8')));

  return messagesJsons;
}


function extractPostsText(facebookPosts) {
  const postsWithData = facebookPosts.filter(post => post.data);

  const dataOfPosts = postsWithData.map(post => post.data[0]);

  const dataOfPostsWithPosts = dataOfPosts.filter(data => data && data.post);

  const posts = dataOfPostsWithPosts.map(data => data.post);

  return posts;
}

const sender = 'Dan Oved';

function extractSentMessages(messagesJsonList) {
  const messages = messagesJsonList.filter(x => x.messages)
    .flatMap(x => x.messages);


  // const flattened = messages.flat();

  const messagesByMe = messages.filter(x => x.sender_name === sender);

  return messagesByMe.map(message => message.content);
}

function getNumberOfCharacters(arrayOfStrings) {
  return arrayOfStrings.reduce(function(sum, inputString) {
    if (!inputString) return sum;
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

  const conversations = loadConversations(fullInputFolderPath);

  const sentMessages = extractSentMessages(conversations);

  console.log(`creating corpus from ${sentMessages.length} with ${getNumberOfCharacters(sentMessages)} characters`);

  const asCorpus = convertPostsToCorpus(sentMessages, '|');

  const fullOutputPath = path.join(__dirname, 'data', outputFile);

  console.log('saving corpus to ' + fullOutputPath);

  fs.writeFileSync(fullOutputPath, asCorpus);
}

main();