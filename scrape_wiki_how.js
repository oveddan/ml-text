const fs = require('fs');
const puppeteer = require('puppeteer');
const {join} = require('path');
const striptags = require('striptags');

const { fileExists,  mkdirp} = require('./util');
const argparse = require('argparse');

const toCorpus = (htmlParagraphs) => {
  const stripped = htmlParagraphs.map(x => striptags(x));

  return stripped.join('\n');
};

const last = (elements) => elements[elements.length - 1];
const toTitle = (link) => last(link.split('/')).replace(/%2/g, "");

const contentsDir = join(__dirname, 'data/wikihow_results');


const filePath = (link) => {
  const title = toTitle(link);

  return join(contentsDir, title + '.txt');
};

async function getLinksOnPage(page) {
  const links = await page.evaluate(() => {
    const links = document.querySelectorAll('#searchresults_list a');

    const result = [];

    links.forEach(link => result.push(link.href));
    return result;
  });

  return links;
}

async function getTextContentFromParagraphsOnPage(page) {
  const paragraphs = await page.evaluate(() => {
    const paragraphs = document.querySelectorAll('#intro p, .step');

    const result = [];

    paragraphs.forEach(paragraph => result.push(paragraph.innerText));

    return result;
  });
  return paragraphs;
}

async function getSearchResults(browser, term, pageNumber) {
  // make sure directory to save contents to exists
  await mkdirp(contentsDir);

  console.log('searching ', term);
  const searchUrl =
      `https://www.wikihow.com/wikiHowTo?search=${term}&start=${pageNumber * 10}`

  const page = await browser.newPage();

  await page.goto(searchUrl)
  
  console.log('getting links on the page.');
  const links = await getLinksOnPage(page);
  
  console.log('scraping pages at links', links);

  for (let i = 0; i < links.length; i++) {
    const link = links[i];

    const savePath = filePath(link);

    if (await fileExists(savePath)) {
      console.log('file already exists...continuing');
      continue;
    }

    await page.goto(link);
    
    const paragraphs = await getTextContentFromParagraphsOnPage(page);

    const asText = toCorpus(paragraphs);

    console.log('saving to', savePath);
    fs.writeFileSync(savePath, asText);
  }

  await page.close();
}

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'Scrape wikihow pages based on a search term, and save the contents as text files.'
  });
  parser.addArgument('searchTerm', {
    type: 'string',
    help: 'Term to search'
  });
  
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();

  const browser = await puppeteer.launch({
    headless: false
  });

  await getSearchResults(browser, args.searchTerm, 0);

  await browser.close();
}

main();
