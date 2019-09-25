const argparse= require('argparse');
const fetch = require('node-fetch');

async function genTextUsingRunwayGpt2(inputString) {
  const inputs = {
    "prompt": inputString,
    "seed": 357,
  };

  const response = await fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(inputs)
  });
  
  const outputs = await response.json();

  console.log('got outputs', outputs);

  return outputs.text;
}


async function main() {
  const parser = argparse.ArgumentParser({
    description: 'Generate text using GPT-2 on runway'
  });

  parser.addArgument('prompt', {
    type: 'string',
    help: 'Text prompt'
  });

  const args = parser.parseArgs();

  const prompt = args.prompt;

  console.log('generating text with prompt:');
  console.log(prompt);

  const resultingText = await genTextUsingRunwayGpt2(prompt);

  console.log('resulting text:');

  console.log(resultingText);
}

main();