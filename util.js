const fs = require('fs');
const mkdirpCb = require('mkdirp');

function fileExists(path) {
  return new Promise((resolve) => {
    fs.access(path, (err) => {
      if (err)
        resolve(false);
      else
        resolve(true);
    });
  });
}

function mkdirp(url) {
  return new Promise((resolve, reject) => {
    mkdirpCb(url, (err, made) => {
      if (err)
        reject(err);
      else
        resolve();
    });
  });
}

module.exports = {
  fileExists,
  mkdirp
}
