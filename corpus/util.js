const fs = require('fs');

export async function fileExists(path) {
  return new Promise((resolve) => {
    fs.access(path, (err) => {
      if (err)
        resolve(false);
      else
        resolve(true);
    });
  });
}