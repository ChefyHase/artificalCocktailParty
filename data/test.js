const data = require('./data.js');

async function main() {
  const xs = await data.nextBatch();
  xs.print(true)
}

main()
