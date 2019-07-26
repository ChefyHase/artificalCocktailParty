const fs = require("fs");
const path = require('path');
const decode = require("wav-decoder").decode.sync;
const tf = require('@tensorflow/tfjs-node-gpu');

class Data {
  constructor(args) {
    this.audioDirName = path.join(__dirname, "drive/'My Drive'", 'cv-valid-dev');
    this.samples = [];
    this.index = 0;
  }

  async loadSamples(index) {
    const buffer = fs.readFileSync(path.join(this.audioDirName, `${index}.wav`));
    let { sampleRate, channelData } = decode(buffer);
    // trim 1 sec.
    channelData = channelData[0].slice(sampleRate * 1);
    let dataArrays = tf.data.array(channelData).batch(512);
    dataArrays = dataArrays.take(dataArrays.size - 1);
    this.samples.push(...await dataArrays.toArray());
  }

  async nextBatch() {
    for (const length = this.index + 5; this.index < length; this.index++) {
      await this.loadSamples(this.index);
    }
    const xs = tf.stack(this.samples);
    this.samples = [];
    return xs.reshape([...xs.shape, 1]);
  }
}

module.exports = new Data();
