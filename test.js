const model = require('./model.js');
const data = require('./data/data.js');
const tf = require('@tensorflow/tfjs-node');


(async() => {
  const [encoder, denoise] = model.build();
  await data.loadSamples(0);
  const [zMean, zLogVar, z] = encoder.predict(data.samples[0].reshape([1, 512, 1]));
  z.print();
  const denoised = denoise.predict([
    z.reshape([1, 512, 1]), data.samples[0].reshape([1, 512, 1])
  ]);
  denoised.flatten().print()
})();
