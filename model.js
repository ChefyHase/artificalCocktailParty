const tf = require('@tensorflow/tfjs-node-gpu');
const data = require('./data/data.js');

class SampleLayer extends tf.layers.Layer {
  static className = 'SampleLayer';

  constructor(args) {
    super({});
  }

  computeOutputShape(inputShape) {
    return inputShape[0];
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const [z_mean, z_log_var] = inputs;
      const batch = z_mean.shape[0];
      const dim = z_mean.shape[1];
      const epsilon = tf.randomNormal([batch, dim]);
      const half = tf.scalar(0.5);
      const temp = z_log_var.mul(half).exp().mul(epsilon);
      const sample = z_mean.add(temp);
      return sample;
    });
  }

  getClassName() {
    return 'SampleLayer';
  }
}
tf.serialization.registerClass(SampleLayer);

class Model {
  constructor(args) {

  }

  dilationLayer(input, dilationRate, filters) {
    const node = tf.layers.conv1d({
      filters: filters,
      kernelSize: 1,
      strides: 1,
      padding: 'same'
    }).apply(input);
    input = tf.layers.reLU().apply(input);
    input = tf.layers.conv1d({
      filters: filters,
      kernelSize: 7,
      strides: 1,
      padding: 'same',
      dilationRate: dilationRate
    }).apply(input);
    input = tf.layers.reLU().apply(input);
    return tf.layers.add().apply([node, input]);
  }

  build() {
    const encoderInput = tf.input({ shape: [512, 1] });
    const encoderLayer1 = this.dilationLayer(encoderInput, 1, 128);
    const encoderLayer2 = this.dilationLayer(encoderLayer1, 1, 128);
    const encoderLayer3 = this.dilationLayer(encoderLayer2, 1, 128);
    const encoderLayer4 = this.dilationLayer(encoderLayer3, 1, 128);
    const encoderLayer5 = this.dilationLayer(encoderLayer4, 1, 128);
    const pooling = tf.layers.globalAveragePooling1d({ name: 'pool' }).apply(encoderLayer5);
    const mean = tf.layers.dense({ units: 512 }).apply(pooling);
    const logVar = tf.layers.dense({ units: 512 }).apply(pooling);
    const z = new SampleLayer().apply([mean, logVar]);
    const encoder = tf.model({ inputs: encoderInput, outputs: [mean, logVar, z] });
    encoder.summary();

    const denoiseAudioInput = tf.input({ shape: [512, 1] });
    const denoiseZInput = tf.input({ shape: [512, 1] });
    const concat = tf.layers.concatenate().apply([denoiseAudioInput, denoiseZInput]);
    const denoiseLayer1 = this.dilationLayer(concat, 1, 128);
    const denoiseLayer2 = this.dilationLayer(denoiseLayer1, 1, 128);
    const denoiseLayer3 = this.dilationLayer(denoiseLayer2, 1, 128);
    const denoiseLayer4 = this.dilationLayer(denoiseLayer3, 1, 128);
    const denoiseLayer5 = this.dilationLayer(denoiseLayer4, 1, 128);
    const x1Conv = tf.layers.conv1d({
      filters: 1,
      kernelSize: 1,
      strides: 1,
      padding: 'same',
      activation: 'tanh'
    }).apply(denoiseLayer5);
    const denoise = tf.model({ inputs: [denoiseAudioInput, denoiseZInput], outputs: x1Conv });
    denoise.summary();

    const model = (inputs) => {
      return tf.tidy(() => {
       const [zMean, zLogVar, z] = encoder.predict(inputs);
       const outputs = denoise.predict([inputs, z.reshape([...z.shape, 1])]);
       return [zMean, zLogVar, outputs];
     });
    }

    return [encoder, denoise, model];
  }

  denoisingLoss(yTrue, yPred) {
    return tf.tidy(() => {
      const denoisingLoss = tf.div(tf.abs(yTrue.sub(yPred)).square().sum(), tf.abs(yTrue).square().sum());
      return denoisingLoss;
    });
  }

  klLoss(zMean, zLogVar) {
    return tf.tidy(() => {
      let klLoss;
      klLoss = tf.scalar(1).add(zLogVar).sub(zMean.square()).sub(zLogVar.exp());
      klLoss = tf.sum(klLoss, -1);
      klLoss = klLoss.mul(tf.scalar(-0.5));
      return klLoss.mean();
    });
  }

  loss(yTrue, [zMean, zLogVar, yPred]) {
    return tf.tidy(() => {
      const denoisingLoss = this.denoisingLoss(yTrue, yPred);
      denoisingLoss.print()
      const klLoss = this.klLoss(zMean, zLogVar).mul(tf.scalar(100));
      klLoss.print()
      const totalLoss = denoisingLoss.add(klLoss);
      totalLoss.print()
      return totalLoss;
    });
  }

  async train() {
    const numIterations = 50;
    const numEpochs = 50;

    const [encoder, denoise, model] = this.build();
    const optimizer = tf.train.adam(0.00001);

    for (let n = 0; n < numIterations; n++) {
      console.log('Fetching training data...');
      const xs = await data.nextBatch();

      for (let m = 0; m < numEpochs; m++) {
        let trainLoss = await optimizer.minimize(() => {
          const noise = tf.randomNormal(xs.shape, 0, 0.4);
          return this.loss(xs, model(xs.add(noise)));
        }, true);
        trainLoss.print();
      }
    }
  }
}

module.exports = new Model();
