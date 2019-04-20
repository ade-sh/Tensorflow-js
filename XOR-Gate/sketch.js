let xs;
let ys;
let yy;
let statusL;
let results;
let resolution = 50;
let cols;
let rows;
let counter = 0;
let inputs;
let training = false;
let batch;
let history;
let stat;
let epochNum=0;
function train() {
  if (!training) {
    training = true;
    AsyTrain().then(finished);
	  
	}
}

function finished() {
  counter++;
  //statusP.html('training pass: ' + counter + '<br>framerate: ' + floor(frameRate()));
  training = false;
  Pred(batch).then(yy => (results = yy));
  stat='training pass: ' + counter + '<br>framerate: ' + floor(frameRate())+'<br>Epochs:'+epochNum+'<br>Memory(Bytes):'+tf.memory().numBytes;
  
  // We need to let the JavaScript event loop tick forward before we call `train()`.
  setTimeout(train, 0);
}

function setup() {
  // Crude interface	
  createCanvas(400, 400);
    statusL=createP();
  noStroke();
  textAlign(CENTER, CENTER);
  xs = tf.tensor2d([[0, 0],[0, 1],[1, 0],[1, 1],[0, 0],[0, 1],[1, 0],[1, 1],[0, 0],[0, 1],[1, 0],[1, 1],[0, 0],[0, 1],[1, 0],[1, 1]]);
  ys = tf.tensor2d([[0],[1],[1],[0],[0],[1],[1],[0],[0],[1],[1],[0],[0],[1],[1],[0]]);

  model = tf.sequential({
	 });
  const hidden = tf.layers.dense({
    units: 2,
    inputShape: [2],
    activation: 'tanh'
	
  });
  const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  });
  model.add(hidden);
  model.add(output);

  const LEARNING_RATE = 0.3;
  const optimizer = tf.train.sgd(LEARNING_RATE);

  model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  
  batch=new Batch();
	cols = width / resolution;
		rows = height / resolution;  
		for (let i = 0; i < cols; i++) {
			for (let j = 0; j < rows; j++) {
			let x1 = i / cols;
			let x2 = j / rows;
			
			batch.add([x1, x2]);
			
    }
  }
            batch.toTensor();
train();			
}

async function AsyTrain() {
	  // This is leaking https://github.com/tensorflow/tfjs/issues/457
    const history = await model.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.1,
    epochs: 3,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        
      },
      onBatchEnd: async (batch, logs) => {
        await tf.nextFrame();
		 },
      onTrainEnd: () => {
        finished();
        results=Pred(batch);
		epochNum++;
		//Pred(0,0);
      },
    },
});
statusL.html(stat+'<br>Loss:'+floor(history.history.loss[0]));
}


async function Pred(inp) {
  //const saveResults = await model.save('localstorage://my-model-1');

  let ys = tf.tidy(() => {
      let data;
      if (inp instanceof Batch) {
        data = inp.data;
      } else {
        data = [inp];
      }
      const xs = data instanceof tf.Tensor ? data : tf.tensor2d(data);    
    return model.predict(xs);
  });
  let res = await ys.data();
  ys.dispose();
  return res;

}
function draw(){
	tf.tidy(() => {
		
	if(results!=null){
	  for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
  
	  let y = results[i + j * rows];
	 if(y!=null){
      fill(y * 255);
      rect(i * resolution, j * resolution, resolution, resolution);
      fill(i - y * 255);
     text(
		nf(y, 0, 2),
        i * resolution + resolution / 2,
        j * resolution + resolution / 2
      );
	  }
	  }
	}}
	});
}

class Batch {
  constructor() {
    // Need to deal with shape
    // this.shape = ??;
    this.data = [];
  }

  add(data) {
    this.data.push(data);
  }

  toTensor() {
    this.data = tf.tensor2d(this.data);
  }
}

