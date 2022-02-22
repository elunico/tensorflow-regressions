const xs = [];
const ys = [];

let a, b, c, d;

let learningRate = 0.2;
let learningSlider;
let optimizer = tf.train.sgd(learningRate);

let curveX, curveY;

function setup() {
  createCanvas(600, 600);

  a = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  c = tf.variable(tf.scalar(random(1)));
  d = tf.variable(tf.scalar(random(1)));
  learningSlider = createSlider(0, 2, 0.1, 0.05);
  learningSlider.style('width', '600px');
  learningSlider.input(() => {
    optimizer = tf.train.adam(learningSlider.value());
  });
}

function predict(xs) {
  xs = tf.tensor1d(xs);
  return xs.mul(xs).mul(xs).mul(a).add(xs.square().mul(b)).add(xs.mul(c)).add(d);
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function mouseDragged() {
  mousePressed();
}

function mousePressed() {
  if (mouseX > width || mouseX < 0 || mouseY > height || mouseY < 0) return;
  xs.push(mouseX / (width / 2) - 1);
  ys.push(mouseY / (height / 2) - 1);
}

function draw() {
  background(255);
  stroke(0);
  strokeWeight(8);

  for (let i = 0; i < min(xs.length, ys.length); i++) {
    point((xs[i] + 1) * (width / 2), (ys[i] + 1) * (height / 2));
  }

  if (xs.length != 0) {
    optimizer.minimize(() => loss(predict(xs), tf.tensor1d(ys)));
  }

  curveX = [];
  for (let i = -1; i < 1; i += 0.01) {
    curveX.push(i);
  }


  tf.tidy(() => {
    let result = predict(curveX);
    curveY = result.dataSync();
  });


  stroke(0);
  strokeWeight(8);

  if (curveY.length > 0) {
    beginShape();
    for (let i = 200; i > 0; i--) {
      point((curveX[i] + 1) * (width / 2), (curveY[i] + 1) * (height / 2));
    }
    endShape();
  }

  strokeWeight(2);
  textSize(20);
  textFont('Courier');
  stroke(255);
  fill(0);
  text('a = ' + nf(a.dataSync(), 2, 4), 10, 25);
  text('b = ' + nf(b.dataSync(), 2, 4), 10, 50);
  text('c = ' + nf(c.dataSync(), 2, 4), 10, 75);
  text('d = ' + nf(d.dataSync(), 2, 4), 10, 100);
  text('lr = ' + nf(learningSlider.value(), 2, 4), 10, 150);



}
