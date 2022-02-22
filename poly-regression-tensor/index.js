const xs = [];
const ys = [];

let a, b, c;

let learningRate = 0.2;
let learningSlider;
let optimizer = tf.train.sgd(learningRate);

let curveX, curveY;

function setup() {
  createCanvas(600, 600)

  a = tf.variable(tf.scalar(random(1)))
  b = tf.variable(tf.scalar(random(1)))
  c = tf.variable(tf.scalar(random(1)))
  learningSlider = createSlider(0, 2, 0.1, 0.05);
  learningSlider.style('width', '50%');
  learningSlider.input(() => {
    optimizer = tf.train.sgd(learningSlider.value());
  });
}

function predict(xs) {
  xs = tf.tensor1d(xs);
  return xs.square().mul(a).add(xs.mul(b)).add(c);
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function mouseDragged() {
  mousePressed()
}

function mousePressed() {
  if (mouseX > width || mouseX < 0 || mouseY > height || mouseY < 0) return;
  xs.push(mouseX / width);
  ys.push(mouseY / height);
}

function draw() {
  background(255);
  stroke(0)
  strokeWeight(8);

  for (let i = 0; i < min(xs.length, ys.length); i++) {
    point(xs[i] * width, ys[i] * height)
  }

  if (xs.length != 0) {
    optimizer.minimize(() => loss(predict(xs), tf.tensor1d(ys)))
  }

  curveX = [];
  for (let i = 0; i < 100; i++) {
    curveX.push(i * 0.01);
  }


  tf.tidy(() => {
    let result = predict(curveX);
    curveY = result.dataSync();
  });


  stroke(0)
  strokeWeight(8);

  if (curveY.length > 0) {
    beginShape();
    for (let i = 0; i < 100; i++) {
      point(curveX[i] * width, curveY[i] * height);
    }
    endShape();
  }

  strokeWeight(2);
  textSize(20)
  textFont('Courier')
  stroke(255);
  fill(0)
  text('a = ' + nf(a.dataSync(), 2, 4), 10, 25);
  text('b = ' + nf(b.dataSync(), 2, 4), 10, 50);
  text('c = ' + nf(c.dataSync(), 2, 4), 10, 75);
  text('lr = ' + nf(learningSlider.value(), 2, 4), 10, 125);



}