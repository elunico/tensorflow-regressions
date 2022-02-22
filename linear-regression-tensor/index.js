const xs = [];
const ys = [];

let m, b;

let learningRate = 0.2;
let learningSlider;
let optimizer = tf.train.sgd(learningRate);

let linex1, linex2, liney1, liney2;

function setup() {
  createCanvas(600, 600);

  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  learningSlider = createSlider(0, 2, 0.1, 0.05);
  learningSlider.style('width', '600px');
  learningSlider.input(() => {
    optimizer = tf.train.sgd(learningSlider.value());
  });
}

function predict(xs) {
  return tf.tensor1d(xs).mul(m).add(b);
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function mouseDragged(event) {
  return mousePressed(event);
}

function mousePressed(event) {
  if (mouseX > width || mouseX < 0 || mouseY > height || mouseY < 0) return;
  xs.push(mouseX / (width / 2) - 1);
  ys.push(mouseY / (height / 2) - 1);
  event.preventDefault();
  return false; // preventDefault
}

function draw() {
  background(255);
  stroke(0);
  strokeWeight(8);

  for (let i = 0; i < min(xs.length, ys.length); i++) {
    point((xs[i] + 1) * (width / 2), (ys[i] + 1) * (height / 2));
  }

  if (xs.length != 0)
    optimizer.minimize(() => loss(predict(xs), tf.tensor1d(ys)));

  tf.tidy(() => {
    let result = predict([-1, 1]);
    result.data().then(values => {
      const [y0, y1] = values;
      [linex1, liney1, linex2, liney2] = [0 * width, (y0 + 1) * (height / 2), 1 * width, (y1 + 1) * (height / 2)];
    });
  });
  line(linex1, liney1, linex2, liney2);

  strokeWeight(2);
  textSize(20);
  textFont('Courier');
  stroke(255);
  fill(0);
  text('m = ' + nf(m.dataSync(), 2, 4), 10, 25);
  text('b = ' + nf(b.dataSync(), 2, 4), 10, 50);
  text('lr = ' + nf(learningSlider.value(), 2, 4), 10, 75);


}
