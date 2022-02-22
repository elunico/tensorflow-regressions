const xs = [];
const ys = [];

let m, b;

let learningRate = 0.2;
let learningSlider;
let optimizer = tf.train.sgd(learningRate);

let linex1, linex2, liney1, liney2;

function setup() {
  createCanvas(600, 600)

  m = tf.variable(tf.scalar(random(1)))
  b = tf.variable(tf.scalar(random(1)))
  learningSlider = createSlider(0, 2, 0.1, 0.05);
  learningSlider.style('width', '50%');
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

  if (xs.length != 0)
    optimizer.minimize(() => loss(predict(xs), tf.tensor1d(ys)))

  tf.tidy(() => {
    let result = predict([0, 1]);
    result.data().then(values => {
      const [y0, y1] = values;
      [linex1, liney1, linex2, liney2] = [0 * width, y0 * height, 1 * width, y1 * height];
    });
  })
  line(linex1, liney1, linex2, liney2);

  strokeWeight(2);
  textSize(20)
  textFont('Courier')
  stroke(255);
  fill(0)
  text('m = ' + nf(m.dataSync(), 2, 4), 10, 25);
  text('b = ' + nf(b.dataSync(), 2, 4), 10, 50);
  text('lr = ' + nf(learningSlider.value(), 2, 4), 10, 75);


}