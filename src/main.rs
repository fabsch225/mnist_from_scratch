mod network;
mod mnist_loader;
mod visualization;
mod serialization;

use network::Network;
use mnist_loader::load_images;
use visualization::overlay_saliency_on_image;

fn main() {
    let mut train_data = load_images("data/train");
    let test_data = load_images("data/test");

    let mut net = Network::new(&[784, 64, 36, 10]);
    //let net = Network::load("trained_network.bin").expect("TODO: panic message");
    net.train(&mut train_data, 5, 0.1);

    let correct = test_data
        .iter()
        .filter(|(x, y)| {
            let predicted = net.predict(x);
            let actual = y.iter().position(|&v| v == 1.0).unwrap();
            predicted == actual
        })
        .count();

    let accuracy = 100.0 * correct as f64 / test_data.len() as f64;
    println!("Test Accuracy: {:.2}%", accuracy);

    let test_image = &test_data[0].0;
    let saliency = net.saliency_map(test_image);

    overlay_saliency_on_image(test_image, &saliency, 28, 28, "saliency_overlay.png");

    //net.save("trained_network.bin").expect("TODO: panic message");
}
