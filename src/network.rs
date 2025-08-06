use std::fs::{read, File};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::io::{Read, Write};
use bincode::config::Configuration;
use crate::serialization::SerializableNetwork;

#[derive(Debug)]
pub struct Network {
    pub(crate) weights: Vec<DMatrix<f64>>,
    pub(crate) biases: Vec<DVector<f64>>,
}

impl Network {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let config = bincode::config::standard();
        let buffer = bincode::encode_to_vec(SerializableNetwork::from(self), config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let mut file = File::create(path)?;
        file.write_all(&buffer)?;
        Ok(())
    }
    pub fn load(path: &str) -> std::io::Result<Self> {
        let buffer = read(path)?;
        let config = bincode::config::standard();

        let (serializable, _) = bincode::decode_from_slice::<SerializableNetwork, Configuration>(&buffer, config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(Network::from(serializable))
    }

    pub fn new(sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();

        let weights = sizes
            .windows(2)
            .map(|w| DMatrix::from_fn(w[1], w[0], |_, _| rng.gen_range(-1.0..1.0)))
            .collect();

        let biases = sizes[1..]
            .iter()
            .map(|&n| DVector::from_fn(n, |_, _| rng.gen_range(-1.0..1.0)))
            .collect();

        Self { weights, biases }
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn sigmoid_prime(x: f64) -> f64 {
        let s = Self::sigmoid(x);
        s * (1.0 - s)
    }

    pub fn feedforward(&self, input: &DVector<f64>) -> DVector<f64> {
        self.weights
            .iter()
            .zip(&self.biases)
            .fold(input.clone(), |a, (w, b)| (w * &a + b).map(Self::sigmoid))
    }

    pub fn train(&mut self, data: &mut [(DVector<f64>, DVector<f64>)], epochs: usize, lr: f64) {
        for _ in 0..epochs {
            data.shuffle(&mut thread_rng());
            for (x, y) in data.iter() {
                let mut activations = vec![x.clone()];
                let mut zs = vec![];

                //forward pass
                for (w, b) in self.weights.iter().zip(self.biases.iter()) {
                    let z = w * activations.last().unwrap() + b;
                    zs.push(z.clone());
                    activations.push(z.map(Self::sigmoid));
                }

                //backward pass
                let mut delta = activations
                    .last()
                    .unwrap()
                    .clone()
                    .zip_map(y, |a, y| a - y)
                    .zip_map(&zs.last().unwrap().map(Self::sigmoid_prime), |d, sp| d * sp);

                for i in (0..self.weights.len()).rev() {
                    let a = &activations[i];
                    let dw = &delta * a.transpose() * lr;
                    let db = delta.clone() * lr;

                    self.weights[i] -= dw;
                    self.biases[i] -= &db;

                    if i != 0 {
                        let z = &zs[i - 1];
                        let sp = z.map(Self::sigmoid_prime);
                        delta = self.weights[i].transpose() * &delta;
                        delta = delta.zip_map(&sp, |d, s| d * s);
                    }
                }
            }
        }
    }

    pub fn predict(&self, input: &DVector<f64>) -> usize {
        self.feedforward(input)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    pub fn saliency_map(&self, input: &DVector<f64>) -> DVector<f64> {
        let mut activations = vec![input.clone()];
        let mut zs = vec![];

        //forward ass
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let z = w * activations.last().unwrap() + b;
            zs.push(z.clone());
            activations.push(z.map(Self::sigmoid));
        }

        let output = activations.last().unwrap();
        let predicted = output.argmax().0;

        let mut y = DVector::zeros(output.len());
        y[predicted] = 1.0;

        //backward pass
        let mut delta = output.clone()
            .zip_map(&y, |a, y| a - y)
            .zip_map(&zs.last().unwrap().map(Self::sigmoid_prime), |d, sp| d * sp);

        //backpropagate
        for i in (0..self.weights.len()).rev() {
            let z = if i > 0 { &zs[i - 1] } else { return self.weights[i].transpose() * delta };
            let sp = z.map(Self::sigmoid_prime);
            delta = self.weights[i].transpose() * delta;
            delta = delta.zip_map(&sp, |d, s| d * s);
        }

        delta
    }
}
