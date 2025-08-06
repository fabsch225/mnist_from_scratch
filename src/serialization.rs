use bincode::{Decode, Encode};
use nalgebra::{DMatrix, DVector};
use crate::network::Network;

#[derive(Encode, Decode, Debug)]
pub struct SerializableMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<f64>,
}

#[derive(Encode, Decode, Debug)]
pub struct SerializableVector {
    pub data: Vec<f64>,
}

#[derive(Encode, Decode, Debug)]
pub struct SerializableNetwork {
    pub weights: Vec<SerializableMatrix>,
    pub biases: Vec<SerializableVector>,
}

impl From<&Network> for SerializableNetwork {
    fn from(net: &Network) -> Self {
        let weights = net.weights.iter().map(|w| SerializableMatrix {
            nrows: w.nrows(),
            ncols: w.ncols(),
            data: w.as_slice().to_vec(),
        }).collect();

        let biases = net.biases.iter().map(|b| SerializableVector {
            data: b.as_slice().to_vec(),
        }).collect();

        SerializableNetwork { weights, biases }
    }
}

impl From<SerializableNetwork> for Network {
    fn from(s: SerializableNetwork) -> Self {
        let weights = s.weights.into_iter().map(|m| {
            DMatrix::from_vec(m.nrows, m.ncols, m.data)
        }).collect();

        let biases = s.biases.into_iter().map(|v| {
            DVector::from_vec(v.data)
        }).collect();

        Network { weights, biases }
    }
}
