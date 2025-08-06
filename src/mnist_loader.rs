use image::io::Reader as ImageReader;
use image::GenericImageView;
use nalgebra::DVector;
use std::fs;
use std::path::Path;

/// Loads MNIST images from a root directory (e.g., "mnist_png/train/")
/// Expects subdirectories named `0`, `1`, ..., `9` containing PNGs.
pub fn load_images(root_dir: &str) -> Vec<(DVector<f64>, DVector<f64>)> {
    let mut dataset = vec![];

    for label_entry in fs::read_dir(root_dir).unwrap() {
        let label_path = label_entry.unwrap().path();
        if !label_path.is_dir() {
            continue;
        }

        // Parse directory name as digit label (0-9)
        let label_str = label_path.file_name().unwrap().to_string_lossy();
        let label_digit: usize = match label_str.parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        // Process each image in this digit directory
        for img_entry in fs::read_dir(&label_path).unwrap() {
            let img_path = img_entry.unwrap().path();
            if img_path.extension().unwrap_or_default() != "png" {
                continue;
            }

            let img = ImageReader::open(&img_path)
                .unwrap()
                .decode()
                .unwrap()
                .to_luma8();

            let width = img.width() as usize;
            let height = img.height() as usize;

            let input = DVector::from_iterator(
                width * height,
                img.pixels().map(|p| p[0] as f64 / 255.0),
            );

            let mut label_vec = vec![0.0; 10];
            label_vec[label_digit] = 1.0;
            let output = DVector::from_vec(label_vec);

            dataset.push((input, output));
        }
    }

    dataset
}
