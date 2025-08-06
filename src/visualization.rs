use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Luma, Pixel, Rgba};
use nalgebra::DVector;

pub fn overlay_saliency_on_image(
    original: &DVector<f64>,
    saliency: &DVector<f64>,
    width: u32,
    height: u32,
    output_path: &str,
) {
    let original_img = ImageBuffer::from_fn(width, height, |x, y| {
        let idx = (y * width + x) as usize;
        let val = (original[idx] * 255.0).clamp(0.0, 255.0) as u8;
        Luma([val])
    });

    let original_rgb = DynamicImage::ImageLuma8(original_img).to_rgb8();

    let max_val = saliency.amax();
    let heatmap = ImageBuffer::from_fn(width, height, |x, y| {
        let idx = (y * width + x) as usize;
        let norm = (saliency[idx].abs() / max_val).clamp(0.0, 1.0);
        let red = (norm * 255.0) as u8;
        Rgba([red, 0, 0, 128])
    });

    let mut final_img = DynamicImage::ImageRgb8(original_rgb).to_rgba8();
    for (x, y, pixel) in heatmap.enumerate_pixels() {
        let base = final_img.get_pixel_mut(x, y);
        let overlay = *pixel;

        let alpha = overlay[3] as f32 / 255.0;
        for c in 0..3 {
            base[c] = ((1.0 - alpha) * base[c] as f32 + alpha * overlay[c] as f32).clamp(0.0, 255.0) as u8;
        }
    }

    final_img.save(output_path).unwrap();
}
