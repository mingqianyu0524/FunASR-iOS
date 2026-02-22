#include "feature_extractor.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <numeric>

// =====================================================================
// FeatureExtractor — Kaldi-compatible 80-dim log-Fbank for SenseVoice
// =====================================================================

FeatureExtractor::FeatureExtractor()
{
    init_hamming_window();
    init_mel_filters();
    init_dft_matrices();
    std::cout << "✅ FeatureExtractor initialized (Hamming, N_FFT=512, 80-mel)" << std::endl;
}

FeatureExtractor::~FeatureExtractor() {}

// --------------- Hamming window (FRAME_LENGTH = 400) ---------------
void FeatureExtractor::init_hamming_window()
{
    hamming_window_.resize(FRAME_LENGTH);
    // Hamming: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
    vDSP_hamm_window(hamming_window_.data(), FRAME_LENGTH, 0);
}

// ------------- Mel filterbank (Kaldi / HTK style) ------------------
void FeatureExtractor::init_mel_filters()
{
    const int n_spectrum_bins = N_FFT / 2 + 1; // 257
    const int n_filters = N_MELS + 2;          // 82 edge points

    // HTK mel scale: m = 2595 * log10(1 + f/700)
    auto hz_to_mel = [](float hz) -> float {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };
    auto mel_to_hz = [](float mel) -> float {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };

    float mel_low = hz_to_mel(0.0f);
    float mel_high = hz_to_mel(static_cast<float>(SAMPLE_RATE) / 2.0f); // 8000 Hz

    // Linearly spaced points in mel domain
    std::vector<float> mel_points(n_filters);
    for (int i = 0; i < n_filters; ++i) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (n_filters - 1);
    }

    // Convert back to Hz, then to FFT bin indices
    std::vector<float> bin_points(n_filters);
    for (int i = 0; i < n_filters; ++i) {
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = hz * N_FFT / SAMPLE_RATE;
    }

    mel_filters_.resize(N_MELS * n_spectrum_bins, 0.0f);

    for (int i = 0; i < N_MELS; ++i) {
        float left   = bin_points[i];
        float center = bin_points[i + 1];
        float right  = bin_points[i + 2];

        // Slaney normalization: 2 / (right - left)
        float norm_factor = 2.0f / (right - left);

        for (int j = 0; j < n_spectrum_bins; ++j) {
            float fj = static_cast<float>(j);
            float weight = 0.0f;

            if (fj > left && fj <= center) {
                weight = (fj - left) / (center - left);
            } else if (fj > center && fj < right) {
                weight = (right - fj) / (right - center);
            }

            mel_filters_[i * n_spectrum_bins + j] = weight * norm_factor;
        }
    }
}

// -------- DFT matrices (n_spectrum_bins x N_FFT) for matmul DFT --------
void FeatureExtractor::init_dft_matrices()
{
    const int n_bins = N_FFT / 2 + 1; // 257
    dft_mat_real_.resize(n_bins * N_FFT);
    dft_mat_imag_.resize(n_bins * N_FFT);

    const float PI = 3.14159265358979323846f;
    for (int k = 0; k < n_bins; ++k) {
        for (int n = 0; n < N_FFT; ++n) {
            float angle = 2.0f * PI * k * n / N_FFT;
            dft_mat_real_[k * N_FFT + n] =  std::cos(angle);
            dft_mat_imag_[k * N_FFT + n] = -std::sin(angle);
        }
    }
    std::cout << "  DFT matrices initialized (" << n_bins << " x " << N_FFT << ")" << std::endl;
}

// ======================== Main processing ===========================
std::vector<float> FeatureExtractor::process(const std::vector<float>& pcm_audio,
                                              int& out_n_frames)
{
    const int n_samples = static_cast<int>(pcm_audio.size());
    if (n_samples < FRAME_LENGTH) {
        throw std::runtime_error("Audio too short for feature extraction");
    }

    // Number of frames (variable length, no padding to fixed size)
    const int n_frames = 1 + (n_samples - FRAME_LENGTH) / HOP_LENGTH;
    const int n_spectrum_bins = N_FFT / 2 + 1; // 257

    out_n_frames = n_frames;

    std::cout << "[FE] n_samples=" << n_samples
              << " n_frames=" << n_frames << std::endl;

    // Allocate power spectrum: [n_frames, n_spectrum_bins]
    std::vector<float> power_spectrum(n_frames * n_spectrum_bins, 0.0f);

    // Working buffers (reused per frame)
    std::vector<float> padded_frame(N_FFT, 0.0f);   // zero-padded to 512
    std::vector<float> output_real(n_spectrum_bins);
    std::vector<float> output_imag(n_spectrum_bins);
    std::vector<float> magnitudes(n_spectrum_bins);

    DSPSplitComplex temp_complex;
    temp_complex.realp = output_real.data();
    temp_complex.imagp = output_imag.data();

    for (int k = 0; k < n_frames; ++k) {
        int sample_idx = k * HOP_LENGTH;

        // 1. Extract frame (400 samples) and zero-pad to 512
        std::fill(padded_frame.begin(), padded_frame.end(), 0.0f);
        int copy_count = std::min(FRAME_LENGTH, n_samples - sample_idx);
        // Apply Hamming window while copying
        for (int i = 0; i < copy_count; ++i) {
            padded_frame[i] = pcm_audio[sample_idx + i] * hamming_window_[i];
        }

        // 2. DFT via matrix multiplication
        // output_real[257] = dft_mat_real_[257x512] * padded_frame[512]
        vDSP_mmul(dft_mat_real_.data(), 1, padded_frame.data(), 1,
                  output_real.data(), 1, n_spectrum_bins, 1, N_FFT);
        vDSP_mmul(dft_mat_imag_.data(), 1, padded_frame.data(), 1,
                  output_imag.data(), 1, n_spectrum_bins, 1, N_FFT);

        // 3. Power spectrum: |X[k]|^2 = real^2 + imag^2
        vDSP_zvmags(&temp_complex, 1, magnitudes.data(), 1, n_spectrum_bins);

        // 4. Store in row k
        std::copy(magnitudes.begin(), magnitudes.end(),
                  power_spectrum.begin() + k * n_spectrum_bins);
    }

    // Apply mel filterbank: [N_MELS, n_spectrum_bins] x [n_spectrum_bins, n_frames]
    // First transpose power_spectrum to [n_spectrum_bins, n_frames]
    std::vector<float> power_T(n_spectrum_bins * n_frames);
    vDSP_mtrans(power_spectrum.data(), 1, power_T.data(), 1,
                n_spectrum_bins, n_frames);

    // log_mel = mel_filters [80 x 257] * power_T [257 x n_frames] → [80 x n_frames]
    std::vector<float> mel_spec(N_MELS * n_frames);
    vDSP_mmul(mel_filters_.data(), 1,
              power_T.data(), 1,
              mel_spec.data(), 1,
              N_MELS, n_frames, n_spectrum_bins);

    // Log: log(max(mel, epsilon))
    float epsilon = 1e-10f;
    int n_total = N_MELS * n_frames;
    vDSP_vthr(mel_spec.data(), 1, &epsilon, mel_spec.data(), 1, n_total);

    // Natural log (kaldi uses ln, not log10)
    vvlogf(mel_spec.data(), mel_spec.data(), &n_total);

    // mel_spec is currently [80, n_frames] (row = mel bin, col = frame)
    // We need [n_frames, 80] for downstream (row = frame, col = mel bin)
    std::vector<float> fbank(n_frames * N_MELS);
    vDSP_mtrans(mel_spec.data(), 1, fbank.data(), 1,
                n_frames, N_MELS);
    // Now fbank is [n_frames, 80] — row-major

    std::cout << "[FE] Feature extraction done: " << n_frames << " x " << N_MELS << std::endl;
    return fbank;
}
