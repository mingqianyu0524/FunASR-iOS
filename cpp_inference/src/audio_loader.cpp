#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include <iostream>
#include "audio_loader.h"

const drwav_uint32 standard_sample_rate = 16000; // 16k
const drwav_uint64 max_frames = 480000; // 30秒 * 16k

std::vector<float> AudioLoader::load(const std::string& filename)
{
    drwav wav;
    if (!drwav_init_file(&wav, filename.c_str(), NULL)) {
        std::cerr << "Error init file: " << filename << std::endl;
        return {};
    }

    drwav_uint32 sample_rate = wav.sampleRate;
    if (sample_rate != standard_sample_rate) {
        std::cerr << "Error sample rate mismatch, current: " << wav.sampleRate << std::endl;
        drwav_uninit(&wav);
        return {};
    }
    
    // pad or truncate
    drwav_uint16 channels = wav.channels;
    std::vector<float> buffer(max_frames * channels, 0.0f);
    drwav_uint64 frames_to_read = std::min(wav.totalPCMFrameCount, max_frames);
    std::cout << "frames to read = " << frames_to_read << std::endl;
    drwav_uint64 frames_read = drwav_read_pcm_frames_f32(&wav, frames_to_read, buffer.data());
    std::cout << "frames read = " << frames_read << std::endl;
    if (frames_to_read != frames_read) {
        if (frames_read == 0 && frames_to_read > 0) {
            std::cerr << "Error reading pcm frames." << std::endl;
            drwav_uninit(&wav);
            return {};
        }
    }

    std::vector<float> result(max_frames, 0.0f);
    if (channels == 1) {
        std::copy(buffer.begin(), buffer.begin() + frames_read, result.begin());
    } else if (channels == 2) {
        for (int i = 0; i < frames_read; i++) {
            float mono = (*(buffer.data() + 2 * i) + *(buffer.data() + 2 * i + 1)) / 2;
            result[i] = mono;
        }
    } else {
        std::cerr << "Error compressing channels, support mono or dual channel only" << std::endl;
        drwav_uninit(&wav);
        return {};
    }
    
    // clean up
    drwav_uninit(&wav);

    return result;
}