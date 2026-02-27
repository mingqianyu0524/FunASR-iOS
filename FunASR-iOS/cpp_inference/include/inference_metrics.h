#pragma once

#include <cstdint>
#include <mach/mach.h>
#include <mach/mach_time.h>

struct InferenceMetrics {
    double fbank_time_ms = 0.0;
    double lfr_cmvn_time_ms = 0.0;
    double onnx_forward_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double total_inference_ms = 0.0;
    double audio_duration_ms = 0.0;
    double rtf = 0.0;
    int64_t memory_bytes_before = 0;
    int64_t memory_bytes_after = 0;
};

inline int64_t get_rss_bytes() {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                 (task_info_t)&info, &count);
    if (kr == KERN_SUCCESS) {
        return static_cast<int64_t>(info.resident_size);
    }
    return 0;
}

inline double mach_time_to_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t timebase = {0, 0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t elapsed = end - start;
    // Convert to nanoseconds then to milliseconds
    double ns = static_cast<double>(elapsed) * timebase.numer / timebase.denom;
    return ns / 1e6;
}
