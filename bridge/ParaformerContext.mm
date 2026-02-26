#import <Foundation/Foundation.h>
#include <vector>
#import "ParaformerContext.h"
#import "ASRBridgeResult.h"
#include "paraformer_engine.h"
#include "inference_metrics.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

@interface ParaformerContext () {
    ParaformerEngine *engine;
    BOOL initialized;
}
@end

@implementation ParaformerContext

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        initialized = NO;
        try {
            std::string base_path([modelPath UTF8String]);
            engine = new ParaformerEngine();

            bool success = engine->initialize(base_path);
            if (success) {
                initialized = YES;
                NSLog(@"ParaformerContext: Engine initialized successfully.");
            } else {
                NSLog(@"ParaformerContext: Engine initialization failed.");
            }
        } catch (const std::exception& e) {
            NSLog(@"ParaformerContext init exception: %s", e.what());
        } catch (...) {
            NSLog(@"ParaformerContext init: unknown exception");
        }
    }
    return self;
}

- (void)dealloc {
    delete engine;
}

- (ASRBridgeResult *)transcribeDataWithMetrics:(NSData *)pcmData {
    ASRBridgeResult *bridgeResult = [[ASRBridgeResult alloc] init];

    if (!initialized || !engine) {
        bridgeResult.text = @"Error: Paraformer engine not initialized.";
        return bridgeResult;
    }

    try {
        NSUInteger sampleCount = [pcmData length] / sizeof(float);
        const float* rawPtr = (const float *)[pcmData bytes];
        std::vector<float> pcm_data(rawPtr, rawPtr + sampleCount);

        InferenceMetrics metrics;
        std::string result = engine->transcribe(pcm_data, metrics);

        bridgeResult.text = [NSString stringWithUTF8String:result.c_str()];
        bridgeResult.fbankTimeMs = metrics.fbank_time_ms;
        bridgeResult.lfrCmvnTimeMs = metrics.lfr_cmvn_time_ms;
        bridgeResult.onnxForwardTimeMs = metrics.onnx_forward_time_ms;
        bridgeResult.decodeTimeMs = metrics.decode_time_ms;
        bridgeResult.totalInferenceMs = metrics.total_inference_ms;
        bridgeResult.audioDurationMs = metrics.audio_duration_ms;
        bridgeResult.rtf = metrics.rtf;
        bridgeResult.memoryBytesUsed = metrics.memory_bytes_after - metrics.memory_bytes_before;

        return bridgeResult;
    } catch (const Ort::Exception& e) {
        bridgeResult.text = [NSString stringWithFormat:@"ONNX Error: %s", e.what()];
        return bridgeResult;
    } catch (const std::exception& e) {
        bridgeResult.text = [NSString stringWithFormat:@"Error: %s", e.what()];
        return bridgeResult;
    } catch (...) {
        bridgeResult.text = @"Error: Unknown exception during Paraformer transcription.";
        return bridgeResult;
    }
}

- (void)startStream {
    if (engine) {
        engine->start_stream();
    }
}

- (NSString *)feedChunk:(NSData *)pcmData {
    if (!initialized || !engine) return @"";

    try {
        NSUInteger sampleCount = [pcmData length] / sizeof(float);
        const float* rawPtr = (const float *)[pcmData bytes];
        std::vector<float> chunk(rawPtr, rawPtr + sampleCount);

        std::string result = engine->feed_chunk(chunk);
        return [NSString stringWithUTF8String:result.c_str()];
    } catch (...) {
        return @"";
    }
}

- (ASRBridgeResult *)endStreamWithMetrics {
    ASRBridgeResult *bridgeResult = [[ASRBridgeResult alloc] init];

    if (!initialized || !engine) {
        bridgeResult.text = @"Error: Paraformer engine not initialized.";
        return bridgeResult;
    }

    try {
        InferenceMetrics metrics;
        std::string result = engine->end_stream(metrics);

        bridgeResult.text = [NSString stringWithUTF8String:result.c_str()];
        bridgeResult.fbankTimeMs = metrics.fbank_time_ms;
        bridgeResult.lfrCmvnTimeMs = metrics.lfr_cmvn_time_ms;
        bridgeResult.onnxForwardTimeMs = metrics.onnx_forward_time_ms;
        bridgeResult.decodeTimeMs = metrics.decode_time_ms;
        bridgeResult.totalInferenceMs = metrics.total_inference_ms;
        bridgeResult.audioDurationMs = metrics.audio_duration_ms;
        bridgeResult.rtf = metrics.rtf;
        bridgeResult.memoryBytesUsed = metrics.memory_bytes_after - metrics.memory_bytes_before;

        return bridgeResult;
    } catch (...) {
        bridgeResult.text = @"Error: Unknown exception during Paraformer stream end.";
        return bridgeResult;
    }
}

@end
