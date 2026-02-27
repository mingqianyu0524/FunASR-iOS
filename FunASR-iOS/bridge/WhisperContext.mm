//
//  WhisperContext.mm
//  FunASR-iOS
//
//  Created by Mingqian Yu on 1/13/26.
//

#import <Foundation/Foundation.h>
#include <vector>
#import "WhisperContext.h"
#include "whisper_engine.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

@interface WhisperContext () {
    WhisperEngine *engine;
    BOOL initialized;
}
@end

@implementation WhisperContext

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        initialized = NO;
        try {
            std::string model_path([modelPath UTF8String]);
            engine = new WhisperEngine();

            std::string encoder_path = model_path + "/encoder_small_fp16.onnx";
            std::string decoder_path = model_path + "/decoder_small_fp16.onnx";
            std::string vocab_path = model_path + "/vocab.json";
            std::string embedder_path = model_path + "/vocab_embedding_fp32.bin";

            bool success = engine->initialize(encoder_path, decoder_path, vocab_path, embedder_path);
            if (success) {
                initialized = YES;
                NSLog(@"✅ WhisperContext: Engine initialized successfully.");
            } else {
                NSLog(@"❌ WhisperContext: Engine initialization failed.");
            }
        } catch (const std::exception& e) {
            NSLog(@"❌ WhisperContext init exception: %s", e.what());
        } catch (...) {
            NSLog(@"❌ WhisperContext init: unknown exception");
        }
    }
    return self;
}

- (void)dealloc {
    delete engine;
}

- (NSString *)transcribeData:(NSData *)pcmData {
    if (!initialized || !engine) {
        NSLog(@"❌ WhisperContext: Engine not initialized, cannot transcribe.");
        return @"Error: Engine not initialized.";
    }

    try {
        // Zero-copy: directly interpret NSData bytes as float array
        NSUInteger sampleCount = [pcmData length] / sizeof(float);
        const float* rawPtr = (const float *)[pcmData bytes];

        // Construct vector from raw pointer (single copy, no NSNumber boxing)
        std::vector<float> pcm_data(rawPtr, rawPtr + sampleCount);

        NSLog(@"🧠 WhisperContext: Running transcribe with %lu samples...", (unsigned long)sampleCount);
        std::string result = engine->transcribe(pcm_data);
        return [NSString stringWithUTF8String:result.c_str()];
    } catch (const Ort::Exception& e) {
        NSLog(@"❌ WhisperContext ONNX Runtime error: %s", e.what());
        return [NSString stringWithFormat:@"ONNX Error: %s", e.what()];
    } catch (const std::exception& e) {
        NSLog(@"❌ WhisperContext exception: %s", e.what());
        return [NSString stringWithFormat:@"Error: %s", e.what()];
    } catch (...) {
        NSLog(@"❌ WhisperContext: unknown exception during transcribe");
        return @"Error: Unknown exception during transcription.";
    }
}

@end
