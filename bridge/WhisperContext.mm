//
//  WhisperContext.mm
//  WhisperIOS
//
//  Created by Mingqian Yu on 1/13/26.
//

#import <Foundation/Foundation.h>
#include <vector>
#import "WhisperContext.h"
#include "whisper_engine.h"

@interface WhisperContext () {
    WhisperEngine *engine;
}
@end

@implementation WhisperContext

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        std::string model_path([modelPath UTF8String]);
        engine = new WhisperEngine();
        
        std::string encoder_path = model_path + "/encoder_small_fp16.onnx";
        std::string decoder_path = model_path + "/decoder_small_fp16.onnx";
        std::string vocab_path = model_path + "/vocab.json";
        engine->initialize(encoder_path, decoder_path, vocab_path);
    }
    return self;
}

- (void)dealloc {
    delete engine;
}

// 6. 实现推理方法
- (NSString *)transcribe:(NSArray<NSNumber *> *)pcmData {
    std::vector<float> pcm_data;
    pcm_data.reserve([pcmData count]);
    for (NSNumber *num in pcmData) {
        pcm_data.push_back([num floatValue]);
    }
    
    std::string result = engine->transcribe(pcm_data);
    return [NSString stringWithUTF8String:result.c_str()];
}

@end
