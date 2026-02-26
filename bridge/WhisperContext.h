//
//  WhisperContext.h
//  FunASR-iOS
//
//  Created by Mingqian Yu on 1/13/26.
//

#ifndef WhisperContext_h
#define WhisperContext_h

#import <Foundation/Foundation.h>

@interface WhisperContext : NSObject

- (instancetype)initWithModelPath:(NSString *)modelPath;

/// Transcribe audio from raw float PCM data passed as NSData (zero-copy from Swift)
- (NSString *)transcribeData:(NSData *)pcmData;

@end

#endif /* WhisperContext_h */
