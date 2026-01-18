//
//  WhisperContext.h
//  WhisperIOS
//
//  Created by Mingqian Yu on 1/13/26.
//

#ifndef WhisperContext_h
#define WhisperContext_h

#import <Foundation/Foundation.h>

@interface WhisperContext : NSObject

- (instancetype)initWithModelPath:(NSString *)modelPath;

- (NSString *)transcribe:(NSArray<NSNumber *> *)pcmData;

@end

#endif /* WhisperContext_h */
