#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "knfc.h"
#include <iostream>

void DestroyFbankResult(FbankResult* result) {
    if (result) {
        delete[] result->frames;
        result->frames = nullptr;
        result->num_frames = 0;
    }
}

FbankResult ComputeFbank(const float *waveform, int32_t waveform_size) {
    knf::FbankOptions options;
    options.htk_compat = true;
    options.use_energy = false;

    // Frame options
    options.frame_opts.window_type = "hanning";
    options.frame_opts.dither = 0.0;

    // Mel options
    options.mel_opts.num_bins = 128;

    knf::OnlineFbank fbank(options);
    fbank.AcceptWaveform(options.frame_opts.samp_freq, waveform, waveform_size);
    fbank.InputFinished();

    int32_t num_frames = fbank.NumFramesReady();
    int32_t num_bins = options.mel_opts.num_bins;

    // Allocate memory for the frames
    float* frames = new float[num_frames * num_bins];

    // Fill the frames
    for (int32_t i = 0; i < num_frames; i++) {
        const float* frame = fbank.GetFrame(i);
        std::copy(frame, frame + num_bins, frames + i * num_bins);
    }

    FbankResult result;
    result.frames = frames;
    result.num_frames = num_frames;
    return result;
}
