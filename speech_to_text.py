import sys
import os
import logging
import argparse
import datetime
import numpy as np
import wavTranscriber
import speaker_diarization

# Debug helpers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True,
                        help='Path to the audio file to run (WAV format)')
    args = parser.parse_args()

    # Run VAD on the input file
    waveFile = args.audio
    logger.debug("Loading wav file %s" % waveFile)
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, 3)

    logger.debug("Processing speaker diarization")
    segments = speaker_diarization.diarize(segments)

    f = open(waveFile.replace(".wav", ".txt"), 'w')
    logger.debug("Processing speech recognition")

    # Point to a path containing the pre-trained models & resolve ~ if used
    model_path = os.path.expanduser("deepspeech/models/")
    # Resolve all the paths of model files
    output_graph, alphabet, lm, trie = wavTranscriber.resolve_models(model_path)
    # Load output_graph, alpahbet, lm and trie
    model_retval = wavTranscriber.load_model(output_graph, alphabet, lm, trie)

    inference_time = 0.0
    for i, segment in enumerate(segments):
        # Run deepspeech on the chunk that just completed VAD
        logger.debug("[Speech recognition] Processing chunk %002d" % (i,))
        audio = np.frombuffer(segment.bytes, dtype=np.int16)
        output = wavTranscriber.stt(model_retval[0], audio, sample_rate)
        inference_time += output[1]
        logger.debug("[Speech recognition] Transcript: %s" % output[0])

        f.write("%s - %s Speaker %s: %s\n" % (str(datetime.timedelta(seconds=round(segment.begin, 3)))[:-3],
                                              str(datetime.timedelta(seconds=round(segment.end, 3)))[:-3],
                                              segment.speaker, output[0]))

    # Summary of the files processed
    f.close()
    logger.debug("Saved transcript @: %s" % waveFile.replace(".wav", ".txt"))

    # Extract filename from the full file path
    filename, ext = os.path.split(os.path.basename(waveFile))
    title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'LM Load Time(s)']
    logger.debug("************************************************************************************************************")
    logger.debug("%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
    logger.debug("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))
    logger.debug("************************************************************************************************************")

    print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))
    print("%-30s %-20.3f %-20.3f %-20.3f %-0.3f" % (filename + ext, audio_length, inference_time, model_retval[1], model_retval[2]))


if __name__ == '__main__':
    main(sys.argv[1:])
