# Speech To Text

Project to combine VAD, Speaker Diarization, Speech Recognition together.

## Getting Started

As DeepSpeech pre-trained English model is too big to commit to git. You could download from (https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz)

Please put the downloaded model files into folder **speech-to-text/deepspeech/models**

```shell
# Create and activate a virtual environment
python3 -m venv speech-to-text/env
source speech-to-text/env/bin/activate

# Install prerequisites
pip3 install -r requirements.txt

# Speech To Text
python3 speech_to_text.py --audio=wavs/test2.wav
```

It will output the txt file with speakers and speech text, side by side the wav file 

## Overview

It's just to combine speaker diarization and speech recognization together. 

Only support 16k sample rate PCM wav file. You can use ffmpeg to convert sound file format. i.e.
```
ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 output.wav
```
**Main flows**
1. Filter out silence frames and break down to segments with webrtcvad.

2. Generate utterances spec with librosa

3. Get utterances features with ghostvlad

4. Classify features with uisrnn model

5. Recognize speeches segment by segment with deepspeech

*It might takes long period(tens minutes) if the wav is too big.(seems uisrnn part takes the longest)
The test wavs in the wavs folder are from movie sound clips. The speech accuracy is not perfect, it might relative to the pretrained deepspeech model and the background noise*

### Prerequisites

- pytorch
- keras
- tensorflow
- pyaudio
- librosa
- webrtcvad
- deepspeech

## References

- [DeepSpeech](https://github.com/mozilla/DeepSpeech)
- [Speaker-Diarization](https://github.com/taylorlu/Speaker-Diarization)
- [uis-rnn](https://github.com/google/uis-rnn)
- [py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
- [librosa](https://github.com/librosa/librosa)
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
- [kaldi](https://github.com/kaldi-asr/kaldi)
- [awesome-diarization](https://github.com/wq2012/awesome-diarization)
