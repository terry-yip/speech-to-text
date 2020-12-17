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

### Tip

Following are the libs version installed in my env, just for your reference.

- absl-py 0.7.1
- astor 0.8.0
- astroid 2.4.2
- audioread 2.1.8
- cffi 1.12.3
- decorator 4.4.0
- deepspeech 0.5.1
- gast 0.2.2
- google-pasta 0.1.7
- grpcio 1.22.0
- h5py 2.9.0
- isort 5.6.4
- joblib 0.13.2
- Keras 2.2.4
- Keras-Applications 1.0.8
- Keras-Preprocessing 1.1.0
- lazy-object-proxy 1.4.3
- librosa 0.7.0
- llvmlite 0.29.0
- Markdown 3.1.1
- mccabe 0.6.1
- numba 0.45.0
- numpy 1.16.4
- pip 19.2
- protobuf 3.9.0
- PyAudio 0.2.11
- pycparser 2.19
- pylint 2.6.0
- PyYAML 5.1.1
- resampy 0.2.1
- scikit-learn 0.21.2
- scipy 1.3.0
- setuptools 41.0.1
- six 1.12.0
- SoundFile 0.10.2
- tensorboard 1.14.0
- tensorflow 1.14.0
- tensorflow-estimator 1.14.0
- termcolor 1.1.0
- toml 0.10.1
- torch 1.1.0.post2
- typed-ast 1.4.1
- webrtcvad 2.0.10
- Werkzeug 0.15.5
- wheel 0.33.4
- wrapt 1.11.2
