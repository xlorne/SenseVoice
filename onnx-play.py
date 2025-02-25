import os
import time
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from utils.model_bin import SenseVoiceSmallONNX

model_path = 'iic/SenseVoiceSmall'

# export model init
model_bin = SenseVoiceSmallONNX(model_path)

# build tokenizer
try:
    from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
    tokenizer = SentencepiecesTokenizer(bpemodel=os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
except:
    tokenizer = None

# inference
wav_or_scp = "./examples/test.mp3"
language_list = [0]
textnorm_list = [15]
t1 = time.time()
res = model_bin(wav_or_scp, language_list, textnorm_list, tokenizer=tokenizer)
print([rich_transcription_postprocess(i) for i in res])
t2 = time.time()
print('time',(t2-t1) * 1000)