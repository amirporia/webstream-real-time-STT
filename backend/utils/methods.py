import re
import requests
import scipy.io.wavfile as wav
import io

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def trim_last_incomplete_confirmed_sentence(confirmed_transciption, confirm_offset_time):

    if len(confirmed_transciption) > 0:
        idx = len(confirmed_transciption) - 1
        while idx != -1:
            if confirm_offset_time == -1:
                if confirmed_transciption[idx][2].strip().endswith(('.', '?', '!')):
                    if idx != len(confirmed_transciption) - 1:
                        confirmed_transciption = confirmed_transciption[:idx + 1]
                    break
            elif confirmed_transciption[idx][1] == confirm_offset_time:
                if idx != len(confirmed_transciption) - 1:
                    confirmed_transciption = confirmed_transciption[:idx + 1]
                break
            idx -= 1

        if idx == -1:
            return [] 

    return confirmed_transciption


def confirmation_process(non_confirmed_transcription, tokenize_transcription, confirmed_transciption, confirm_offset_time):

    sliced_tokenize_transcription = [(a,b," ".join(t.split())) for a,b,t in tokenize_transcription]


    if len(non_confirmed_transcription) == 0 or (len(sliced_tokenize_transcription) > 0 and remove_punctuation(non_confirmed_transcription[0][2]) != remove_punctuation(sliced_tokenize_transcription[0][2])):
        non_confirmed_transcription = sliced_tokenize_transcription

    elif len(non_confirmed_transcription) > 0:

        if len(confirmed_transciption) > 0:
            if (confirm_offset_time != -1 or (confirm_offset_time == -1 and not confirmed_transciption[-1][2].strip().endswith(('.', '?', '!')))):
                for ct_word in reversed(confirmed_transciption):
                    if ct_word[2].strip().endswith(('.', '?', '!')):
                        break
                    if (ct_word[1] - confirm_offset_time) >= 5:
                        confirm_offset_time = ct_word[1]
                        break
                    
        confirmed_transciption = trim_last_incomplete_confirmed_sentence(confirmed_transciption, confirm_offset_time)

        idx = 0
        for slice_idx in range(len(non_confirmed_transcription)):
            if non_confirmed_transcription[slice_idx][0] >= confirm_offset_time:
                idx = slice_idx
                break

        while idx < min(len(non_confirmed_transcription), len(sliced_tokenize_transcription)):
            if remove_punctuation((non_confirmed_transcription[idx][2]).lower()) == remove_punctuation((sliced_tokenize_transcription[idx][2]).lower()):
                confirmed_transciption.append(sliced_tokenize_transcription[idx])
                idx += 1
            else:
                break
        non_confirmed_transcription = sliced_tokenize_transcription

    return non_confirmed_transcription, confirmed_transciption, confirm_offset_time


def sentence_trim_buffer(tokenize_transcription, non_confirmed_transcription, confirmed_transcription, confirm_offset_time, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE):

    if len(confirmed_transcription) == 0:
        return 0, non_confirmed_transcription, confirm_offset_time  # No confirmed sentences to remove
    
    # Find the last confirmed sentence ending with ., ?, or !
    last_sentence = None
    for sentence in reversed(confirmed_transcription):
        if sentence[2].strip().endswith(('.', '?', '!')):
            last_sentence = sentence[2]
            break

    if not last_sentence:
        return 0, non_confirmed_transcription, confirm_offset_time  # No complete sentence found, do not trim buffer
    
    confirmed_words = last_sentence.strip()
    end_time = None
    end_word_idx = -1

    # Find the corresponding end timestamp of the last word in the sentence
    for start, end, word in tokenize_transcription:
        if confirmed_words in word:
            end_time = end

    if end_time is None:
        return 0, non_confirmed_transcription, confirm_offset_time  # No match found, do not trim buffer
    
    for idx in range(len(non_confirmed_transcription)):
        if confirmed_words in non_confirmed_transcription[idx][2]:
            end_word_idx = idx

    if tokenize_transcription[-1][1] == end_time:
        return -1, non_confirmed_transcription[end_word_idx + 1:], -1
    
    # Compute bytes to remove
    bytes_to_remove = int(end_time * sample_rate * bytes_per_sample)

    # Trim buffer  
    return bytes_to_remove, non_confirmed_transcription[end_word_idx + 1:], -1


def threshold_trim_buffer(tokenize_transcription, non_confirmed_transcription, confirmed_transcription, buffer, confirm_offset_time, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE):
    
    non_confirmed_transcription = []
    confirmed_transcription = trim_last_incomplete_confirmed_sentence(confirmed_transcription, confirm_offset_time)
    end_time = 0

    while end_time < 30 and len(tokenize_transcription) > 0:
        last_tr = tokenize_transcription.pop(0)
        confirmed_transcription.append(last_tr)
    
    confirmed_transcription[-1] = (*confirmed_transcription[-1][:2], confirmed_transcription[-1][2] + "..")

    if len(tokenize_transcription) > 0:
        non_confirmed_transcription.extend([(a, b, " ".join(t.split())) for a,b,t in tokenize_transcription])
        return buffer[int(end_time * sample_rate * bytes_per_sample):], confirmed_transcription, non_confirmed_transcription, -1
    
    return bytearray(), confirmed_transcription, non_confirmed_transcription, -1



def model_transcribe(asr, pcm_array, trans):

    transcription = asr.transcribe(pcm_array, init_prompt=trans)
    tokenize_transcription = asr.ts_words(transcription)
    return tokenize_transcription