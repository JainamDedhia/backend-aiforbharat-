# services/dubber.py
import subprocess, asyncio, os, json, time
import edge_tts
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.config import (
    jobs, s3, transcribe_client, translate_client, polly_client,
    S3_BUCKET, POLLY_VOICES, EDGE_VOICES, TRANSCRIBE_LANG_OPTIONS
)

NON_LATIN_LANGS = {'hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa'}

def _get_subtitle_font(lang_code: str) -> str:
    if lang_code in ('hi', 'mr', 'bn', 'pa'):
        return 'Noto Sans Devanagari'
    if lang_code == 'gu':
        return 'Noto Sans Gujarati'
    if lang_code == 'ta':
        return 'Noto Sans Tamil'
    if lang_code == 'te':
        return 'Noto Sans Telugu'
    if lang_code == 'kn':
        return 'Noto Sans Kannada'
    if lang_code == 'ml':
        return 'Noto Sans Malayalam'
    return 'Arial'


def _ensure_noto_fonts():
    result = subprocess.run('fc-list | grep -i noto', shell=True, capture_output=True, text=True)
    if 'Noto' not in result.stdout:
        print('[FONTS] Installing Noto fonts...')
        subprocess.run('apt-get install -y fonts-noto fonts-noto-core fonts-noto-extra > /dev/null 2>&1', shell=True)
        subprocess.run('fc-cache -fv > /dev/null 2>&1', shell=True)
        print('[FONTS] Noto fonts installed.')


def _split_into_word_chunks(segments, words_per_chunk=3):
    """Split subtitle segments into small N-word chunks with proportional timing."""
    srt_chunks = []
    for seg in segments:
        text = (seg.get('subtitle_text') or seg.get('translated') or '').strip()
        if not text:
            continue
        words = text.split()
        if not words:
            continue

        seg_start = seg['start']
        seg_end = seg['end']
        seg_duration = max(0.1, seg_end - seg_start)

        groups = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
        if not groups:
            continue

        chunk_duration = seg_duration / len(groups)
        for i, group_text in enumerate(groups):
            chunk_start = seg_start + i * chunk_duration
            chunk_end = chunk_start + chunk_duration - 0.05
            srt_chunks.append({'start': chunk_start, 'end': chunk_end, 'text': group_text})

    return srt_chunks


def _fmt_srt_time(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _generate_tts_chunk(i, seg, chunks_dir, target_language, use_polly):
    raw_chunk = f"{chunks_dir}/raw_{i}.mp3"
    fitted_chunk = f"{chunks_dir}/fitted_{i}.mp3"
    try:
        if use_polly:
            voice_id, engine, lang_code = POLLY_VOICES[target_language]
            polly_resp = polly_client.synthesize_speech(
                Text=seg['translated'], OutputFormat='mp3',
                VoiceId=voice_id, Engine=engine, LanguageCode=lang_code)
            with open(raw_chunk, 'wb') as f:
                f.write(polly_resp['AudioStream'].read())
        else:
            # EDGE_VOICES lookup — fallback to Hindi if language not found
            voice = EDGE_VOICES.get(target_language)
            if not voice:
                print(f"[TTS] No voice found for '{target_language}', falling back to hi-IN-MadhurNeural")
                voice = 'hi-IN-MadhurNeural'
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(edge_tts.Communicate(seg['translated'], voice=voice).save(raw_chunk))
            loop.close()

        result = subprocess.run(
            f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '{raw_chunk}'",
            shell=True, capture_output=True, text=True)
        tts_duration = float(result.stdout.strip())
        target_duration = max(0.1, seg["end"] - seg["start"])
        speed = max(0.5, min(2.0, tts_duration / target_duration))
        subprocess.run(f"ffmpeg -i '{raw_chunk}' -filter:a 'atempo={speed:.4f}' '{fitted_chunk}' -y", shell=True, capture_output=True)
        return {"index": i, "file": fitted_chunk, "start": seg["start"], "duration": target_duration}
    except Exception as e:
        print(f"[TTS] Chunk {i} error: {e}")
        return None


def run_dubbing(job_id: str, video_path: str, target_language: str, subtitle_language: str, original_filename: str):
    try:
        timestamp = int(time.time())
        jobs[job_id] = {"status": "transcribing", "progress": 15}

        audio_path = f"/tmp/{job_id}_audio.wav"
        subprocess.run(f"ffmpeg -i '{video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y", shell=True, capture_output=True)

        s3_audio_key = f"input/audio_{timestamp}.wav"
        s3.upload_file(audio_path, S3_BUCKET, s3_audio_key)

        job_name = f"dub_{timestamp}"
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': f"s3://{S3_BUCKET}/{s3_audio_key}"},
            MediaFormat='wav',
            IdentifyLanguage=True,
            LanguageOptions=TRANSCRIBE_LANG_OPTIONS,
            OutputBucketName=S3_BUCKET,
            OutputKey=f"transcripts/{job_name}.json"
        )

        while True:
            response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            state = response['TranscriptionJob']['TranscriptionJobStatus']
            if state in ['COMPLETED', 'FAILED']:
                break
            time.sleep(3)

        if state == 'FAILED':
            raise Exception("AWS Transcribe job failed")

        jobs[job_id]["progress"] = 35

        transcript_obj = s3.get_object(Bucket=S3_BUCKET, Key=f"transcripts/{job_name}.json")
        transcript_data = json.loads(transcript_obj['Body'].read())
        items = transcript_data['results']['items']

        raw_segments = []
        current = {"words": [], "start": None, "end": None}
        last_end = 0

        for item in items:
            if item['type'] == 'pronunciation':
                word = item['alternatives'][0]['content']
                start = float(item['start_time'])
                end = float(item['end_time'])
                if current["start"] is None:
                    current["start"] = start
                if start - last_end > 1.5 and current["words"]:
                    current["end"] = last_end
                    raw_segments.append({"start": current["start"], "end": current["end"], "text": " ".join(current["words"])})
                    current = {"words": [word], "start": start, "end": end}
                else:
                    current["words"].append(word)
                current["end"] = end
                last_end = end
            elif item['type'] == 'punctuation' and current["words"]:
                current["words"][-1] += item['alternatives'][0]['content']

        if current["words"]:
            raw_segments.append({"start": current["start"], "end": current["end"], "text": " ".join(current["words"])})

        jobs[job_id] = {"status": "translating", "progress": 50}

        translated_segments = []
        for seg in raw_segments:
            try:
                resp = translate_client.translate_text(Text=seg['text'], SourceLanguageCode='auto', TargetLanguageCode=target_language)
                translated_text = resp['TranslatedText']
            except Exception as e:
                print(f"[Translation Error - Audio]: {e}")
                translated_text = seg['text']

            subtitle_text = ""
            if subtitle_language != 'none':
                if subtitle_language == target_language:
                    subtitle_text = translated_text
                else:
                    try:
                        resp_sub = translate_client.translate_text(Text=seg['text'], SourceLanguageCode='auto', TargetLanguageCode=subtitle_language)
                        subtitle_text = resp_sub['TranslatedText']
                    except Exception as e:
                        print(f"[Translation Error - Subtitle]: {e}")
                        subtitle_text = seg['text']

            translated_segments.append({
                "start": seg["start"], "end": seg["end"],
                "original": seg["text"], "translated": translated_text, "subtitle_text": subtitle_text
            })

        jobs[job_id] = {"status": "generating", "progress": 65}

        chunks_dir = f"/tmp/{job_id}_chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        use_polly = target_language in POLLY_VOICES

        chunk_results = [None] * len(translated_segments)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_generate_tts_chunk, i, seg, chunks_dir, target_language, use_polly): i for i, seg in enumerate(translated_segments)}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    chunk_results[result["index"]] = result

        chunk_files = [c for c in chunk_results if c is not None]
        chunk_files.sort(key=lambda x: x["index"])

        jobs[job_id]["progress"] = 80

        result = subprocess.run(
            f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '{video_path}'",
            shell=True, capture_output=True, text=True)
        total_duration = float(result.stdout.strip())

        silent_base = f"/tmp/{job_id}_silent.mp3"
        subprocess.run(f"ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t {total_duration} {silent_base} -y", shell=True, capture_output=True)

        inputs = f"-i '{silent_base}'"
        filter_parts = []
        for i, chunk in enumerate(chunk_files):
            delay_ms = int(chunk["start"] * 1000)
            inputs += f" -i '{chunk['file']}'"
            filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

        mix_inputs = "".join([f"[a{i}]" for i in range(len(chunk_files))])
        filter_complex = ";".join(filter_parts) + f";[0:a]{mix_inputs}amix=inputs={len(chunk_files)+1}[out]"
        final_audio = f"/tmp/{job_id}_final_audio.mp3"
        subprocess.run(f"ffmpeg {inputs} -filter_complex \"{filter_complex}\" -map \"[out]\" -t {total_duration} {final_audio} -y", shell=True)

        dubbed_video = f"/tmp/{job_id}_dubbed.mp4"
        subprocess.run(f"ffmpeg -i '{video_path}' -i {final_audio} -map 0:v -map 1:a -c:v copy -shortest {dubbed_video} -y", shell=True)

        # ── Subtitles ──
        output_video = dubbed_video

        if subtitle_language != 'none' and translated_segments:
            if subtitle_language in NON_LATIN_LANGS:
                _ensure_noto_fonts()

            word_chunks = _split_into_word_chunks(translated_segments, words_per_chunk=3)

            srt_path = f"/tmp/{job_id}_subs.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(word_chunks):
                    if chunk['text'].strip():
                        f.write(f"{i+1}\n{_fmt_srt_time(chunk['start'])} --> {_fmt_srt_time(chunk['end'])}\n{chunk['text']}\n\n")

            captioned_video = f"/tmp/{job_id}_captioned.mp4"
            font_name = _get_subtitle_font(subtitle_language)

            # ── Font size: fixed small value, NOT % of height ──
            # ASS/ffmpeg FontSize is in "points" relative to a 288p reference frame.
            # For a 1280px tall video, FontSize=18 renders as ~80px on screen — good size.
            # Rule of thumb: use 16-20 for vertical reels, never go above 24.
            font_size = 18

            print(f"[CAPTIONS] font={font_name} size={font_size} lang={subtitle_language} chunks={len(word_chunks)}")

            srt_escaped = srt_path.replace("'", "\\'")
            cmd = (
                f"ffmpeg -i '{dubbed_video}' "
                f"-vf \"subtitles='{srt_escaped}':force_style='"
                f"FontName={font_name},"
                f"FontSize={font_size},"
                f"PrimaryColour=&H00FFFFFF,"
                f"OutlineColour=&H00000000,"
                f"BackColour=&H80000000,"
                f"Outline=2,"
                f"Shadow=0,"
                f"Alignment=2,"
                f"MarginV=50,"
                f"Bold=1'\" "
                f"-c:a copy '{captioned_video}' -y"
            )
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                output_video = captioned_video
                print("[CAPTIONS] Success")
            else:
                print(f"[CAPTIONS] ffmpeg error: {result.stderr[-500:]}")
                # Fallback: no font name
                cmd_fallback = (
                    f"ffmpeg -i '{dubbed_video}' "
                    f"-vf \"subtitles='{srt_escaped}':force_style='"
                    f"FontSize={font_size},"
                    f"PrimaryColour=&H00FFFFFF,"
                    f"OutlineColour=&H00000000,"
                    f"Outline=2,"
                    f"Alignment=2,"
                    f"MarginV=50'\" "
                    f"-c:a copy '{captioned_video}' -y"
                )
                result2 = subprocess.run(cmd_fallback, shell=True, capture_output=True, text=True)
                output_video = captioned_video if result2.returncode == 0 else dubbed_video
                print(f"[CAPTIONS] Fallback {'succeeded' if result2.returncode == 0 else 'failed — no captions'}")

        out_key = f"output/{job_id}_{target_language}.mp4"
        s3.upload_file(output_video, S3_BUCKET, out_key)
        download_url = s3.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': out_key}, ExpiresIn=3600)

        base_name = original_filename.rsplit('.', 1)[0]
        jobs[job_id] = {
            "status": "done", "progress": 100,
            "result": {
                "downloadUrl": download_url,
                "s3Url": f"s3://{S3_BUCKET}/{out_key}",
                "filename": f"{base_name}_{target_language}.mp4",
                "language": target_language,
            }
        }

        try:
            import shutil
            for p in [video_path, audio_path, silent_base, final_audio, dubbed_video]:
                try: os.remove(p)
                except: pass
            shutil.rmtree(chunks_dir, ignore_errors=True)
        except:
            pass

    except Exception as e:
        jobs[job_id] = {"status": "error", "progress": 0, "message": str(e)}