from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import subprocess, asyncio, os, json, time, boto3, uuid
import google.generativeai as genai
from PIL import Image
import cv2, librosa, numpy as np
import edge_tts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ───
AWS_REGION = "ap-south-1"
S3_BUCKET = "aiforbharat-dubbing"
AWS_ACCESS_KEY = "AKIAXUK5NBCHADT4CPAW"        # your key from notebook
AWS_SECRET_KEY = "l/hTO9IcNv8TNUxNJcZIRtg4WJ2g36n+iusXytCU"  # your secret from notebook
GEMINI_API_KEYS = [
#    "AIzaSyBXuzjgOGgTsPnlk5E6KS6ul--1FdUu5YA",
    "AIzaSyAuKez3qCdal_6s3ieQCwvdAa_Q64A_zBo",
 #   "AIzaSyDH3RZsQTfMag8h6b0OU0DKI9QuA8Xa4uk",
  #  "AIzaSyCApGxL42DeUb7y4kMqZNuT7JG_YLGfOEs",
]

# ─── AWS CLIENTS ───
s3 = boto3.client('s3', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
transcribe_client = boto3.client('transcribe', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
translate_client = boto3.client('translate', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
polly_client = boto3.client('polly', region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# ─── GEMINI SETUP ───
current_key_index = 0

def get_gemini_model():
    global current_key_index
    genai.configure(api_key=GEMINI_API_KEYS[current_key_index])
    return genai.GenerativeModel("gemini-2.5-flash")

def call_gemini(content, retry=True):
    global current_key_index
    model = get_gemini_model()
    try:
        response = model.generate_content(content)
        return response.text.strip()
    except Exception as e:
        error_str = str(e).lower()
        if ('quota' in error_str or 'limit' in error_str or 'exhausted' in error_str) and retry:
            current_key_index += 1
            if current_key_index < len(GEMINI_API_KEYS):
                return call_gemini(content, retry=True)
            else:
                raise Exception("All Gemini API keys exhausted")
        else:
            raise e

# ─── JOB STORE ───
jobs = {}

POLLY_VOICES = {
    'hi': ('Kajal', 'neural', 'hi-IN'),
    'en': ('Kajal', 'neural', 'en-IN'),
}
EDGE_VOICES = {
    'ta': 'ta-IN-ValluvarNeural',
    'te': 'te-IN-ShrutiNeural',
    'bn': 'bn-IN-BashkarNeural',
    'mr': 'mr-IN-AarohiNeural',
    'hi': 'hi-IN-MadhurNeural',
    'en': 'en-IN-PrabhatNeural',
}
TRANSCRIBE_LANG_OPTIONS = ['hi-IN', 'ta-IN', 'te-IN', 'en-IN', 'en-US']

# ════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════
@app.get("/health")
async def health():
    return {"status": "ok", "message": "CreatorMentor API running"}

# ════════════════════════════════════════
# ANALYZE ENDPOINTS
# ════════════════════════════════════════
@app.post("/api/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": 5}

    tmp_path = f"/tmp/{job_id}_{file.filename}"
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    background_tasks.add_task(run_analysis, job_id, tmp_path, file.filename)
    return {"job_id": job_id}


@app.get("/api/analyze/{job_id}")
async def get_analysis(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]


def run_analysis(job_id: str, video_path: str, filename: str):
    try:
        jobs[job_id]["progress"] = 10

        # Video metadata
        result = subprocess.run(
            f"ffprobe -v error -select_streams v:0 "
            f"-show_entries stream=width,height,duration -of json '{video_path}'",
            shell=True, capture_output=True, text=True)
        probe = json.loads(result.stdout)['streams'][0]
        width, height = int(probe['width']), int(probe['height'])
        duration = float(probe.get('duration', 0))
        is_vertical = (height / width) >= 1.7

        jobs[job_id]["progress"] = 20

        # Audio analysis
        audio_path = f"/tmp/{job_id}_audio.wav"
        subprocess.run(
            f"ffmpeg -i '{video_path}' -vn -acodec pcm_s16le -ar 22050 -ac 1 {audio_path} -y",
            shell=True, capture_output=True)

        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        avg_energy = float(np.mean(rms))
        max_energy = float(np.max(rms))

        low_energy_zones = []
        in_low, zone_start = False, 0
        for t, r in zip(times, rms):
            if r < max_energy * 0.20 and not in_low:
                in_low, zone_start = True, float(t)
            elif r >= max_energy * 0.20 and in_low:
                in_low = False
                if float(t) - zone_start > 1.5:
                    low_energy_zones.append((round(zone_start, 1), round(float(t), 1)))

        energy_per_second = {}
        for t, r in zip(times, rms):
            sec = int(t)
            if sec not in energy_per_second:
                energy_per_second[sec] = []
            energy_per_second[sec].append(float(r))
        energy_timeline = {sec: round(float(np.mean(vals)), 4) for sec, vals in energy_per_second.items()}

        jobs[job_id]["progress"] = 35

        # Extract frames
        frames_folder = f"/tmp/{job_id}_frames"
        os.makedirs(frames_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        MAX_FRAMES = 20
        target_frames = min(MAX_FRAMES, max(1, int(duration)))
        frame_step = max(1, total_frames_count // target_frames)
        count, saved = 0, 0
        frame_timestamps = []

        while True:
            ret, frame = cap.read()
            if not ret or saved >= MAX_FRAMES:
                break
            if count % frame_step == 0:
                path = f"{frames_folder}/frame_{saved:04d}.jpg"
                cv2.imwrite(path, frame)
                frame_timestamps.append(round(count / fps, 1))
                saved += 1
            count += 1
        cap.release()

        jobs[job_id]["progress"] = 50

        # Gemini Vision
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
        all_images = [Image.open(f"{frames_folder}/{fname}") for fname in frame_files]

        audio_context = []
        for ts in frame_timestamps:
            sec = int(ts)
            energy = energy_timeline.get(sec, avg_energy)
            level = "high" if energy > avg_energy * 1.5 else "low" if energy < avg_energy * 0.5 else "medium"
            audio_context.append({"timestamp": ts, "audio_energy": level})

        FRAME_PROMPT = f"""You are the world's leading viral content strategist with 10M+ reels analyzed.
I'm giving you {len(all_images)} frames from a creator's reel.
Frame timestamps: {frame_timestamps}
Audio energy per frame: {json.dumps(audio_context)}

CRITICAL RULES:
- Frame scores HONEST and STRICT. Most amateur reels score 2-5.
- Score 8-10 ONLY if frame would genuinely stop a scrolling user
- Score 1-3 for no face, no energy, dark/blurry frames

Return ONLY this JSON array, no markdown, no extra text:
[{{"timestamp":0.0,"has_face":true,"face_expression":"excited/happy/neutral/talking/surprised/none","has_text_overlay":false,"text_content":"","visual_energy":"dead/low/medium/high/explosive","composition":"good/average/poor","scroll_risk":"low/medium/high/critical","scroll_risk_reason":"1 sentence reason","best_thing":"best thing about frame","brightness":"dark/normal/bright","movement":"static/slow/fast","frame_score":5}}]"""

        raw = call_gemini([FRAME_PROMPT] + all_images)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        all_frame_analyses = json.loads(raw)

        jobs[job_id]["progress"] = 65

        # Calculate metrics
        total = len(all_frame_analyses)
        hook_frames = [f for f in all_frame_analyses if f.get('timestamp', 0) <= 3.0]
        hook_has_face = any(f.get('has_face') for f in hook_frames)
        hook_has_text = any(f.get('has_text_overlay') for f in hook_frames)
        hook_avg_score = float(np.mean([f.get('frame_score', 3) for f in hook_frames])) if hook_frames else 3.0
        hook_energy = any(f.get('visual_energy') in ['high', 'explosive'] for f in hook_frames)
        face_count = sum(1 for f in all_frame_analyses if f.get('has_face'))
        face_percentage = (face_count / total * 100) if total > 0 else 0
        text_count = sum(1 for f in all_frame_analyses if f.get('has_text_overlay'))
        text_percentage = (text_count / total * 100) if total > 0 else 0
        critical_frames = [f for f in all_frame_analyses if f.get('scroll_risk') in ['high', 'critical']]
        dropoff_timestamps = [(f.get('timestamp'), f.get('scroll_risk_reason', '')) for f in critical_frames]
        avg_frame_score = float(np.mean([f.get('frame_score', 3) for f in all_frame_analyses]))
        best_frame = max(all_frame_analyses, key=lambda f: f.get('frame_score', 0))
        worst_frame = min(all_frame_analyses, key=lambda f: f.get('frame_score', 10))

        jobs[job_id]["progress"] = 75

        # Mentor report
        time.sleep(5)
        MENTOR_PROMPT = f"""You are India's top viral content mentor.
Frame analysis summary: best frame at {best_frame.get('timestamp')}s (score {best_frame.get('frame_score')}/10), worst at {worst_frame.get('timestamp')}s (score {worst_frame.get('frame_score')}/10).
Video: {duration:.1f}s, vertical: {is_vertical}, face: {face_percentage:.0f}%, text: {text_percentage:.0f}%, hook face: {hook_has_face}, hook energy: {hook_energy}, low audio zones: {len(low_energy_zones)}.

Write a mentor review under 280 words in paragraphs (no bullet points). Reference India. Then output EXACTLY:
VIRAL_SCORE: [1-100]
HOOK_SCORE: [1-10]
RETENTION_SCORE: [1-10]
ENGAGEMENT_SCORE: [1-10]
PLATFORM_FIT_SCORE: [1-10]
CAPTION_SCORE: [1-10]
AUDIO_SCORE: [1-10]"""

        mentor_text = call_gemini(MENTOR_PROMPT)
        score_keys = ['VIRAL_SCORE','HOOK_SCORE','RETENTION_SCORE','ENGAGEMENT_SCORE',
                      'PLATFORM_FIT_SCORE','CAPTION_SCORE','AUDIO_SCORE']
        lines = mentor_text.split('\n')
        score_lines = [l for l in lines if any(k in l for k in score_keys)]
        review_lines = [l for l in lines if not any(k in l for k in score_keys)]
        review_text = '\n'.join(review_lines).strip()
        scores = {}
        for sl in score_lines:
            if ':' in sl:
                key, val = sl.split(':', 1)
                digits = ''.join(filter(str.isdigit, val))
                if digits:
                    scores[key.strip()] = int(digits)
        for k in score_keys:
            if k not in scores:
                scores[k] = 5

        jobs[job_id]["progress"] = 88

        # India strategy
        time.sleep(5)
        INDIA_PROMPT = f"""You are a digital growth strategist for Indian creators.
Viral score: {scores.get('VIRAL_SCORE')}/100, face: {face_percentage:.0f}%, captions: {text_percentage:.0f}%, duration: {duration:.1f}s.
Answer in exactly 3 short sentences: 1) Which Indian audience will connect most? 2) Which language to dub into first and how many million people it unlocks? 3) Which platform to post on first and why?"""

        india_recommendation = call_gemini(INDIA_PROMPT)

        viral_score = scores.get('VIRAL_SCORE', 50)

        jobs[job_id] = {
            "status": "done",
            "progress": 100,
            "result": {
                "filename": filename,
                "score": viral_score,
                "metrics": {
                    "hookPower": scores.get('HOOK_SCORE', 5),
                    "retention": scores.get('RETENTION_SCORE', 5),
                    "engagement": scores.get('ENGAGEMENT_SCORE', 5),
                    "platformFit": scores.get('PLATFORM_FIT_SCORE', 5),
                    "captions": scores.get('CAPTION_SCORE', 5),
                    "audio": scores.get('AUDIO_SCORE', 5),
                },
                "formatChecks": [
                    {"label": "9:16 aspect ratio", "passed": is_vertical},
                    {"label": "Under 90 seconds", "passed": duration <= 90},
                    {"label": "Face in hook", "passed": hook_has_face},
                    {"label": "Text overlay present", "passed": hook_has_text},
                    {"label": "Audio energy good", "passed": len(low_energy_zones) < 3},
                ],
                "dropOffMoments": [{"timestamp": ts, "reason": r} for ts, r in dropoff_timestamps[:5]],
                "energyTimeline": [f.get('visual_energy', 'low') for f in all_frame_analyses],
                "mentorAnalysis": review_text,
                "indiaStrategy": [s.strip() for s in india_recommendation.split('\n') if s.strip()][:3],
                "currentViews": int(viral_score * 300),
                "potentialViews": int(min(100, viral_score + 30) * 500),
                "duration": duration,
                "isVertical": is_vertical,
            }
        }

        # Cleanup
        try:
            os.remove(video_path)
            os.remove(audio_path)
            import shutil
            shutil.rmtree(frames_folder, ignore_errors=True)
        except:
            pass

    except Exception as e:
        jobs[job_id] = {"status": "error", "progress": 0, "message": str(e)}


# ════════════════════════════════════════
# DUB ENDPOINTS
# ════════════════════════════════════════
@app.post("/api/dub")
async def dub_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("hi"),
    add_captions: str = Form("true")
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "uploading", "progress": 5}

    tmp_path = f"/tmp/{job_id}_{file.filename}"
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    add_captions_bool = add_captions.lower() == "true"
    background_tasks.add_task(run_dubbing, job_id, tmp_path, target_language, add_captions_bool, file.filename)
    return {"job_id": job_id}


@app.get("/api/dub/{job_id}")
async def get_dub(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]


def run_dubbing(job_id: str, video_path: str, target_language: str, add_captions: bool, original_filename: str):
    try:
        timestamp = int(time.time())
        jobs[job_id] = {"status": "transcribing", "progress": 15}

        audio_path = f"/tmp/{job_id}_audio.wav"
        subprocess.run(
            f"ffmpeg -i '{video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y",
            shell=True, capture_output=True)

        s3_audio_key = f"input/audio_{timestamp}.wav"
        s3.upload_file(audio_path, S3_BUCKET, s3_audio_key)
        s3_audio_uri = f"s3://{S3_BUCKET}/{s3_audio_key}"

        job_name = f"dub_{timestamp}"
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_audio_uri},
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
            time.sleep(5)

        if state == 'FAILED':
            raise Exception("Transcribe failed")

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
                resp = translate_client.translate_text(
                    Text=seg['text'],
                    SourceLanguageCode='auto',
                    TargetLanguageCode=target_language)
                translated_text = resp['TranslatedText']
            except:
                translated_text = seg['text']
            translated_segments.append({
                "start": seg["start"], "end": seg["end"],
                "original": seg["text"], "translated": translated_text
            })

        jobs[job_id] = {"status": "generating", "progress": 65}

        chunks_dir = f"/tmp/{job_id}_chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        use_polly = target_language in POLLY_VOICES
        chunk_files = []

        async def edge_tts_chunk(text, filename, lang):
            voice = EDGE_VOICES.get(lang, 'en-IN-PrabhatNeural')
            await edge_tts.Communicate(text, voice=voice).save(filename)

        for i, seg in enumerate(translated_segments):
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
                    asyncio.run(edge_tts_chunk(seg['translated'], raw_chunk, target_language))

                result = subprocess.run(
                    f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '{raw_chunk}'",
                    shell=True, capture_output=True, text=True)
                tts_duration = float(result.stdout.strip())
                target_duration = max(0.1, seg["end"] - seg["start"])
                speed = max(0.5, min(2.0, tts_duration / target_duration))

                subprocess.run(
                    f"ffmpeg -i '{raw_chunk}' -filter:a 'atempo={speed:.4f}' '{fitted_chunk}' -y",
                    shell=True, capture_output=True)
                chunk_files.append({"file": fitted_chunk, "start": seg["start"], "duration": target_duration})
            except Exception as e:
                print(f"Chunk {i} error: {e}")

        jobs[job_id]["progress"] = 80

        result = subprocess.run(
            f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '{video_path}'",
            shell=True, capture_output=True, text=True)
        total_duration = float(result.stdout.strip())

        silent_base = f"/tmp/{job_id}_silent.mp3"
        subprocess.run(
            f"ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t {total_duration} {silent_base} -y",
            shell=True, capture_output=True)

        inputs = f"-i '{silent_base}'"
        filter_parts = []
        for i, chunk in enumerate(chunk_files):
            delay_ms = int(chunk["start"] * 1000)
            inputs += f" -i '{chunk['file']}'"
            filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

        mix_inputs = "".join([f"[a{i}]" for i in range(len(chunk_files))])
        filter_complex = ";".join(filter_parts) + f";[0:a]{mix_inputs}amix=inputs={len(chunk_files)+1}[out]"
        final_audio = f"/tmp/{job_id}_final_audio.mp3"

        subprocess.run(
            f"ffmpeg {inputs} -filter_complex \"{filter_complex}\" -map \"[out]\" -t {total_duration} {final_audio} -y",
            shell=True)

        dubbed_video = f"/tmp/{job_id}_dubbed.mp4"
        subprocess.run(
            f"ffmpeg -i '{video_path}' -i {final_audio} -map 0:v -map 1:a -c:v copy -shortest {dubbed_video} -y",
            shell=True)

        output_video = dubbed_video
        if add_captions and translated_segments:
            def fmt(s):
                h = int(s // 3600); m = int((s % 3600) // 60)
                sec = int(s % 60); ms = int((s % 1) * 1000)
                return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

            srt_path = f"/tmp/{job_id}_subs.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(translated_segments):
                    f.write(f"{i+1}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['translated']}\n\n")

            captioned_video = f"/tmp/{job_id}_captioned.mp4"
            subprocess.run(
                f"ffmpeg -i {dubbed_video} "
                f"-vf \"subtitles={srt_path}:force_style='FontSize=11,FontName=Arial,"
                f"PrimaryColour=&H00ffffff,OutlineColour=&H00000000,"
                f"BackColour=&H80000000,Outline=1,Shadow=1,Alignment=2,MarginV=30'\" "
                f"-c:a copy {captioned_video} -y",
                shell=True)
            output_video = captioned_video

        out_key = f"output/{job_id}_{target_language}.mp4"
        s3.upload_file(output_video, S3_BUCKET, out_key)

        download_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': out_key},
            ExpiresIn=3600
        )

        base_name = original_filename.rsplit('.', 1)[0]
        jobs[job_id] = {
            "status": "done",
            "progress": 100,
            "result": {
                "downloadUrl": download_url,
                "s3Url": f"s3://{S3_BUCKET}/{out_key}",
                "filename": f"{base_name}_{target_language}.mp4",
                "language": target_language,
            }
        }

        # Cleanup
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
