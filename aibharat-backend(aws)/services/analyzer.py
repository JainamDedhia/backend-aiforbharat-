import subprocess, os, json, time
import concurrent.futures
import cv2, librosa, numpy as np
from PIL import Image
from core.config import jobs, save_analysis
from services.nova import call_nova, call_nova_text


def _analyze_frame_batch(batch_images, batch_timestamps, batch_audio_context):
    FRAME_PROMPT = f"""You are the world's leading viral content strategist — you've personally analyzed over 10 million Instagram Reels, TikToks, and YouTube Shorts. You know exactly why videos go viral and why they flop.

I'm giving you {len(batch_images)} frames extracted from a creator's reel.

Frame timestamps: {batch_timestamps}
Audio energy per frame (from separate analysis): {json.dumps(batch_audio_context)}

TEXT DETECTION — THIS IS THE MOST IMPORTANT RULE:
- Scan the ENTIRE frame carefully — top, middle, bottom, corners
- Look for ANY burned-in text: captions, subtitles, titles, Hindi/Hinglish/English words, numbers
- Text overlays are usually bold, white, yellow, or orange colored text
- Bottom 40% of frame is where most text overlays appear
- If you see ANY words or letters anywhere in the frame — set has_text_overlay to TRUE
- Copy the EXACT text into text_content, even if it is Hindi or mixed language
- NEVER set has_text_overlay to false if there are visible words in the frame

SCORING RULES:
- Frame scores must be HONEST and BRUTALLY STRICT. Most amateur reels score 2-5. Do NOT be generous.
- Score 8-10 ONLY if this frame would genuinely make someone STOP scrolling. This is rare.
- Score 1-3 for: no face, no energy, no text, dark/blurry, static/boring frames
- Score 4-6 for: average frames with some elements but nothing compelling
- Score 7 ONLY if frame has face + energy + text all together
- Low audio + dead visual = score must be 1-2, no exceptions
- Be specific in scroll_risk_reason — name exactly what would make someone scroll

OTHER RULES:
- face_expression: be precise — is the person reacting, laughing, shocked, explaining?
- visual_energy: dead=nothing moving, low=minimal movement, medium=some action, high=fast cuts or strong reaction, explosive=extreme energy/jump cut/shocking moment
- scene_description: describe exactly what is happening — who, what, where

For EVERY frame return ONLY this JSON array (no extra text, no markdown, no preamble):
[
  {{
    "timestamp": 0.0,
    "has_face": true,
    "face_expression": "excited/happy/neutral/talking/surprised/shocked/laughing/none",
    "has_text_overlay": false,
    "text_content": "EXACT text visible in frame or empty string",
    "visual_energy": "dead/low/medium/high/explosive",
    "composition": "good/average/poor",
    "scroll_risk": "low/medium/high/critical",
    "scroll_risk_reason": "specific 1-sentence reason with exact visual detail",
    "best_thing": "the ONE most compelling thing about this frame",
    "brightness": "dark/normal/bright",
    "movement": "static/slow/fast",
    "scene_description": "1 sentence: who is doing what, what is visible",
    "frame_score": 5
  }}
]"""

    raw = call_nova(FRAME_PROMPT, images=batch_images)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return json.loads(raw)


def run_analysis(job_id: str, video_path: str, filename: str,user_id: str = "default_user"):
    try:
        jobs[job_id]["progress"] = 10

        result = subprocess.run(
            f"ffprobe -v error -select_streams v:0 "
            f"-show_entries stream=width,height,duration -of json '{video_path}'",
            shell=True, capture_output=True, text=True)
        probe = json.loads(result.stdout)['streams'][0]
        width, height = int(probe['width']), int(probe['height'])
        duration = float(probe.get('duration', 0))
        is_vertical = (height / width) >= 1.7

        jobs[job_id]["progress"] = 20

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
            energy_per_second.setdefault(sec, []).append(float(r))
        energy_timeline = {sec: round(float(np.mean(vals)), 4) for sec, vals in energy_per_second.items()}

        jobs[job_id]["progress"] = 35

        frames_folder = f"/tmp/{job_id}_frames"
        os.makedirs(frames_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        MAX_FRAMES = 30
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

        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
        all_images = [Image.open(f"{frames_folder}/{fname}") for fname in frame_files]

        audio_context = []
        for ts in frame_timestamps:
            sec = int(ts)
            energy = energy_timeline.get(sec, avg_energy)
            level = "high" if energy > avg_energy * 1.5 else "low" if energy < avg_energy * 0.5 else "medium"
            audio_context.append({"timestamp": ts, "audio_energy": level})

        BATCH_SIZE = 10
        batches = []
        for i in range(0, len(all_images), BATCH_SIZE):
            batches.append({
                "images": all_images[i:i+BATCH_SIZE],
                "timestamps": frame_timestamps[i:i+BATCH_SIZE],
                "audio": audio_context[i:i+BATCH_SIZE],
            })

        results = [None] * len(batches)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _analyze_frame_batch,
                    b["images"], b["timestamps"], b["audio"]
                ): i for i, b in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        all_frame_analyses = []
        for r in results:
            all_frame_analyses.extend(r)
        all_frame_analyses.sort(key=lambda x: x.get('timestamp', 0))

        jobs[job_id]["progress"] = 65

        total = len(all_frame_analyses)
        hook_frames = [f for f in all_frame_analyses if f.get('timestamp', 0) <= 3.0]
        hook_has_face = any(f.get('has_face') for f in hook_frames)
        hook_has_text = any(f.get('has_text_overlay') for f in all_frame_analyses)
        hook_energy = any(f.get('visual_energy') in ['high', 'explosive'] for f in hook_frames)
        hook_avg_score = float(np.mean([f.get('frame_score', 3) for f in hook_frames])) if hook_frames else 3.0
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

        MENTOR_PROMPT = f"""You are India's top viral content mentor — you've helped 10,000+ creators grow from 0 to millions of followers. You speak directly, kindly, and with specific advice.

You just finished a frame-by-frame analysis of a creator's reel. Here is the COMPLETE data:

FULL FRAME-BY-FRAME ANALYSIS:
{json.dumps(all_frame_analyses, indent=2)}

VIDEO DATA:
- Total duration: {duration:.1f} seconds
- Vertical 9:16 format: {is_vertical}
- Face visible in: {face_percentage:.0f}% of video
- Text/captions in: {text_percentage:.0f}% of frames
- Hook (first 3s) has face: {hook_has_face}
- Hook (first 3s) has energy: {hook_energy}
- Low audio energy zones: {low_energy_zones if low_energy_zones else "None"}
- Best frame: {best_frame.get('timestamp')}s (score: {best_frame.get('frame_score')}/10) — {best_frame.get('scene_description', '')}
- Worst frame: {worst_frame.get('timestamp')}s (score: {worst_frame.get('frame_score')}/10) — {worst_frame.get('scene_description', '')}
- Average frame quality: {avg_frame_score:.1f}/10

PROVEN PLATFORM BENCHMARKS (cite these in your advice):
- Meta Research: Face in first 3s = 35% higher retention
- HubSpot 300k video study: Face presence 60%+ = 38% more engagement
- Instagram Research: Captions = completion rate goes from 37% to 65%
- TikTok Creator Academy: 12-20 cuts/min = optimal pacing
- Instagram Research: Trending audio in first 3s = 41% better retention

WRITE A MENTOR REVIEW that:
1. Opens with ONE brutally honest sentence about this reel's biggest strength or problem — be specific, reference what you actually saw in the frames
2. Highlights the BEST moment with exact timestamp and exactly why it works visually
3. Calls out the MOST CRITICAL problem with exact timestamp and backs it with a platform stat
4. Gives exactly 3 numbered action items — each specific, actionable, backed by a stat, and directly tied to what you saw in the frames
5. Ends with one encouraging sentence about what this creator can achieve in India

Tone: Friendly but direct. Like a mentor who genuinely wants them to grow. Reference India. Keep it under 280 words.
Do NOT use bullet points — write in paragraphs.
Do NOT give generic advice — every sentence must reference something specific you saw in the actual frame data.

Then on new lines, output EXACTLY these scores (be strict and honest, not generous):
VIRAL_SCORE: [1-100]
HOOK_SCORE: [1-10]
RETENTION_SCORE: [1-10]
ENGAGEMENT_SCORE: [1-10]
PLATFORM_FIT_SCORE: [1-10]
CAPTION_SCORE: [1-10]
AUDIO_SCORE: [1-10]"""

        mentor_text = call_nova_text(MENTOR_PROMPT)
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

        INDIA_PROMPT = f"""You are a digital growth strategist specializing in the Indian creator economy.

A creator just made this reel:
- Viral score: {scores.get('VIRAL_SCORE')}/100
- Face presence: {face_percentage:.0f}%
- Has captions: {text_percentage:.0f}% of frames
- Duration: {duration:.1f}s
- Content detected from frames: {[f.get('text_content') for f in all_frame_analyses if f.get('text_content')]}
- Scene descriptions: {[f.get('scene_description') for f in all_frame_analyses if f.get('scene_description')][:5]}

Answer in exactly 3 short sentences:
1. Which specific Indian audience (state/region + age group + language) will connect with this content most, and why based on the content?
2. Which ONE Indian language should they dub this into FIRST for maximum reach, and exactly how many million people does that unlock?
3. Which platform (Instagram Reels / YouTube Shorts / Moj / Josh) should they post on first and why?

Be specific with real numbers. No fluff."""

        india_recommendation = call_nova_text(INDIA_PROMPT)
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

        # Save to DynamoDB
        save_analysis(user_id, job_id, jobs[job_id]["result"])

        try:
            os.remove(video_path)
            os.remove(audio_path)
            import shutil
            shutil.rmtree(frames_folder, ignore_errors=True)
        except:
            pass

    except Exception as e:
        jobs[job_id] = {"status": "error", "progress": 0, "message": str(e)}
