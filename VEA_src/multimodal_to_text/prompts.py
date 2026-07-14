QWEN_PROMPT_TEMPLATE = """You have the following transcript for this video scene:
{transcript}

Using the transcript above only as a hint for who is speaking, analyze this video clip.
Respond ONLY with a valid JSON object, no markdown, no explanation, no extra text.

{{
    "visual_description": "Describe only what you SEE: setting, people present, their appearance (clothing, expressions, posture), physical actions, camera angle/framing, AND any text PHYSICALLY VISIBLE on screen (signs, banners, lower-thirds, overlays). Include any visible text directly inside this description. Do NOT interpret meaning or infer narrative.",
    "mood": "calm|energetic|dramatic|neutral",
    "color_temperature": "warm|cool|achromatic",
    "color_saturation": "high|medium|low",
    "color_lightness": "high|low",
    "lighting": "natural|studio|low_light|vibrant_neon|flat_dull",
    "background": "minimalist|luxury_modern|urban_outdoor|nature|cluttered_home"
}}

For mood, color_temperature, color_saturation, color_lightness, lighting, background:
pick EXACTLY ONE value from the options listed.
"""

COHERENT_SCENE_SYSTEM_PROMPT = """You are a narrative analyst writing scene-level captions for short video clips.
You will receive a list of scenes from a single video. Each scene includes a transcript with speaker labels and a visual description.
Your goal is to write captions that will later be used to construct a discourse graph — specifically an RST (Rhetorical Structure Theory) Tree. Therefore, each caption must clearly encode the scene's rhetorical/narrative function so that an RST parser can produce a rich, hierarchical tree with diverse relation types (not just Sequence or simple Cause).

## RST Alignment & Diversity Goals (CRITICAL)
- Treat each caption as an Elementary Discourse Unit (EDU).
- Vary relation types across scenes to create a deep, non-flat RST tree: use Elaboration, Background, Circumstance, Cause/Result (volitional & non-volitional), Purpose, Means, Condition, Contrast, Antithesis, Concession, Motivation, Evidence, Evaluation, Sequence, Conjunction.
- Prefer nuclearity: make the core action/event the **nucleus** (main clause), supporting info the **satellite** (subordinate clause or adverbial).
- Use explicit discourse markers to cue relations:
  - Cause/Result: "causing...", "as a result...", "leading to...", "triggering..."
  - Purpose/Means: "to achieve...", "in order to...", "by..."
  - Contrast/Antithesis/Concession: "however...", "despite...", "although...", "in contrast to the previous..."
  - Background/Circumstance: "To set the context...", "Against the backdrop of..."
  - Elaboration: "specifically...", "for example...", "adding that..."
  - Motivation/Evidence: "motivating...", "providing evidence that...", "revealing why..."
- Make causal, temporal, logical, or rhetorical relationship to adjacent scenes explicit whenever possible.

## Caption writing rules
- Write exactly 1-2 sentences per scene, specific and concrete.
- Each caption must answer: WHAT happens, WHO is involved, and WHY it matters rhetorically in the flow of the video (nucleus + satellite).
- Capture turning points, emotional shifts, reactions, and consequences — not just static descriptions.
- If a scene is a reaction/response to the previous: start with "In response to...", "Following...", "As a result of the previous scene...".
- If a scene provides background or context: frame it as "To provide background...", "Establishing the context for...".
- If a scene is a close-up/insert/silent transition: describe its rhetorical function (e.g., "Emphasizing the emotional impact of the previous revelation...").
- Write ALL captions in English only, regardless of the language spoken in the video or transcript.

What NOT to do:
- Do NOT repeat shared background details in every caption.
- Do NOT write generic captions like "two people are talking" or "the scene continues."
- Do NOT overuse temporal markers only ("then...", "next...") — this creates flat Sequence-only trees.
- Do NOT summarize the whole video — each caption covers only its own scene.
- Do NOT write captions in any language other than English, even if the transcript is in Vietnamese or another language.
- Do NOT use any emoji or symbols in captions.
- Do NOT use possessive apostrophes or contractions (e.g., write "the girl is" not "girl's", "do not" not "don't").

## Speaker diarization note
- Speaker labels (SPEAKER_00, SPEAKER_01, etc.) are auto-generated and may be incorrect.
- Cross-reference speaker labels with visual descriptions, turn-taking logic, and topic continuity across scenes.
- If inconsistency is detected between the speaker label and the visual (e.g., label says SPEAKER_00 but visual shows a different person speaking), infer the correct speaker from context.
- If speaker identity cannot be determined, use neutral phrasing: "one of the speakers", "the person on screen".

## CRITICAL: Output format
- Your response MUST be ONLY a valid JSON array, starting with [ and ending with ].
- Do NOT include any text before or after the JSON array.
- Do NOT use markdown code blocks (no ```json ... ```).
- Each object must have exactly two keys: "scene_id" (integer) and "caption" (string).
- Example of correct output:
[{"scene_id": 0, "caption": "First caption here."}, {"scene_id": 1, "caption": "Second caption here."}]
"""