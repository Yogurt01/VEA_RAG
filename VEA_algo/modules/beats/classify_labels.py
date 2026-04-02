import csv
import json
import os
import re

# Set the paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'class_labels_indices.csv')
json_path = os.path.join(script_dir, 'config.json')

# Smart keywords for "vibe" classification
# Group 1: Music genres & styles
vibe_genres = [
    'music', 'song', 'jazz', 'pop', 'rock', 'metal', 'punk', 'grunge', 
    'reggae', 'country', 'swing', 'bluegrass', 'funk', 'folk', 'disco', 
    'opera', 'techno', 'dubstep', 'electronica', 'trance', 'salsa', 
    'flamenco', 'blues', 'afrobeat', 'gospel', 'ska', 'lullaby', 
    'a capella', 'ambient', 'rhythm and blues', 'soundtrack', 'theme',
    'jingle'
]

# Group 2: Emotions & Mood
vibe_moods = [
    'happy', 'funny', 'sad', 'tender', 'exciting', 'angry', 'scary', 'tense'
]

# Group 3: Atmosphere & Environmental acoustics
vibe_atmos = [
    'inside', 'outside', 'room', 'hall', 'urban', 'rural', 'natural',
    'space', 'silence', 'noise', 'cacophony', 'static', 'distortion',
    'reverberation', 'echo', 'background', 'field recording', 'atmosphere'
]

vibe_keywords = set(vibe_genres + vibe_moods + vibe_atmos)

# Create a regex to match any vibe keyword as whole words (case-insensitive)
vibe_pattern = re.compile(rf"\b({'|'.join(re.escape(k) for k in vibe_keywords)})\b", re.IGNORECASE)

config_data = {}

with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        index = row['index']
        name = row['display_name']
        
        # Check for vibe keywords
        if vibe_pattern.search(name):
            label_type = 'vibe'
        else:
            # Everything else (objects, humans, animals, actions) falls into tag
            label_type = 'tag'
            
        config_data[index] = {
            "label": name,
            "type": label_type
        }

# Save to config.json
with open(json_path, mode='w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=4, ensure_ascii=False)

print(f"Successfully processed {len(config_data)} labels.")
print(f"Output saved to {json_path}")
