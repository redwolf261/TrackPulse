"""
Step 4: error taxonomy - primary cause assigned per false-WET error from direct visual inspection
(done during ground-truth labeling pass above). One primary cause each.
"""
import json

OUT_DIR = 'c:/Users/Rivan/Projects/AI_Grand_Prix/experiments/exp00_rscd_baseline'

# Cause assigned from visual inspection notes taken while labeling each image (see conversation).
# Category vocabulary per brief: DARK_ASPHALT, REFLECTION, SHADOW, SPRAY, KERB, GRASS, ADVERTISING,
# CAMERA_ANGLE, MOTION_BLUR, LENS_WATER, MIXED_SURFACE, DRY_RACING_LINE, STANDING_WATER, UNKNOWN
causes = {
    'contrast_01.jpg': 'DARK_ASPHALT',   # small dry asphalt strip, dark tarmac tone
    'contrast_02.jpg': 'DARK_ASPHALT',   # top-down pit lane, dark asphalt with shadow contrast
    'contrast_03.jpg': 'DARK_ASPHALT',   # Monza pit lane, dark asphalt, blue pit-lane paint marking nearby
    'contrast_04.jpg': 'SHADOW',         # strong hard shadows across pit box concrete
    'contrast_05.jpg': 'SHADOW',         # strong hard shadows, dark tire rubber marks
    'contrast_06.jpg': 'DARK_ASPHALT',   # dark worn pit lane asphalt
    'contrast_10.jpg': 'DARK_ASPHALT',   # dark grey pit floor, high contrast
    'contrast_12.jpg': 'DARK_ASPHALT',   # wide dry pit lane, dark tarmac
    'contrast_13.jpg': 'DARK_ASPHALT',   # overcast flat light, dark matte tarmac
    'contrast_14.jpg': 'DARK_ASPHALT',   # dark pit lane asphalt, overcast
    'contrast_15.jpg': 'DARK_ASPHALT',   # dark racing track tarmac, bright sun creating high contrast
    'contrast_16.jpg': 'DARK_ASPHALT',   # dark wet-look tarmac (actually dry) under overcast light
    'contrast_17.jpg': 'DARK_ASPHALT',   # dark grey track through fence, overcast
    'contrast_18.jpg': 'DARK_ASPHALT',   # dark asphalt strip visible
    'contrast_20.jpg': 'DARK_ASPHALT',   # dark tarmac, sun creating sheen-like highlight on track
    'contrast_21.jpg': 'DARK_ASPHALT',   # dark tarmac, sharp sun highlight band across track
    'contrast_22.jpg': 'REFLECTION',     # floodlit night race - artificial light creates specular highlights on track
    'contrast_23.jpg': 'DARK_ASPHALT',   # dark grey asphalt track, sunny
    'contrast_24.jpg': 'REFLECTION',     # floodlit night race - visible highlight streaks on track surface
    'contrast_25.jpg': 'DARK_ASPHALT',   # dark sandy-grey desert asphalt
    'contrast_26.jpg': 'SPRAY',          # white tire-lockup smoke cloud visually resembles water spray
    'contrast_27.jpg': 'SPRAY',          # white tire-lockup smoke cloud visually resembles water spray
    'contrast_28.jpg': 'MOTION_BLUR',    # heavy motion blur + night floodlight glare on dark track
    'contrast_29.jpg': 'REFLECTION',     # floodlit night race, lower-confidence call (0.537) but still wrong; track sheen from floodlights
    'contrast_30.jpg': 'REFLECTION',     # night race, sparks + floodlight reflections on track
    'contrast_31.jpg': 'DARK_ASPHALT',   # dark grey desert asphalt, sunny
    'contrast_32.jpg': 'DARK_ASPHALT',   # dark grey desert asphalt, sunny
    'contrast_33.jpg': 'REFLECTION',     # floodlit night race, visible sheen on track
    'contrast_35.jpg': 'DARK_ASPHALT',   # dark wide track surface with visible tire rubber marks (also could read as texture/staining)
}

with open(f'{OUT_DIR}/contrast_set_inference_results.json', encoding='utf-8') as f:
    results = json.load(f)

errors = [r for r in results if r['ground_truth'] not in ('AMBIGUOUS',) and r['predicted_class'] != r['ground_truth']]
print(f"total errors (excluding ambiguous): {len(errors)}")

missing = [r['image_id'] for r in errors if r['image_id'] not in causes]
print("errors missing a cause assignment:", missing)

import collections
cause_counts = collections.Counter()
tagged = []
for r in errors:
    cause = causes.get(r['image_id'], 'UNKNOWN')
    cause_counts[cause] += 1
    tagged.append({
        'image_id': r['image_id'],
        'ground_truth': r['ground_truth'],
        'predicted_class': r['predicted_class'],
        'confidence': r['confidence'],
        'primary_cause': cause,
    })

print("\ncause distribution across", len(errors), "errors:")
for c, n in cause_counts.most_common():
    print(f"  {c}: {n} ({100*n/len(errors):.1f}%)")

with open(f'{OUT_DIR}/error_taxonomy.json', 'w', encoding='utf-8') as f:
    json.dump({'n_errors': len(errors), 'cause_distribution': dict(cause_counts), 'tagged_errors': tagged}, f, indent=2)
print(f"\nsaved {OUT_DIR}/error_taxonomy.json")
