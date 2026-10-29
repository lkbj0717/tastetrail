TasteTrail v13

Start with work/culture_v13/RUN_GUIDE.txt.
Keep culture_v11 and culture_v13 adjacent under work/.
The public source excludes all secrets, raw MovieLens ratings, Qloo payloads, private model weights and user-level evaluation records.
Use the included bootstrap scripts and your own authorized keys to rebuild runtime assets.
New live API responses or rebuilt assets do not automatically reproduce the recorded confirmation results.
Final evaluation provenance records the original experiment hashes. docs/RELEASE.json relates the curated public configuration to that snapshot; selected numerical parameters and confirmation results are unchanged.
