# yolo_processed

Provenance: **SYNTHETIC**
Active classes: `0 cut`, `1 bruise`, `2 wound`.
abrasion, laceration, and swelling were dropped. They were not remapped.
Raw source datasets were left unchanged.

- Unique images: 124
- Splits: {'test': 18, 'train': 87, 'val': 19}
- Boxes per class: {'bruise': 63, 'cut': 61}
- Images per class: {'bruise': 63, 'cut': 61}
- Zero-box classes: ['wound']

See `manifest.csv` and `processing_summary.json`.
Do not train until remaining imbalance / missing wound is accepted or new honest labels exist.
