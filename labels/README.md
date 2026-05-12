# Labels

This directory contains the label files used in our experiments on the [Toyota Woven Traffic Safety (WTS) dataset](https://woven-visionai.github.io/wts-dataset-homepage/).

## Files

| File         | Split | # Scenes |
|--------------|-------|----------|
| `train.json` | Train | 97       |
| `val.json`   | Val   | 48       |

These files correspond to the `train` and `val` splits of the WTS dataset.

## Format

Each file is a JSON object mapping a **scene ID** to its metadata:

```json
{
  "20230707_8_SN46_T1": {
    "camera": "Camera2_1",
    "class": 2
  },
  ...
}
```

### Fields

| Field    | Type   | Description |
|----------|--------|-------------|
| *key*    | string | **Scene ID** from the WTS dataset (e.g. `20230707_8_SN46_T1`). |
| `camera` | string | The overhead camera used in our experiments. Each WTS scene has multiple overhead cameras; we manually selected the one with the best viewing angle for each scene. |
| `class`  | int    | Class label for the scene: `1` = **near-miss**, `2` = **collision**. |

## Note on `normal_trimmed` videos

The WTS dataset contains a `normal_trimmed/` subfolder under both the `train` and `val` splits. Videos in this folder represent **normal** (non-event) driving and are assigned class label `0` by default. Because each `normal_trimmed` scene has only a single overhead camera, no `camera` field is needed, and these scenes are **not included** in `train.json` / `val.json`.

To assemble a complete label mapping in your code, treat all videos under `normal_trimmed/` as `class: 0` and use the provided JSON files for the remaining scenes.

## Class summary

| Class | Meaning   | Source                       |
|-------|-----------|------------------------------|
| 0     | Normal    | `normal_trimmed/` (implicit) |
| 1     | Near-miss | `train.json` / `val.json`    |
| 2     | Collision | `train.json` / `val.json`    |
