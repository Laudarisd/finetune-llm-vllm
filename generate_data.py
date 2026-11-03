import json
import re
import hashlib
import random
import sys
from pathlib import Path

# The generation helpers live under train_data/src; add that package root so
# this entry point works when started from any current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "train_data"))

from src.core.class_config import (
    ALL_CLASSES,
    CLASS_GROUPS,
    CLASS_NAME_MAP,
    FIXTURE_CLASSES,
    ROOM_CLASSES,
    STRUCTURAL_CLASSES,
)
from src.core.path_utils import build_image_metadata, resolve_image_path_from_json
from src.core.queries_answer import (
    COMPARISON_TEMPLATES,
    FOLLOWUP_COMPARE_PROMPTS,
    FOLLOWUP_COUNT_PROMPTS,
    FOLLOWUP_DETAIL_PROMPTS,
    FOLLOWUP_POLYGON_PROMPTS,
    GROUP_PRESENCE_MULTI_TEMPLATES,
    GROUP_QUERY_TEMPLATES,
    IMAGE_LEVEL_QUERIES,
    IMAGE_TYPE_QUERIES,
    OBJECT_CONCISE_QUERY_TEMPLATES,
    OBJECT_COUNT_MULTI_TEMPLATES,
    OBJECT_PRESENCE_MULTI_TEMPLATES,
    OBJECT_QUERY_TEMPLATES,
    RELATION_GRAPH_QUERIES,
    SUMMARY_FOLLOWUP_OBJECT_PROMPTS,
    YES_NO_VARIANTS,
)
from src.relation_logic.geometric_relation import build_relation_graph, polygon_area


# ============================================================
# Dataset paths
# ============================================================

DATASET_ROOT = PROJECT_ROOT / "raw_data"                    # Root folder for raw dataset
JSON_DIR = DATASET_ROOT / "json"                           # Folder with annotation JSON files
IMAGE_DIR = DATASET_ROOT / "images"                        # Folder with floor-plan images
OUTPUT_DIR = PROJECT_ROOT / "train_data"                    # Output directory for generated dataset
TRAIN_OUTPUT_JSONL = OUTPUT_DIR / "vlm_train.jsonl"        # Train split output file
VAL_OUTPUT_JSONL = OUTPUT_DIR / "vlm_val.jsonl"            # Validation split output file


# ============================================================
# Split and generation controls
# ============================================================

VAL_SPLIT_PERCENT = 20                                     # Validation split percentage by image stem hash
MAX_RELATION_OBJECTS_PER_TYPE = 8                          # Max linked objects per category in relation graph
INCLUDE_RELATION_GRAPH = True                              # Whether to generate relation graph samples
INCLUDE_IMAGE_METADATA_IN_ANSWER = False                   # Keep assistant JSON clean for fine-tuning
SHUFFLE_SEED = 42                                          # Deterministic shuffle so saved JSONL order is mixed.


# Relation-graph distance thresholds in image pixels.
ROOM_ADJACENCY_EPSILON = 20.0
DOOR_ROOM_LINK_EPSILON = 18.0
FIXTURE_ROOM_LINK_EPSILON = 30.0
WALL_BOUNDARY_EPSILON = 14.0


# ============================================================
# Basic helpers
# ============================================================

def normalize_name(label: str) -> str:
    """Convert CamelCase label into readable class name."""
    return CLASS_NAME_MAP.get(label, re.sub(r"(?<!^)([A-Z])", r" \1", label).lower())


def pluralize_name(name: str, count: int) -> str:
    """Return simple pluralized class name."""
    if count == 1:
        return name
    if name.endswith("s"):
        return name
    return f"{name}s"


def polygon_to_list(points):
    """Round polygon points into simple xy list."""
    cleaned = []
    for point in points:
        if len(point) < 2:
            continue
        cleaned.append([round(float(point[0]), 2), round(float(point[1]), 2)])
    return cleaned


def collect_group_annotations(group_labels, by_class):
    """Collect grouped annotations and counts."""
    annotations = []
    counts = {}
    for label in group_labels:
        polygons = by_class.get(label, [])
        if polygons:
            counts[label] = len(polygons)
            for polygon in polygons:
                annotations.append({"class": label, "polygon": polygon})
    return annotations, counts


def safe_json_dumps(value) -> str:
    """Serialize value to JSON or string."""
    if isinstance(value, str):
        value = {"answer": value}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ============================================================
# Layout summaries
# ============================================================

def summarize_layout(shapes, counts, classes_present):
    """Build global structured layout summary."""
    room_types = [label for label in classes_present if label in ROOM_CLASSES]
    structural_elements = [label for label in classes_present if label in STRUCTURAL_CLASSES]
    fixtures = [label for label in classes_present if label in FIXTURE_CLASSES]
    other_classes = [
        label for label in classes_present
        if label not in ROOM_CLASSES and label not in STRUCTURAL_CLASSES and label not in FIXTURE_CLASSES
    ]
    absent_classes = [label for label in ALL_CLASSES if label not in classes_present]

    return {
        "classes_present": classes_present,
        "classes_absent": absent_classes,
        "counts": counts,
        "room_types": room_types,
        "structural_elements": structural_elements,
        "fixtures": fixtures,
        "other_classes": other_classes,
        "total_objects": len(shapes),
    }


def summarize_layout_concise(shapes, counts, classes_present):
    """Build short natural summary."""
    room_names = [normalize_name(label) for label in classes_present if label in ROOM_CLASSES]
    structural_names = [normalize_name(label) for label in classes_present if label in STRUCTURAL_CLASSES]
    fixture_names = [normalize_name(label) for label in classes_present if label in FIXTURE_CLASSES]

    parts = []
    if room_names:
        parts.append(f"rooms: {', '.join(room_names[:6])}")
    if structural_names:
        parts.append(f"structural elements: {', '.join(structural_names[:6])}")
    if fixture_names:
        parts.append(f"fixtures: {', '.join(fixture_names[:6])}")

    return {
        "answer": "; ".join(parts) if parts else "No annotated classes found.",
        "total_objects": len(shapes),
        "counts": counts,
    }


def build_floorplan_type_answer():
    """Return concise answer for image-type question."""
    return {"answer": "This is a floor plan."}


def with_image_metadata(answer, image_metadata):
    """Attach image metadata to dictionary answers."""
    if not INCLUDE_IMAGE_METADATA_IN_ANSWER:
        return answer
    if not isinstance(answer, dict):
        return answer
    merged = dict(answer)
    for key, value in image_metadata.items():
        if key not in merged:
            merged[key] = value
    return merged


# ============================================================
# Split helper
# ============================================================

def split_bucket_for_json(json_path: Path) -> str:
    """Deterministic image-level train/val split."""
    stem = json_path.stem.lower().encode("utf-8")
    digest = hashlib.md5(stem).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < VAL_SPLIT_PERCENT else "train"


# ============================================================
# Ranking helpers
# ============================================================

def total_polygon_area(polygons):
    """Sum polygon areas."""
    return sum(polygon_area(polygon) for polygon in polygons)


def rank_present_labels(by_class):
    """Rank present labels by count and area."""
    present_labels = [label for label in ALL_CLASSES if by_class.get(label)]
    return sorted(
        present_labels,
        key=lambda label: (-len(by_class[label]), -total_polygon_area(by_class[label]), normalize_name(label)),
    )


def rank_absent_labels(by_class):
    """Rank absent labels alphabetically."""
    absent_labels = [label for label in ALL_CLASSES if not by_class.get(label)]
    return sorted(absent_labels, key=normalize_name)


def rank_groups(by_class):
    """Rank non-empty groups by total annotations."""
    ranked = []
    for group_name, group_labels in CLASS_GROUPS.items():
        annotations, counts = collect_group_annotations(group_labels, by_class)
        total_count = len(annotations)
        if total_count == 0:
            continue
        ranked.append((group_name, group_labels, annotations, counts, total_count))
    ranked.sort(key=lambda item: (-item[4], item[0]))
    return ranked


def build_comparison_pairs(ranked_labels):
    """Build deterministic comparison pairs without random sampling."""
    pairs = []
    for index in range(len(ranked_labels) - 1):
        pairs.append((ranked_labels[index], ranked_labels[index + 1]))
    return pairs


# ============================================================
# Answer builders
# ============================================================

def build_image_level_answer(question, classes_present, counts, layout_summary, mode="structured"):
    """Build image-level answer in concise or structured form."""
    lowered = question.lower()
    absent_classes = [label for label in ALL_CLASSES if label not in classes_present]

    if mode == "concise":
        if "present in this floor plan and which classes are absent" in lowered:
            return {
                "answer": (
                    f"Present classes: {', '.join(normalize_name(c) for c in classes_present)}. "
                    f"Absent classes: {', '.join(normalize_name(c) for c in absent_classes[:10])}."
                )
            }

        if "total annotated polygons" in lowered:
            return {
                "answer": (
                    f"There are {layout_summary['total_objects']} annotated polygons across "
                    f"{len(classes_present)} present classes."
                )
            }

        if "counts" in lowered or "list all classes" in lowered:
            text = ", ".join(f"{normalize_name(label)}: {count}" for label, count in counts.items())
            return {"answer": text if text else "No classes are present."}

        return summarize_layout_concise(
            shapes=[None] * layout_summary["total_objects"],
            counts=counts,
            classes_present=classes_present,
        )

    if "present in this floor plan and which classes are absent" in lowered:
        return {"classes_present": classes_present, "classes_absent": absent_classes, "counts": counts}

    if "total annotated polygons" in lowered:
        return {
            "total_objects": layout_summary["total_objects"],
            "classes_present": classes_present,
            "counts": counts,
        }

    if "counts" in lowered or "list all classes" in lowered:
        return {"classes_present": classes_present, "counts": counts}

    return layout_summary


def build_object_answer(label, polygons, mode="structured"):
    """Build object answer."""
    present = len(polygons) > 0
    readable_name = normalize_name(label)

    if mode == "concise":
        if present:
            noun = pluralize_name(readable_name, len(polygons))
            return {"answer": f"There are {len(polygons)} {noun}."}
        return {"answer": f"No {pluralize_name(readable_name, 2)} are present."}

    return {"class": label, "present": present, "count": len(polygons), "polygons": polygons}


def build_presence_reply(label, count):
    """Build short yes/no conversational reply."""
    readable_name = normalize_name(label)
    if count > 0:
        return {"answer": f"Yes, there are {count} {pluralize_name(readable_name, count)}."}
    return {"answer": f"No, there are no {pluralize_name(readable_name, 2)}."}


def build_presence_short_reply(label, count):
    """Build extra short yes/no reply."""
    if count > 0:
        return {"answer": YES_NO_VARIANTS[True][count % len(YES_NO_VARIANTS[True])]}
    return {"answer": YES_NO_VARIANTS[False][0]}


def build_followup_count_reply(label, count):
    """Build follow-up count reply."""
    readable_name = normalize_name(label)
    return {"answer": f"There are {count} {pluralize_name(readable_name, count)} in total."}


def build_group_answer(group_name, group_labels, annotations, group_counts, total_count, mode="structured"):
    """Build grouped answer."""
    if mode == "concise":
        if total_count == 0:
            return {"answer": f"No {group_name} are present."}
        count_text = ", ".join(f"{normalize_name(label)}: {count}" for label, count in group_counts.items())
        return {"answer": f"{group_name} are present. Counts: {count_text}."}

    return {
        "group": group_name,
        "classes": group_labels,
        "present": total_count > 0,
        "count": total_count,
        "counts": group_counts,
        "annotations": annotations,
    }


def build_comparison_answer(label_a, label_b, count_a, count_b, mode="structured"):
    """Build comparison answer."""
    if count_a == count_b:
        comparison = "equal"
        more_common = None
        less_common = None
    elif count_a > count_b:
        comparison = label_a
        more_common = label_a
        less_common = label_b
    else:
        comparison = label_b
        more_common = label_b
        less_common = label_a

    if mode == "concise":
        if comparison == "equal":
            return {
                "answer": (
                    f"{normalize_name(label_a)} and {normalize_name(label_b)} appear equally often "
                    f"({count_a} each)."
                )
            }
        return {
            "answer": (
                f"{normalize_name(more_common)} appears more often than "
                f"{normalize_name(less_common)} ({max(count_a, count_b)} vs {min(count_a, count_b)})."
            )
        }

    return {
        "object_a": label_a,
        "object_b": label_b,
        "count_a": count_a,
        "count_b": count_b,
        "comparison": comparison,
        "more_common": more_common,
        "less_common": less_common,
    }


def build_summary_followup_answer(label, count):
    """Build reply for summary -> object follow-up."""
    readable_name = normalize_name(label)
    if count > 0:
        return {"answer": f"Yes, {pluralize_name(readable_name, count)} are present."}
    return {"answer": f"No, no {pluralize_name(readable_name, 2)} are present."}


# ============================================================
# Sample builders
# ============================================================

def make_single_turn_sample(image_path: Path, question: str, answer) -> dict:
    """Create one single-turn multimodal sample."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": safe_json_dumps(answer)}],
            },
        ]
    }


def make_multi_turn_sample(image_path: Path, turns) -> dict:
    """Create one multi-turn multimodal sample."""
    messages = []
    first_user = True

    for role, text in turns:
        if role == "user":
            if first_user:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": text},
                        ],
                    }
                )
                first_user = False
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    }
                )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": safe_json_dumps(text)}],
                }
            )

    return {"messages": messages}


def add_unique_single_turn(samples, seen_keys, image_path: Path, question: str, answer):
    """Add unique single-turn sample."""
    key = ("single", str(image_path), question)
    if key in seen_keys:
        return False
    seen_keys.add(key)
    samples.append(make_single_turn_sample(image_path, question, answer))
    return True


def conversation_signature(turns):
    """Create dedup key for multi-turn conversation."""
    return tuple((role, safe_json_dumps(text) if role == "assistant" else text) for role, text in turns)


def add_unique_multi_turn(samples, seen_keys, image_path: Path, turns):
    """Add unique multi-turn sample."""
    key = ("multi", str(image_path), conversation_signature(turns))
    if key in seen_keys:
        return False
    seen_keys.add(key)
    samples.append(make_multi_turn_sample(image_path, turns))
    return True


# ============================================================
# Multi-turn conversation builders
# ============================================================

def build_multi_turn_count_dialogues(
    image_path,
    label,
    count,
    samples,
    seen_keys,
    image_metadata,
    max_presence_prompts=2,
    include_detail=True,
):
    """Add object presence -> follow-up count dialogues."""
    readable_name = normalize_name(label)

    for first_question in OBJECT_PRESENCE_MULTI_TEMPLATES[:max_presence_prompts]:
        turns = [
            ("user", first_question.format(object=readable_name)),
            ("assistant", with_image_metadata(build_presence_short_reply(label, count), image_metadata)),
            ("user", FOLLOWUP_COUNT_PROMPTS[count % len(FOLLOWUP_COUNT_PROMPTS)]),
            ("assistant", with_image_metadata(build_followup_count_reply(label, count), image_metadata)),
        ]
        add_unique_multi_turn(samples, seen_keys, image_path, turns)

    if include_detail:
        turns = [
            ("user", f"is there any {readable_name} in this floor plan"),
            ("assistant", with_image_metadata(build_presence_reply(label, count), image_metadata)),
            ("user", FOLLOWUP_DETAIL_PROMPTS[count % len(FOLLOWUP_DETAIL_PROMPTS)]),
            ("assistant", with_image_metadata(build_followup_count_reply(label, count), image_metadata)),
        ]
        add_unique_multi_turn(samples, seen_keys, image_path, turns)


def build_multi_turn_object_count_to_detail_dialogues(
    image_path,
    label,
    count,
    polygons,
    samples,
    seen_keys,
    image_metadata,
    max_count_prompts=2,
):
    """Add count-first object dialogues with polygon-detail follow-up."""
    readable_name = normalize_name(label)
    plural_name = pluralize_name(readable_name, 2)

    for index, first_question in enumerate(OBJECT_COUNT_MULTI_TEMPLATES[:max_count_prompts]):
        turns = [
            ("user", first_question.format(object=plural_name)),
            ("assistant", with_image_metadata(build_followup_count_reply(label, count), image_metadata)),
            ("user", FOLLOWUP_POLYGON_PROMPTS[index % len(FOLLOWUP_POLYGON_PROMPTS)]),
            ("assistant", with_image_metadata(build_object_answer(label, polygons, mode="structured"), image_metadata)),
        ]
        add_unique_multi_turn(samples, seen_keys, image_path, turns)


def build_multi_turn_image_to_object_dialogues(image_path, label, count, samples, seen_keys, image_metadata):
    """Add image type -> object -> follow-up count dialogues."""
    readable_name = normalize_name(label)
    summary_template = SUMMARY_FOLLOWUP_OBJECT_PROMPTS[count % len(SUMMARY_FOLLOWUP_OBJECT_PROMPTS)]
    plural_name = pluralize_name(readable_name, 2)
    if "any {object}" in summary_template or "are there" in summary_template:
        summary_object = plural_name
    else:
        summary_object = readable_name

    turns = [
        ("user", IMAGE_TYPE_QUERIES[count % len(IMAGE_TYPE_QUERIES)]),
        ("assistant", with_image_metadata(build_floorplan_type_answer(), image_metadata)),
        ("user", summary_template.format(object=summary_object)),
        ("assistant", with_image_metadata(build_summary_followup_answer(label, count), image_metadata)),
        ("user", f"how many {plural_name} are there in total"),
        ("assistant", with_image_metadata(build_followup_count_reply(label, count), image_metadata)),
    ]
    add_unique_multi_turn(samples, seen_keys, image_path, turns)


def build_multi_turn_comparison_dialogues(image_path, label_a, label_b, count_a, count_b, samples, seen_keys, image_metadata):
    """Add multi-turn comparison dialogues."""
    turns = [
        ("user", f"are there any {pluralize_name(normalize_name(label_a), 2)}"),
        ("assistant", with_image_metadata(build_summary_followup_answer(label_a, count_a), image_metadata)),
        ("user", f"and {pluralize_name(normalize_name(label_b), 2)}"),
        ("assistant", with_image_metadata(build_summary_followup_answer(label_b, count_b), image_metadata)),
        ("user", FOLLOWUP_COMPARE_PROMPTS[(count_a + count_b) % len(FOLLOWUP_COMPARE_PROMPTS)]),
        ("assistant", with_image_metadata(
            build_comparison_answer(label_a, label_b, count_a, count_b, mode="concise"),
            image_metadata,
        )),
    ]
    add_unique_multi_turn(samples, seen_keys, image_path, turns)


def build_multi_turn_group_dialogues(image_path, group_name, group_counts, total_count, samples, seen_keys, image_metadata):
    """Add group presence -> follow-up total dialogues."""
    turns = [
        ("user", GROUP_PRESENCE_MULTI_TEMPLATES[total_count % len(GROUP_PRESENCE_MULTI_TEMPLATES)].format(group=group_name)),
        ("assistant", with_image_metadata(
            {"answer": f"Yes, {group_name} are present."} if total_count > 0 else {"answer": f"No, {group_name} are not present."},
            image_metadata,
        )),
        ("user", FOLLOWUP_COUNT_PROMPTS[total_count % len(FOLLOWUP_COUNT_PROMPTS)]),
        ("assistant", with_image_metadata(
            {"answer": f"There are {total_count} total annotations in {group_name}."},
            image_metadata,
        )),
    ]
    add_unique_multi_turn(samples, seen_keys, image_path, turns)

    if group_counts:
        detail_text = ", ".join(f"{normalize_name(label)}: {count}" for label, count in group_counts.items())
        turns = [
            ("user", f"are there any {group_name} in this floor plan"),
            ("assistant", with_image_metadata({"answer": f"Yes, {group_name} are present."}, image_metadata)),
            ("user", "break the total down by class"),
            ("assistant", with_image_metadata({"answer": detail_text}, image_metadata)),
        ]
        add_unique_multi_turn(samples, seen_keys, image_path, turns)


def build_multi_turn_total_annotation_dialogue(image_path, layout_summary, samples, seen_keys, image_metadata):
    """Add overall total annotation dialogue."""
    total_objects = layout_summary["total_objects"]

    turns = [
        ("user", "how many total annotated objects are in this image"),
        ("assistant", with_image_metadata({"answer": f"There are {total_objects} annotated objects in total."}, image_metadata)),
        ("user", "which classes do they belong to"),
        ("assistant", with_image_metadata({"counts": layout_summary["counts"]}, image_metadata)),
    ]
    add_unique_multi_turn(samples, seen_keys, image_path, turns)


# ============================================================
# Core generation
# ============================================================

def build_single_turn_samples_for_file(
    image_path,
    image_metadata,
    shapes,
    by_class,
    classes_present,
    counts,
    layout_summary,
    relation_graph,
):
    """Build all single-turn samples."""
    samples = []
    seen_keys = set()

    # Image-level samples
    for question in IMAGE_LEVEL_QUERIES:
        answer = build_image_level_answer(
            question=question,
            classes_present=classes_present,
            counts=counts,
            layout_summary=layout_summary,
            mode="structured",
        )
        add_unique_single_turn(samples, seen_keys, image_path, question, with_image_metadata(answer, image_metadata))

    # Image type sample
    add_unique_single_turn(
        samples,
        seen_keys,
        image_path,
        "what kind of image is this",
        with_image_metadata(build_floorplan_type_answer(), image_metadata),
    )

    # Present object samples
    ranked_present_labels = rank_present_labels(by_class)
    for label in ranked_present_labels:
        polygons = by_class[label]
        for template in OBJECT_QUERY_TEMPLATES:
            answer = build_object_answer(label, polygons, mode="structured")
            add_unique_single_turn(
                samples,
                seen_keys,
                image_path,
                template.format(object=normalize_name(label)),
                with_image_metadata(answer, image_metadata),
            )

        for template in OBJECT_CONCISE_QUERY_TEMPLATES:
            answer = build_object_answer(label, polygons, mode="concise")
            add_unique_single_turn(
                samples,
                seen_keys,
                image_path,
                template.format(object=normalize_name(label)),
                with_image_metadata(answer, image_metadata),
            )

    # Absent object samples
    ranked_absent_labels = rank_absent_labels(by_class)
    for label in ranked_absent_labels:
        for template in OBJECT_CONCISE_QUERY_TEMPLATES:
            answer = build_object_answer(label, [], mode="concise")
            add_unique_single_turn(
                samples,
                seen_keys,
                image_path,
                template.format(object=normalize_name(label)),
                with_image_metadata(answer, image_metadata),
            )

    # Group samples
    ranked_groups = rank_groups(by_class)
    for group_name, group_labels, annotations, group_counts, total_count in ranked_groups:
        for template in GROUP_QUERY_TEMPLATES:
            answer = build_group_answer(
                group_name=group_name,
                group_labels=group_labels,
                annotations=annotations,
                group_counts=group_counts,
                total_count=total_count,
                mode="structured",
            )
            add_unique_single_turn(
                samples,
                seen_keys,
                image_path,
                template.format(group=group_name),
                with_image_metadata(answer, image_metadata),
            )

    # Comparison samples
    comparison_pairs = build_comparison_pairs(ranked_present_labels)
    for label_a, label_b in comparison_pairs:
        count_a = len(by_class[label_a])
        count_b = len(by_class[label_b])
        for template in COMPARISON_TEMPLATES:
            answer = build_comparison_answer(
                label_a=label_a,
                label_b=label_b,
                count_a=count_a,
                count_b=count_b,
                mode="structured",
            )
            add_unique_single_turn(
                samples,
                seen_keys,
                image_path,
                template.format(
                    object_a=normalize_name(label_a),
                    object_b=normalize_name(label_b),
                ),
                with_image_metadata(answer, image_metadata),
            )

    # Relation graph sample
    if INCLUDE_RELATION_GRAPH:
        for question in RELATION_GRAPH_QUERIES:
            add_unique_single_turn(
                samples,
                seen_keys,
                image_path,
                question,
                with_image_metadata(relation_graph, image_metadata),
            )

    return samples


def build_multi_turn_samples_for_file(
    image_path,
    image_metadata,
    by_class,
    layout_summary,
):
    """Build multi-turn conversational samples."""
    samples = []
    seen_keys = set()

    ranked_present_labels = rank_present_labels(by_class)
    ranked_groups = rank_groups(by_class)
    comparison_pairs = build_comparison_pairs(ranked_present_labels)

    # Build follow-up dialogues for every present object class.
    for label in ranked_present_labels:
        count = len(by_class[label])
        build_multi_turn_count_dialogues(
            image_path,
            label,
            count,
            samples,
            seen_keys,
            image_metadata,
            max_presence_prompts=len(OBJECT_PRESENCE_MULTI_TEMPLATES),
            include_detail=True,
        )

        build_multi_turn_object_count_to_detail_dialogues(
            image_path=image_path,
            label=label,
            count=count,
            polygons=by_class[label],
            samples=samples,
            seen_keys=seen_keys,
            image_metadata=image_metadata,
            max_count_prompts=len(OBJECT_COUNT_MULTI_TEMPLATES),
        )
        build_multi_turn_image_to_object_dialogues(image_path, label, count, samples, seen_keys, image_metadata)

    # Absent label follow-up dialogue
    absent_labels = rank_absent_labels(by_class)
    for label in absent_labels:
        build_multi_turn_count_dialogues(
            image_path,
            label,
            0,
            samples,
            seen_keys,
            image_metadata,
            max_presence_prompts=len(OBJECT_PRESENCE_MULTI_TEMPLATES),
            include_detail=False,
        )
        build_multi_turn_object_count_to_detail_dialogues(
            image_path=image_path,
            label=label,
            count=0,
            polygons=[],
            samples=samples,
            seen_keys=seen_keys,
            image_metadata=image_metadata,
            max_count_prompts=len(OBJECT_COUNT_MULTI_TEMPLATES),
        )

    # Comparison dialogues
    for label_a, label_b in comparison_pairs:
        count_a = len(by_class[label_a])
        count_b = len(by_class[label_b])
        build_multi_turn_comparison_dialogues(
            image_path, label_a, label_b, count_a, count_b, samples, seen_keys, image_metadata
        )

    # Group dialogues
    for group_name, _group_labels, _annotations, group_counts, total_count in ranked_groups:
        build_multi_turn_group_dialogues(
            image_path, group_name, group_counts, total_count, samples, seen_keys, image_metadata
        )

    # Total annotation dialogue
    build_multi_turn_total_annotation_dialogue(
        image_path, layout_summary, samples, seen_keys, image_metadata
    )

    return samples


def shuffle_samples_for_image(samples, image_path: Path):
    """Shuffle one image's samples so prompts are not saved in repeated blocks."""
    seed_text = f"{SHUFFLE_SEED}:{image_path}"
    rng = random.Random(seed_text)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled


def build_samples_for_file(json_path: Path):
    """Build complete dataset samples for one annotation file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = resolve_image_path_from_json(data, json_path, IMAGE_DIR)

    if image_path is None:
        print(f"Skipping {json_path.name}: image not found.")
        return []

    # Parse shapes
    shapes = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = polygon_to_list(shape.get("points", []))
        if label and points:
            shapes.append({"label": label, "polygon": points})

    # Group by class
    by_class = {label: [] for label in ALL_CLASSES}
    for shape in shapes:
        by_class.setdefault(shape["label"], []).append(shape["polygon"])

    # Shared summaries
    image_metadata = build_image_metadata(data, image_path)
    classes_present = sorted({shape["label"] for shape in shapes})
    counts = {label: len(by_class.get(label, [])) for label in ALL_CLASSES if by_class.get(label)}
    layout_summary = summarize_layout(shapes, counts, classes_present)
    relation_graph = build_relation_graph(
        shapes=shapes,
        room_labels=ROOM_CLASSES,
        fixture_labels=FIXTURE_CLASSES,
        room_adjacency_gap=ROOM_ADJACENCY_EPSILON,
        door_room_gap=DOOR_ROOM_LINK_EPSILON,
        fixture_room_gap=FIXTURE_ROOM_LINK_EPSILON,
        window_wall_gap=WALL_BOUNDARY_EPSILON,
        max_objects_per_type=MAX_RELATION_OBJECTS_PER_TYPE,
    )

    # Build sample sets
    single_turn_samples = build_single_turn_samples_for_file(
        image_path=image_path,
        image_metadata=image_metadata,
        shapes=shapes,
        by_class=by_class,
        classes_present=classes_present,
        counts=counts,
        layout_summary=layout_summary,
        relation_graph=relation_graph,
    )

    multi_turn_samples = build_multi_turn_samples_for_file(
        image_path=image_path,
        image_metadata=image_metadata,
        by_class=by_class,
        layout_summary=layout_summary,
    )

    return shuffle_samples_for_image(single_turn_samples + multi_turn_samples, image_path)


# ============================================================
# Main
# ============================================================

def main():
    """Generate train and validation JSONL files."""
    stats = {
        "train_images": 0,
        "val_images": 0,
        "train_samples": 0,
        "val_samples": 0,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_samples = []
    val_samples = []

    for json_path in sorted(JSON_DIR.glob("*.json")):
        samples = build_samples_for_file(json_path)

        if not samples:
            continue

        split = split_bucket_for_json(json_path)

        if split == "train":
            stats["train_images"] += 1
            stats["train_samples"] += len(samples)
            train_samples.extend(samples)
        else:
            stats["val_images"] += 1
            stats["val_samples"] += len(samples)
            val_samples.extend(samples)

    random.Random(f"{SHUFFLE_SEED}:train").shuffle(train_samples)
    random.Random(f"{SHUFFLE_SEED}:val").shuffle(val_samples)

    with TRAIN_OUTPUT_JSONL.open("w", encoding="utf-8") as train_file:
        for sample in train_samples:
            train_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with VAL_OUTPUT_JSONL.open("w", encoding="utf-8") as val_file:
        for sample in val_samples:
            val_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total_images = stats["train_images"] + stats["val_images"]
    total_samples = stats["train_samples"] + stats["val_samples"]

    print(f"Saved {stats['train_samples']} samples to {TRAIN_OUTPUT_JSONL}")
    print(f"Saved {stats['val_samples']} samples to {VAL_OUTPUT_JSONL}")
    print(
        f"Image split: train={stats['train_images']} ({100 - VAL_SPLIT_PERCENT}%), "
        f"val={stats['val_images']} ({VAL_SPLIT_PERCENT}%)"
    )
    if total_samples and total_images:
        print(f"Average samples per image: {total_samples / total_images:.2f}")


if __name__ == "__main__":
    main()
