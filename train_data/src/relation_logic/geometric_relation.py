"""
Simple geometry helpers and relation-graph builder for floor-plan annotations.

This file keeps all polygon math and relation extraction away from data_generator.py
so the dataset-generation flow stays easier to read.
"""

from __future__ import annotations

import math


def polygon_area(points):
    """Return polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_center(points):
    """Return polygon center point."""
    if not points:
        return (0.0, 0.0)

    if len(points) < 3:
        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        return (x, y)

    signed_area = 0.0
    center_x = 0.0
    center_y = 0.0

    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        signed_area += cross
        center_x += (x1 + x2) * cross
        center_y += (y1 + y2) * cross

    if abs(signed_area) < 1e-9:
        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        return (x, y)

    signed_area *= 0.5
    center_x /= 6.0 * signed_area
    center_y /= 6.0 * signed_area
    return (center_x, center_y)


def polygon_box(points):
    """Return polygon bounding box as (min_x, min_y, max_x, max_y)."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def distance_between_points(point_a, point_b):
    """Return Euclidean distance between two points."""
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def distance_from_point_to_segment(point, segment_start, segment_end):
    """Return shortest distance from one point to one line segment."""
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return distance_between_points(point, segment_start)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projection = (x1 + t * dx, y1 + t * dy)
    return distance_between_points(point, projection)


def turn_type(point_a, point_b, point_c):
    """Return line-turn direction for three ordered points."""
    value = (point_b[1] - point_a[1]) * (point_c[0] - point_b[0]) - (
        point_b[0] - point_a[0]
    ) * (point_c[1] - point_b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def point_lies_on_segment(segment_start, point, segment_end):
    """Return True when point lies on the segment bounds."""
    return (
        min(segment_start[0], segment_end[0]) - 1e-9
        <= point[0]
        <= max(segment_start[0], segment_end[0]) + 1e-9
        and min(segment_start[1], segment_end[1]) - 1e-9
        <= point[1]
        <= max(segment_start[1], segment_end[1]) + 1e-9
    )


def segments_touch_or_cross(seg_a_start, seg_a_end, seg_b_start, seg_b_end):
    """Return True when two segments intersect or touch."""
    orient_1 = turn_type(seg_a_start, seg_a_end, seg_b_start)
    orient_2 = turn_type(seg_a_start, seg_a_end, seg_b_end)
    orient_3 = turn_type(seg_b_start, seg_b_end, seg_a_start)
    orient_4 = turn_type(seg_b_start, seg_b_end, seg_a_end)

    if orient_1 != orient_2 and orient_3 != orient_4:
        return True
    if orient_1 == 0 and point_lies_on_segment(seg_a_start, seg_b_start, seg_a_end):
        return True
    if orient_2 == 0 and point_lies_on_segment(seg_a_start, seg_b_end, seg_a_end):
        return True
    if orient_3 == 0 and point_lies_on_segment(seg_b_start, seg_a_start, seg_b_end):
        return True
    if orient_4 == 0 and point_lies_on_segment(seg_b_start, seg_a_end, seg_b_end):
        return True
    return False


def polygon_edges(polygon):
    """Yield every polygon edge."""
    if len(polygon) < 2:
        return
    for index in range(len(polygon)):
        yield polygon[index], polygon[(index + 1) % len(polygon)]


def point_inside_polygon(point, polygon):
    """Return True when point lies inside polygon."""
    x, y = point
    inside = False

    if len(polygon) < 3:
        return False

    last_index = len(polygon) - 1
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[last_index]
        crosses = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1
        )
        if crosses:
            inside = not inside
        last_index = index

    return inside


def polygons_touch_or_overlap(polygon_a, polygon_b):
    """Return True when polygons intersect, touch, or contain each other."""
    for seg_a_start, seg_a_end in polygon_edges(polygon_a):
        for seg_b_start, seg_b_end in polygon_edges(polygon_b):
            if segments_touch_or_cross(seg_a_start, seg_a_end, seg_b_start, seg_b_end):
                return True

    if polygon_a and point_inside_polygon(polygon_a[0], polygon_b):
        return True
    if polygon_b and point_inside_polygon(polygon_b[0], polygon_a):
        return True
    return False


def distance_between_polygons(polygon_a, polygon_b):
    """Return the shortest distance between two polygons."""
    if not polygon_a or not polygon_b:
        return float("inf")

    if polygons_touch_or_overlap(polygon_a, polygon_b):
        return 0.0

    min_distance = float("inf")

    for point in polygon_a:
        for seg_start, seg_end in polygon_edges(polygon_b):
            min_distance = min(
                min_distance,
                distance_from_point_to_segment(point, seg_start, seg_end),
            )

    for point in polygon_b:
        for seg_start, seg_end in polygon_edges(polygon_a):
            min_distance = min(
                min_distance,
                distance_from_point_to_segment(point, seg_start, seg_end),
            )

    return min_distance


def build_enriched_shapes(shapes):
    """Add simple geometry fields to raw shape records."""
    enriched = []
    for shape in shapes:
        polygon = shape["polygon"]
        enriched.append(
            {
                "id": len(enriched),
                "label": shape["label"],
                "polygon": polygon,
                "bbox": polygon_box(polygon),
                "center": polygon_center(polygon),
                "area": polygon_area(polygon),
            }
        )
    return enriched


def top_shapes_by_area(shapes, limit):
    """Return the largest shapes first."""
    ranked = sorted(shapes, key=lambda item: (-item["area"], item["label"], item["id"]))
    return ranked[:limit]


def build_relation_graph(
    shapes,
    room_labels,
    fixture_labels,
    room_adjacency_gap,
    door_room_gap,
    fixture_room_gap,
    window_wall_gap,
    max_objects_per_type,
):
    """
    Build a simple geometry-based relation graph.

    The result links:
    - room to nearby room
    - door to nearby rooms
    - fixture to room
    - window to nearby walls
    """
    enriched_shapes = build_enriched_shapes(shapes)
    rooms = [shape for shape in enriched_shapes if shape["label"] in room_labels]
    doors = [shape for shape in enriched_shapes if shape["label"] == "Door"]
    windows = [shape for shape in enriched_shapes if shape["label"] == "Window"]
    walls = [shape for shape in enriched_shapes if shape["label"] == "Wall"]
    fixtures = [shape for shape in enriched_shapes if shape["label"] in fixture_labels]

    relation_graph = {
        "rooms": [],
        "doors": [],
        "fixtures": [],
        "windows": [],
        "summary": {
            "room_adjacency_edges": 0,
            "door_room_links": 0,
            "fixture_room_links": 0,
            "window_wall_links": 0,
        },
    }

    room_neighbors = {room["id"]: [] for room in rooms}

    for index, room_a in enumerate(rooms):
        for room_b in rooms[index + 1:]:
            distance = distance_between_polygons(room_a["polygon"], room_b["polygon"])
            if distance <= room_adjacency_gap:
                room_neighbors[room_a["id"]].append(
                    {"room_id": room_b["id"], "distance": round(distance, 2)}
                )
                room_neighbors[room_b["id"]].append(
                    {"room_id": room_a["id"], "distance": round(distance, 2)}
                )
                relation_graph["summary"]["room_adjacency_edges"] += 1

    room_lookup = {room["id"]: room for room in rooms}

    for room in top_shapes_by_area(rooms, max_objects_per_type):
        nearby_rooms = sorted(
            room_neighbors[room["id"]],
            key=lambda item: (item["distance"], item["room_id"]),
        )
        relation_graph["rooms"].append(
            {
                "room_id": room["id"],
                "class": room["label"],
                "polygon": room["polygon"],
                "adjacent_rooms": [
                    {
                        "room_id": item["room_id"],
                        "class": room_lookup[item["room_id"]]["label"],
                        "distance": item["distance"],
                    }
                    for item in nearby_rooms
                ],
            }
        )

    for door in top_shapes_by_area(doors, max_objects_per_type):
        linked_rooms = []
        for room in rooms:
            distance = distance_between_polygons(door["polygon"], room["polygon"])
            if distance <= door_room_gap:
                linked_rooms.append(
                    {
                        "room_id": room["id"],
                        "class": room["label"],
                        "distance": round(distance, 2),
                    }
                )
        linked_rooms.sort(key=lambda item: (item["distance"], item["room_id"]))
        linked_rooms = linked_rooms[:2]

        relation_graph["doors"].append(
            {
                "door_id": door["id"],
                "polygon": door["polygon"],
                "connected_rooms": linked_rooms,
            }
        )
        relation_graph["summary"]["door_room_links"] += len(linked_rooms)

    for fixture in top_shapes_by_area(fixtures, max_objects_per_type):
        inside_rooms = []
        nearby_rooms = []

        for room in rooms:
            distance = distance_between_polygons(fixture["polygon"], room["polygon"])
            if point_inside_polygon(fixture["center"], room["polygon"]):
                inside_rooms.append(
                    {
                        "room_id": room["id"],
                        "class": room["label"],
                        "distance": round(distance, 2),
                    }
                )
            elif distance <= fixture_room_gap:
                nearby_rooms.append(
                    {
                        "room_id": room["id"],
                        "class": room["label"],
                        "distance": round(distance, 2),
                    }
                )

        inside_rooms.sort(key=lambda item: (item["distance"], item["room_id"]))
        nearby_rooms.sort(key=lambda item: (item["distance"], item["room_id"]))
        linked_rooms = inside_rooms or nearby_rooms[:1]

        relation_graph["fixtures"].append(
            {
                "fixture_id": fixture["id"],
                "class": fixture["label"],
                "polygon": fixture["polygon"],
                "rooms": linked_rooms,
            }
        )
        relation_graph["summary"]["fixture_room_links"] += len(linked_rooms)

    for window in top_shapes_by_area(windows, max_objects_per_type):
        nearby_walls = []
        for wall in walls:
            distance = distance_between_polygons(window["polygon"], wall["polygon"])
            if distance <= window_wall_gap:
                nearby_walls.append(
                    {"wall_id": wall["id"], "distance": round(distance, 2)}
                )

        nearby_walls.sort(key=lambda item: (item["distance"], item["wall_id"]))

        relation_graph["windows"].append(
            {
                "window_id": window["id"],
                "polygon": window["polygon"],
                "nearby_walls": nearby_walls,
            }
        )
        relation_graph["summary"]["window_wall_links"] += len(nearby_walls)

    return relation_graph
