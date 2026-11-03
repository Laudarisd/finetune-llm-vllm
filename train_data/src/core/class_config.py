"""
Shared class definitions for floor-plan dataset generation.
"""

ALL_CLASSES = [
    "AirRoom",
    "Balcony",
    "Basin",
    "Bathroom",
    "Bathtub",
    "Bedroom",
    "Commode",
    "Door",
    "DressingRoom",
    "Elevator",
    "Entrance",
    "Gas",
    "Hallway",
    "JunctionI",
    "JunctionL",
    "JunctionT",
    "JunctionX",
    "LivingKitchen",
    "OtherRoom",
    "Pantry",
    "Sink",
    "Stairs",
    "UtilityRoom",
    "Wall",
    "Window",
]

CLASS_NAME_MAP = {
    "AirRoom": "air room",
    "Balcony": "balcony",
    "Basin": "basin",
    "Bathroom": "bathroom",
    "Bathtub": "bathtub",
    "Bedroom": "bedroom",
    "Commode": "commode",
    "Door": "door",
    "DressingRoom": "dressing room",
    "Elevator": "elevator",
    "Entrance": "entrance",
    "Gas": "gas",
    "Hallway": "hallway",
    "JunctionI": "I-junction",
    "JunctionL": "L-junction",
    "JunctionT": "T-junction",
    "JunctionX": "X-junction",
    "LivingKitchen": "living kitchen",
    "OtherRoom": "other room",
    "Pantry": "pantry",
    "Sink": "sink",
    "Stairs": "stairs",
    "UtilityRoom": "utility room",
    "Wall": "wall",
    "Window": "window",
}

CLASS_GROUPS = {
    "junction": ["JunctionI", "JunctionL", "JunctionT", "JunctionX"],
    "walls and junctions": ["Wall", "JunctionI", "JunctionL", "JunctionT", "JunctionX"],
    "walls and windows": ["Wall", "Window"],
    "doors and windows": ["Door", "Window"],
    "structural elements": [
        "Wall", "Window", "Door", "Stairs", "Elevator",
        "JunctionI", "JunctionL", "JunctionT", "JunctionX",
    ],
    "room annotations": [
        "AirRoom", "Balcony", "Bathroom", "Bedroom", "DressingRoom",
        "Entrance", "Hallway", "LivingKitchen", "OtherRoom",
        "Pantry", "UtilityRoom",
    ],
    "bathroom fixtures": ["Basin", "Bathtub", "Commode", "Sink"],
    "service elements": ["Elevator", "Gas", "Sink", "Stairs", "UtilityRoom"],
}

ROOM_CLASSES = {
    "AirRoom", "Balcony", "Bathroom", "Bedroom", "DressingRoom",
    "Hallway", "LivingKitchen", "OtherRoom", "Pantry",
    "UtilityRoom", "Entrance",
}

STRUCTURAL_CLASSES = {
    "Wall", "Window", "Door", "Stairs", "Elevator",
    "JunctionI", "JunctionL", "JunctionT", "JunctionX",
}

FIXTURE_CLASSES = {
    "Basin", "Bathtub", "Commode", "Gas", "Sink",
}
