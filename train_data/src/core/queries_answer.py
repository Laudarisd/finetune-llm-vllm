"""
Query and reply template constants for dataset generation.

Keeping these in one file makes data_generator.py easier to read and edit.
"""

# ============================================================
# Single-turn query templates
# ============================================================

IMAGE_LEVEL_QUERIES = [
    "summarize this floor plan with room types, structural elements, fixtures, and counts",
    "list all classes present in this floor plan and their counts",
    "which classes are present in this floor plan and which classes are absent",
    "how many total annotated polygons are in this image and which classes do they belong to",
    "give me an overall description of this floor plan layout",
    "I need to know how many classes are there",
    "Can you count how many classes are there?",
    "Can you see uploaded image?",
    "Can you see uplaoded image?",
    "what classes can you see in this image",
    "what annotations are present in this floor plan",
    "how many classes are present in this floor plan",
    "give me a quick summary of this image",
    "what do you observe in this floor plan image",
    "can you list the visible room and object classes",
    "what can you see in this floor plan",
    "can you explain this floor plan",
    "what is shown in this image",
    "what rooms and objects are visible here",
    "can you analyze this plan for me",
    "please inspect this floor plan",
    "tell me what is present in this image",
    "what labels are visible in this floor plan",
    "what classes are detected in this image",
    "can you give me a full overview of this plan",
    "what is inside this floor plan",
    "which rooms are visible here",
    "which objects are visible here",
    "can you identify the visible annotations",
    "what elements are annotated in this image",
    "can you read this floor plan",
    "what does this plan contain",
    "can you tell me all visible categories",
    "show me all detected classes in this floor plan",
    "what are the main components in this floor plan",
    "can you summarize all visible annotations",
]

OBJECT_QUERY_TEMPLATES = [
    "analyze the {object} in this floor plan: tell me whether it is present, how many instances exist, and return all polygons if present",
    "for the {object} class, report presence, count, and every annotated polygon in this image",
    "inspect this plan for {object}: confirm whether it appears, give the total count, and provide the full polygon list",
    "can you count all {object} in this floor plan and return their polygons",
    "please count the {object} instances and show each annotation polygon",
    "count how many {object} are in this image and list all polygon coordinates",
    "can you detect {object} in this floor plan",
    "can you find all {object} in this image",
    "show me all {object} in this floor plan",
    "where are the {object} in this image",
    "return the polygon points for all {object}",
    "mark all {object} in this floor plan",
    "locate every {object} in this image",
    "tell me if {object} are present and show their polygons",
    "please identify all {object} and return their coordinates",
    "find the {object} and give me all annotation polygons",
    "can you check where the {object} are",
    "highlight all {object} annotations",
    "which {object} are visible and where are they located",
    "give me all {object} polygons from this floor plan",
    "show the annotation points for each {object}",
    "can you detect and count the {object}",
    "please inspect the image and return all {object} polygons",
    "find every {object} and list their polygon coordinates",
    "can you see any {object} here and show their shapes",
]

OBJECT_CONCISE_QUERY_TEMPLATES = [
    "is there any {object} in this floor plan",
    "how many {object} are visible in this floor plan",
    "does this floor plan contain {object}",
    "what is the total number of {object} in this floor plan",
    "can you count the {object} in this floor plan",
    "count the number of {object} in this image",
    "can you tell me how many {object} are in this floor plan",
    "can you see any {object}",
    "are there any {object} here",
    "do you see {object} in this image",
    "how many {object} are there",
    "how many {object} can you see",
    "can you find {object} in this floor plan",
    "does this image have any {object}",
    "is {object} present in this image",
    "is {object} available in this floor plan",
    "what number of {object} do you see",
    "tell me the count of {object}",
    "please count the {object}",
    "can you check for {object}",
    "do we have {object} here",
    "is there {object} in this image",
    "how many {object} are present",
    "can you detect {object}",
    "please tell me if {object} are visible",
    "can you confirm whether {object} exist here",
    "are {object} present in this floor plan",
    "can you identify {object} in this image",
    "do you find any {object} here",
    "what is the count for {object}",
]

GROUP_QUERY_TEMPLATES = [
    "focus on {group} in this floor plan: say whether any are present, give counts by class, and return all annotations",
    "analyze only the {group} in this image and provide presence, total count, class-wise counts, and polygons",
    "are there any {group} in this floor plan",
    "how many {group} are there here",
    "can you check {group} in this image",
    "tell me if {group} are present",
    "can you detect {group} in this floor plan",
    "show me the {group} in this image",
    "which {group} are present here",
    "please inspect the {group} in this floor plan",
    "can you analyze {group} only",
    "give me the details for {group}",
    "what {group} can you see in this image",
    "do we have any {group} here",
    "can you count the {group}",
    "return all polygons for {group}",
    "show the annotations for {group}",
    "can you list all visible {group}",
    "what is the total count for {group}",
    "can you break down the {group} by class",
    "highlight the {group} in this floor plan",
    "tell me which classes inside {group} are visible",
    "can you check whether {group} exist in this image",
    "give me all annotation polygons for {group}",
]

COMPARISON_TEMPLATES = [
    "compare {object_a} and {object_b} in this floor plan and tell me which appears more often",
    "between {object_a} and {object_b}, which class has the higher count in this image",
    "which is more in this plan: {object_a} or {object_b}",
    "are there more {object_a} than {object_b}",
    "can you compare {object_a} and {object_b}",
    "what appears more: {object_a} or {object_b}",
    "which is greater in number, {object_a} or {object_b}",
    "which one appears more, {object_a} or {object_b}",
    "do we have more {object_a} than {object_b}",
    "can you tell me whether {object_a} or {object_b} is more common",
    "which class is more frequent: {object_a} or {object_b}",
    "what has the higher count in this image, {object_a} or {object_b}",
    "between {object_a} and {object_b}, which one is larger in count",
    "can you check which is more: {object_a} or {object_b}",
    "which do you see more often, {object_a} or {object_b}",
    "is {object_a} more than {object_b}",
    "is {object_b} more than {object_a}",
    "tell me which appears less: {object_a} or {object_b}",
    "which one has fewer instances, {object_a} or {object_b}",
    "compare the counts of {object_a} and {object_b}",
]

RELATION_GRAPH_QUERIES = [
    "describe the room adjacency, door links, fixture placements, and window-wall relations in this floor plan",
    "which rooms are next to each other",
    "which rooms are connected by doors",
    "where are the fixtures placed",
    "which windows are near walls",
    "can you explain room connections in this floor plan",
    "can you explain how the rooms are connected",
    "which rooms are adjacent in this floor plan",
    "tell me the neighboring rooms in this layout",
    "which spaces are next to each other",
    "which rooms share boundaries",
    "what rooms are connected through doors",
    "can you tell me which rooms are linked by doors",
    "how are the rooms connected in this plan",
    "which room connects to which",
    "show me the room connections in this floor plan",
    "where are the doors connecting rooms",
    "which fixtures belong to which rooms",
    "tell me where the fixtures are placed",
    "which rooms contain fixtures",
    "where do the sink, basin, commode, bathtub, and gas belong",
    "which windows are attached to or near walls",
    "can you explain the window and wall relationship",
    "what is the relation between windows and walls here",
    "can you build the room relation graph for this floor plan",
    "give me the structural and room relationship graph",
    "describe the adjacency graph of this floor plan",
    "can you summarize all room, door, fixture, and window relations",
]

# ============================================================
# Multi-turn query templates
# ============================================================

IMAGE_TYPE_QUERIES = [
    "what kind of image is this",
    "what type of drawing is this",
    "what does this image show",
    "is this a floor plan",
    "can you tell what this image is",
    "what am i looking at here",
    "can you explain this image",
    "what is shown here",
]

FOLLOWUP_COUNT_PROMPTS = [
    "how many",
    "what is the total number",
    "how many are there in total",
    "can you give me the count",
    "tell me the exact count",
    "how many of them are there",
]

FOLLOWUP_COMPARE_PROMPTS = [
    "which is more",
    "which appears more often",
    "which one has the higher count",
    "which one is greater in number",
    "which has more in this plan",
    "which one has fewer",
]

FOLLOWUP_DETAIL_PROMPTS = [
    "can you count them",
    "give me the total count",
    "can you provide the exact number",
    "give me a precise count",
    "tell me the full count",
]

OBJECT_PRESENCE_MULTI_TEMPLATES = [
    "is there {object}",
    "are there any {object}",
    "can you see any {object}",
    "is {object} present in this plan",
    "do you find any {object} here",
    "does this floor plan contain {object}",
]

OBJECT_COUNT_MULTI_TEMPLATES = [
    "can you count the {object}",
    "count how many {object} are in this floor plan",
    "what is the exact number of {object} here",
    "how many {object} can you see",
    "please count all {object}",
    "what is the total count of {object}",
]

FOLLOWUP_POLYGON_PROMPTS = [
    "show me their polygons",
    "can you return all polygon points",
    "list all annotation polygons for them",
    "can you share the coordinates",
    "give me all polygon points",
    "show me where they are",
    "draw them for me",
    "highlight them",
]

GROUP_PRESENCE_MULTI_TEMPLATES = [
    "are there any {group} here",
    "does this image contain {group}",
    "can you check if {group} are present",
    "do you see {group} in this plan",
    "which {group} are visible here",
]

SUMMARY_FOLLOWUP_OBJECT_PROMPTS = [
    "is there {object}",
    "is there any {object}",
    "are there any {object}",
    "do you see {object}",
    "is {object} present",
    "can you check {object}",
    "does this plan contain {object}",
]

YES_NO_VARIANTS = {
    True: [
        "Yes.",
        "Yes, it is present.",
        "Yes, there are some visible.",
        "Yes, present in this floor plan.",
        "Yes, I can see them.",
    ],
    False: [
        "No.",
        "No, it is not present.",
        "No, none are visible.",
        "No, I do not see any.",
        "No, not present in this plan.",
    ],
}
