"""
Constants for the RE (Relationship Extraction) module.
"""

# Default values for extraction settings
DEFAULT_SUMMARY_LANGUAGE = "English"
DEFAULT_MAX_GLEANING = 1
DEFAULT_ENTITY_NAME_MAX_LENGTH = 256

# Number of description fragments to trigger LLM summary
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8
# Max description token size to trigger LLM summary
DEFAULT_SUMMARY_MAX_TOKENS = 1200
# Recommended LLM summary output length in tokens
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
# Maximum token size sent to LLM for summary
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000

# Default entities to extract if ENTITY_TYPES is not specified
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
]

# Hierarchical type configuration
DEFAULT_USE_HIERARCHICAL_TYPES = True  # Enable hierarchical type support
DEFAULT_TYPE_MODE = "fine"  # Options: "coarse", "fine", "both"
DEFAULT_MAX_TYPES_FOR_PROMPT = None  # Maximum types to include in prompts

# Type extraction modes
TYPE_MODE_COARSE_ONLY = "coarse"
TYPE_MODE_FINE_ONLY = "fine"
TYPE_MODE_BOTH = "both"
TYPE_MODE_AUTO = "auto"  # Automatically choose based on context

# Default relationships to extract if RELATIONSHIP_TYPES is not specified
# If None, extract all relationships found in the text
DEFAULT_RELATIONSHIP_TYPES = None  # None means extract all

# Separator for graph fields
GRAPH_FIELD_SEP = "<SEP>"

# Default temperature for LLM
DEFAULT_TEMPERATURE = 1.0

# Async configuration defaults
DEFAULT_MAX_ASYNC = 4  # Default maximum async operations

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Default llm and embedding timeout
DEFAULT_LLM_TIMEOUT = 180
DEFAULT_EMBEDDING_TIMEOUT = 30

# Default max source IDs per entity/relation
DEFAULT_MAX_SOURCE_IDS_PER_ENTITY = 300
DEFAULT_MAX_SOURCE_IDS_PER_RELATION = 300

# Source IDs limitation methods
SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
DEFAULT_SOURCE_IDS_LIMIT_METHOD = SOURCE_IDS_LIMIT_METHOD_FIFO

# Maximum number of file paths stored in entity/relation file_path field
DEFAULT_MAX_FILE_PATHS = 100
