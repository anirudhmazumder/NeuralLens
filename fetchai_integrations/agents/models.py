"""
Shared message models for NeuralLens multi-agent pipeline.
All models use primitive types for reliable uAgents serialization.
Complex objects (dicts, nested lists) are JSON-stringified.
"""

from uagents import Model


class SensorRequest(Model):
    """Orchestrator → Sensor: start the analysis pipeline."""
    image_url: str           # direct URL to the image
    prompt: str              # user's optimization goal
    industry: str            # detected industry category
    asset_type: str          # detected asset type
    session_sender: str      # original ASI:One user address
                             # passed through entire chain
                             # so final result reaches user


class SensorResult(Model):
    """Sensor → Interpreter: raw model outputs."""
    image_url: str
    # Activations compressed: zlib.compress(float32 bytes) → base64 str.
    # ~70k float32 = 280KB raw → ~60KB compressed → safe for uAgents.
    # Decompress in interpreter: np.frombuffer(zlib.decompress(base64.b64decode(s)), np.float32)
    tribe_activations: str   # base64(zlib(float32 bytes))
    deepgaze_heatmap: str    # json.dumps(list) of 900 floats — small, keep as str
    heatmap_url: str         # Cloudinary URL of before heatmap
    prompt: str
    industry: str
    asset_type: str
    session_sender: str


class InterpreterResult(Model):
    """Interpreter → Strategist: NES scores and insights."""
    image_url: str
    nes_total: float         # 0-100 Neural Engagement Score
    roi_values: str          # json.dumps(dict) of 7 regions
    profile: str             # human readable emotion profile
    issues: str              # json.dumps(list) of issue strings
    insights: str            # json.dumps(list) of zone dicts
                             # each dict: zone, type, meaning,
                             # action, tribe_score, gaze_score
    valence: float           # -1.0 to 1.0
    arousal: float           # -1.0 to 1.0
    heatmap_url: str         # passed through from sensor
    prompt: str
    industry: str
    asset_type: str
    session_sender: str


class StrategistResult(Model):
    """Strategist → Executor: specific change instructions."""
    image_url: str
    visual_changes: str      # json.dumps(list of dicts)
    text_changes: str        # json.dumps(list of dicts)
    new_copy: str            # ready-to-paste caption/copy
    optimization_strategy: str
    nes_total: float         # before score, passed through
    profile: str             # before profile, passed through
    issues: str              # passed through from interpreter
    heatmap_url: str         # before heatmap, passed through
    session_sender: str


class FinalResult(Model):
    """Executor → Orchestrator: completed analysis result."""
    nes_before: float
    nes_after: float
    delta: float
    profile_before: str
    profile_after: str
    issues: str              # json.dumps(list)
    changes: str             # json.dumps(list) combined changes
    optimized_image_url: str
    heatmap_before_url: str
    heatmap_after_url: str   # after heatmap or optimized image
    new_copy: str
    optimization_strategy: str
    session_sender: str      # used by orchestrator to route
                             # result back to original user
