"""Print all 5 agent addresses without starting any server."""
from uagents import Agent

agents = [
    ("ORCHESTRATOR", "neurallens orchestrator lahacks 2026 neuromarketing main"),
    ("SENSOR",       "neurallens sensor tribe deepgaze lahacks 2026 brain"),
    ("INTERPRETER",  "neurallens interpreter nes roi scorer lahacks 2026"),
    ("STRATEGIST",   "neurallens strategist gemma optimizer lahacks 2026"),
    ("EXECUTOR",     "neurallens executor cloudinary image lahacks 2026"),
]

print()
for name, seed in agents:
    a = Agent(name=name, seed=seed)
    print(f"{name}_ADDRESS={a.address}")
print()
