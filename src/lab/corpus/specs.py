"""What a synthetic call is made of: its domain, caller, events, TTS, policy and deployment LLM.

Every choice comes from a generator seeded by the spec's seed, so a spec can be made again.
"""

import hashlib
import random
from dataclasses import asdict, dataclass

from pipecat_tone.tagger.labels import TTS_ADAPTERS, label_map

# tone-bench's families are left out entirely: banking, cards and loans; insurance; airlines, hotels and
# travel; food delivery and restaurants; internet, cable and phone; health care.
DOMAINS = [
    "electric and gas utility: bills, outages, starting and stopping service",
    "water utility: leaks, meter readings, payment plans",
    "online retailer: returns, refunds and exchanges",
    "parcel courier: missed deliveries, redelivery, tracking",
    "furniture store: delivery scheduling and damaged items",
    "home appliance repair: booking a technician, warranty coverage",
    "plumbing and heating company: emergency and routine service visits",
    "moving company: quotes, bookings and damage claims",
    "gym membership: cancellations, freezes and billing",
    "video streaming subscription: account access and billing",
    "business software support: logins, billing and outages",
    "university registrar: enrollment, transcripts and fees",
    "motor vehicle office: license renewals, appointments and titles",
    "tax preparation service: appointments, documents and refund status",
    "apartment property management: maintenance requests, rent and leases",
    "car dealership service department: repairs and recalls",
    "independent auto repair shop: estimates, repair status and pickup",
    "city services line: parking tickets, trash pickup and permits",
    "public library: holds, fines and card renewals",
    "concert and event ticketing: transfers, refunds and cancelled shows",
    "rideshare support: lost items, fare disputes and driver complaints",
    "consumer electronics warranty and repairs",
    "home security system provider: alarms, installation and monitoring",
    "solar panel installer: installation scheduling and performance problems",
    "school district office: enrollment, bus routes and lunch accounts",
    "employer HR helpdesk: payroll questions and time off",
    "corporate IT helpdesk: password resets and laptop problems",
    "wedding and event venue: bookings and changes",
    "florist and gift delivery: orders and delivery problems",
    "custom bakery: cake orders and pickups",
    "bicycle shop: repairs and rentals",
    "video game platform support: account recovery and purchases",
    "charity donor services: donations, receipts and recurring gifts",
    "dry cleaning and laundry service: pickups and lost items",
    "pet grooming and boarding: bookings and pickups",
    "postal mail forwarding and PO boxes",
]

# Domains held out for model selection and final checks; the rest train.
DEV_DOMAINS = {DOMAINS[i] for i in (3, 14, 23, 32)}
TEST_DOMAINS = {DOMAINS[i] for i in (7, 19, 27, 35)}

PERSONAS = [
    "an older caller who takes their time and likes things confirmed twice",
    "a busy professional calling between meetings, brisk and to the point",
    "an anxious first-timer who worries about getting things wrong",
    "a chatty, good-humored caller who jokes around",
    "a curt, impatient caller who wants this over with",
    "a caller who speaks English as a second language, in simple phrasing",
    "a dry, sarcastic caller who has been let down before",
    "a polite caller who is quietly upset",
    "a caller who is angry from the first sentence",
    "a cheerful caller with good news to share",
    "a tired parent juggling kids in the background",
    "a confused caller who mixes up details and needs correcting",
]

MOOD_ARCS = [
    "starts calm and stays calm",
    "starts frustrated and calms down once helped",
    "starts upset and gets angrier before it resolves",
    "starts cheerful and stays warm",
    "starts worried and ends relieved",
    "starts neutral, gets bad news, ends resigned",
    "starts hostile and stays hostile",
    "starts sarcastic and softens",
]

EVENTS = [
    "the caller tells a joke or says something funny",
    "the caller is sarcastic at least once",
    "the caller becomes hostile or rude",
    "the caller asks the agent to slow down",
    "the caller asks the agent to repeat something",
    "the caller reads a code or number back correctly",
    "the caller reads a code or number back wrong",
    "the caller corrects the agent",
    "the caller gets bad news",
    "the caller gets good news",
    "the caller thanks the agent warmly",
    "the caller's reading disagrees with their words",
]

# Contrast cases: situations where the obvious rule is wrong, so the tagger learns the
# exceptions too. Corpus v1 calls (seeds from V1_START) each plant one.
CONTRAST = [
    "the agent gives the caller a code to write down, and later in the call mentions it again in passing",
    "early on the caller asks the agent to slow down, and much later the agent gives another detail to write down",
    "the caller reads a code back correctly and the agent only confirms it",
    "the caller is hostile, clearly frustrated (0.85 or above) for several turns in a row",
    "the caller's words are neutral but their reading is a weak guess at frustration (below 0.5)",
    'the caller says something like "great, perfect" with high sarcasm (0.75 or above)',
    "the caller makes a clear joke",
    "the agent has to give the caller bad news about what they asked for",
    "the caller is upset early, the agent apologizes, and later the caller complains about the same problem again",
]
V1_START = 1000
# v2 asks some calls to breathe (the BREATHE preset), and gives every call a moment a breath could fit.
V2_START = 3000
BREATHE_SHARE = 0.5
BREATH_MOMENTS = [
    "the agent has to explain a long, multi-step process the caller must follow",
    "the agent has to give hard news, such as a refused claim, a cancelled order or a long delay",
    "the agent reads out several options before the caller chooses one",
    "the caller is upset and talks fast, and the agent answers calmly at length",
    "the agent has to walk the caller through troubleshooting, step by step",
    "the agent apologizes for a serious mistake, then explains how it will be put right",
]

# v3 plants a moment for a delivery the teacher seldom uses, as those are where the tagger is weakest. Each describes
# a situation, not a label, so the teacher still decides how it's voiced.
V3_START = 4000
RARE_MOMENTS = [
    "the caller shares happy personal news, such as a wedding, a new job or a baby, and the agent is glad for them",
    "the caller thanks the agent from the heart for going out of their way to help",
    "the call is about a bereavement or a loss, such as closing the account of someone who has died",
    "the caller fears a fix won't work, and the agent knows it will and reassures them firmly",
    "the caller describes an odd problem, and the agent asks interested questions to get to the bottom of it",
    "the caller reveals something unexpected that changes the situation, and the agent reacts to it",
    "the caller is in a rush and asks the agent to keep it quick",
    "the agent reads out a long address, code or set of steps the caller has to write down",
]

# The developer's own guidance lines; MATCH_STYLE and SLOW_FOR_DETAILS are always on.
GUIDANCE = [
    "If the caller is clearly upset, apologize once, then focus on fixing the problem.",
    "If sarcasm is 0.7 or above, answer what the caller means, not their literal words.",
    "Give bad news plainly and kindly, then explain what happens next.",
    "Keep a steady, reassuring voice with anxious callers.",
    "Stay professional; light humor is fine if the caller starts it.",
    "Never sound cheerful about a problem the caller is having.",
    "With callers who sound confused, confirm each detail before moving on.",
]

# Deployment LLMs from tone-bench's models.yaml, weighted toward phonellm-mini, the one the tagger is built for.
AGENT_MODELS = {
    "phonellm-mini": 0.35,
    "gpt-4.1": 0.2,
    "qwen3.5-35b-a3b": 0.15,
    "nemotron-3.5-bolt": 0.15,
    "phonellm-max": 0.15,
}

TTS_WEIGHTS = {"cartesia": 0.25, "elevenlabs": 0.25, "fish": 0.25, "minimax": 0.25}


@dataclass
class Spec:
    """Everything chosen for one call before its scenario is written."""

    id: str
    seed: int
    domain: str
    split: str
    persona: str
    mood_arc: str
    events: list[str]
    turns: int
    tts: str
    delivery: list[str] | None
    sounds: list[str] | None
    guidance: list[str]
    agent_model: str
    breathe: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def make_spec(seed: int) -> Spec:
    rng = random.Random(seed)
    domain = rng.choice(DOMAINS)
    tts = _weighted(rng, TTS_WEIGHTS)
    delivery, sounds = _policy_names(rng, tts)
    spec = Spec(
        id=f"c{seed:05d}-{hashlib.sha1(str(seed).encode()).hexdigest()[:6]}",
        seed=seed,
        domain=domain,
        split="dev" if domain in DEV_DOMAINS else "test" if domain in TEST_DOMAINS else "train",
        persona=rng.choice(PERSONAS),
        mood_arc=rng.choice(MOOD_ARCS),
        events=rng.sample(EVENTS, rng.randint(2, 5)),
        turns=max(6, min(28, int(rng.gauss(14, 5)))),
        tts=tts,
        delivery=delivery,
        sounds=sounds,
        guidance=rng.sample(GUIDANCE, rng.choice([0, 1, 1, 2])),
        agent_model=_weighted(rng, AGENT_MODELS),
    )
    if seed >= V1_START:
        # Its own random stream, so every field above is what it was in corpus v0.
        spec.events.insert(0, random.Random(f"contrast-{seed}").choice(CONTRAST))
    if seed >= V3_START:
        spec.events.insert(0, random.Random(f"rare-{seed}").choice(RARE_MOMENTS))
    if seed >= V2_START:
        breath = random.Random(f"breath-{seed}")
        spec.events.append(breath.choice(BREATH_MOMENTS))
        spec.breathe = breath.random() < BREATHE_SHARE
        names = label_map().sounds["breath"].get(tts) or []
        if spec.breathe and spec.sounds is not None and names and breath.random() < 0.85:
            spec.sounds = sorted({*spec.sounds, names[0]})
    return spec


def _policy_names(rng: random.Random, tts: str) -> tuple[list[str] | None, list[str] | None]:
    """The policy's delivery and sound lists: mostly names the label map uses, some others, sometimes all."""
    caps = TTS_ADAPTERS[tts]().capabilities()
    labels = label_map()
    declared = [t.name for t in caps.delivery]
    mapped = sorted({n for by in labels.deliveries.values() for n in by.get(tts, [])})
    pool = [n for n in mapped if caps.free_text_delivery or n in declared]
    if rng.random() < 0.1:
        delivery = None
    else:
        chosen = {n for n in pool if rng.random() < 0.7} | {"slow", "fast"}
        chosen |= set(rng.sample([n for n in declared if n not in chosen], rng.randint(0, 3)))
        delivery = sorted(chosen)
    sound_names = [t.name for t in caps.sounds]
    # Breath is offered by v2's BREATHE specs, on a stream of its own.
    mapped_sounds = sorted({n for k, by in labels.sounds.items() if k != "breath" for n in by.get(tts, [])})
    if rng.random() < 0.15:
        sounds = None
    else:
        sounds = sorted(
            {n for n in mapped_sounds if rng.random() < 0.6}
            | set(rng.sample(sound_names, rng.randint(0, min(2, len(sound_names)))))
        )
    return delivery, sounds


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]
