"""r1's voice-agent test set, never trained on: conversations between an AI voice agent and a person that a teacher
model writes from the prompts built here, three lines taken from each. Pure functions, run on Modal by
``lab.reader.pipeline.voice_generate``."""

import hashlib
import json
import random
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from lab.reader.schemas import MAX_PREVIOUS, MAX_TEXT, PLAYER, LineId, LineInput

# --- Tone's corpus lines -----------------------------------------------------------------------------------------------

SOURCE = "generated"
LICENCE = "ours (teacher Qwen3.8-27B)"
# The teacher that writes the generated sets, pinned.
TEACHER_REPO = "Qwen/Qwen3.8-27B-FP8"
TEACHER_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
MAX_WORDS = 30
URL = re.compile(r"https?://|www\.|\.com\b", re.I)


class CorpusLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_id: LineId
    source: Literal["generated"]
    source_ref: str
    licence: str
    split: Literal["train", "dev", "test"]
    group_key: str
    input: LineInput
    slices: list[str] = []
    twin_of: str | None = None
    generator: str | None = None


def short_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def clean_text(text: str) -> str | None:
    t = re.sub(r"\s+", " ", text).strip()
    if not t or URL.search(t) or len(t) > MAX_TEXT or len(t.split()) > MAX_WORDS:
        return None
    if re.search(r"[<>{}\[\]*_#|]", t.replace("[NAME]", "")):  # markup, tables, markdown
        return None
    return t


def norm_key(text: str) -> str:
    """Lowercase a to z, digits and spaces only, the ends trimmed: the key the set's texts are deduplicated by."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# --- the set -----------------------------------------------------------------------------------------------------------

PROMPT_PREFIX = "tv1"
SEED = 501
SEEDS = ("sarcasm", "lookalike", "none")  # one per conversation, interleaved in prompt order
ASR_SHARE = 1 / 3  # conversations whose person side reads as speech-to-text output
OPENER_SHARE = 0.05  # agent lines taken from the agent's opening turn (no line before)
AFTER_MARK_SHARE = 0.25  # agent lines taken from the turn right after the marked one
MAX_PER_TEXT = 2  # a text at most twice in the set; a (text, line before) pair once
# Conversations taken per domain by seed, those with a usable mark first.
QUOTA = {"sarcasm": 8, "lookalike": 8, "none": 7}

SYSTEM = (
    "You write realistic transcripts of conversations between people and AI voice agents, for a dataset. People on "
    "the phone rarely talk like a book: they are brief, they hesitate, they wander."
)

AGENT_NAMES = (
    "Ava",
    "Sam",
    "Maya",
    "Leo",
    "Nora",
    "Theo",
    "Iris",
    "Max",
    "Zoe",
    "Kai",
    "Ruby",
    "Eli",
    "Jade",
    "Finn",
    "Mila",
    "Owen",
    "Tess",
    "Jonah",
    "Lena",
    "Ravi",
    "Priya",
    "Omar",
    "Hana",
    "Luca",
)

# Each domain: `intro` names the two parties (after "between"), `first` who speaks first, `scenarios` what happens,
# `setbacks` and `good_news` what the agent says before a sarcasm or look-alike seed. Companies are made up.
DOMAINS: dict[str, dict] = {
    "isp_support": {
        "companies": ("Northline Broadband", "Brightwave Internet"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the support line of {company}, a home broadband provider, and a customer who phones in",
        "scenarios": (
            "the person's internet drops every evening around eight",
            "the person's speed is far below what they pay for",
            "every light on the person's router went red after a storm",
            "the person moved house and the service was never switched on",
            "the person's wifi does not reach the upstairs rooms",
            "the person was charged for a router they sent back",
            "the person wants to know why the internet was down all yesterday",
            "the person cannot log in to see their bill",
        ),
        "setbacks": (
            "the earliest engineer visit is in nine days",
            "the wait to speak to a human advisor is about forty minutes",
            "there is a fee to send a new router",
            "there is an outage in the area with no time for a fix yet",
            "the speed in the contract is only an 'up to' speed",
        ),
        "good_news": (
            "an engineer can come tomorrow morning",
            "a credit will be taken off the next bill",
            "the line has been reset remotely and is working again",
            "a new router is on its way free of charge",
        ),
    },
    "bank_card": {
        "companies": ("Harbor Bank", "Meridian Bank"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the card services line of {company}, and a customer who phones in",
        "scenarios": (
            "the person's card was declined at the supermarket although there is money in the account",
            "the person lost their card on a night out",
            "the person sees a payment they do not recognise",
            "the person's new card has not arrived after two weeks",
            "the person is travelling abroad next week and wants the card to work",
            "a cash machine kept the person's card",
            "the person forgot their PIN",
            "the person was charged twice for one purchase",
        ),
        "setbacks": (
            "a replacement card takes seven to ten working days",
            "the card is frozen and cannot be unfrozen over the phone",
            "a disputed payment can take up to forty-five days to look into",
            "the person has to visit a branch with photo ID",
        ),
        "good_news": (
            "the card is unblocked and works right away",
            "the double charge is refunded today",
            "a new card will arrive tomorrow by express post",
            "the strange payment was stopped before it went through",
        ),
    },
    "airline_rebooking": {
        "companies": ("Skyward Air", "Bluecrest Airlines"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the bookings line of {company}, an airline, and a passenger who phones in",
        "scenarios": (
            "the person's flight was cancelled the night before a wedding",
            "the person missed a connection because the first flight was late",
            "the person wants to fly a day earlier",
            "the person's bag did not arrive with them",
            "the person's flight is delayed by six hours",
            "the person booked the wrong date by mistake",
            "the person wants to know if they can bring their dog",
            "the person needs a wheelchair at the airport for their mother",
        ),
        "setbacks": (
            "the next free seat is in two days",
            "changing the flight costs a hundred and twenty dollars",
            "the only seat left is a middle seat at the back with two stops",
            "the bag is still in another city",
            "the delay is caused by weather, so there is no compensation",
        ),
        "good_news": (
            "there is a seat on a flight in two hours at no extra cost",
            "the person has been upgraded for free",
            "the bag was found and will be delivered tonight",
            "the change fee is waived",
        ),
    },
    "restaurant_booking": {
        "companies": ("Olive and Ash", "The Copper Pot"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent that answers the phone at {company}, a restaurant, and a person who phones in",
        "scenarios": (
            "the person wants a table for six on Saturday night",
            "the person wants to book a birthday dinner and asks about a cake",
            "the person needs to cancel tonight's booking",
            "the person asks whether there are vegan and gluten-free dishes",
            "the person wants to move their booking to later",
            "the person is running late for their booking",
            "the person wants a quiet table for an anniversary",
            "the person asks about parking and step-free access",
        ),
        "setbacks": (
            "Saturday is full until ten at night",
            "the private room needs a large deposit",
            "the kitchen closes in twenty minutes",
            "the table can only be held for fifteen minutes",
        ),
        "good_news": (
            "a table by the window has just opened up",
            "the restaurant will have a small cake ready at no charge",
            "the booking has been moved",
            "there is free parking right behind the building",
        ),
    },
    "doctor_appointment": {
        "companies": ("Riverside Medical Practice", "Elm Street Clinic"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent that books appointments at {company}, and a patient who phones in",
        "scenarios": (
            "the person needs an appointment for a cough that will not go away",
            "the person wants a check-up for their child",
            "the person needs a repeat prescription",
            "the person wants their test results from last week",
            "the person needs to move an appointment",
            "the person wants a same-day appointment for a rash",
            "the person is phoning for their elderly father",
            "the person wants to register as a new patient",
        ),
        "setbacks": (
            "the next appointment is in three weeks",
            "test results cannot be given over the phone",
            "the doctor the person likes is away for a month",
            "repeat prescriptions take three working days",
        ),
        "good_news": (
            "there is a cancellation this afternoon",
            "the results are back and everything is normal",
            "the prescription is ready to collect today",
            "a nurse can call back within the hour",
        ),
    },
    "food_delivery": {
        "companies": ("QuickBite", "DashPlate"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the help line of {company}, a food delivery app, and a customer who phones in",
        "scenarios": (
            "the person's order is an hour late",
            "the person got someone else's order",
            "items are missing from the person's order",
            "the food arrived cold and spilled",
            "the driver marked the order delivered but it never came",
            "the person was charged for an order that was cancelled",
            "the restaurant ignored the person's allergy note",
            "the person wants to change the delivery address after ordering",
        ),
        "setbacks": (
            "the refund can only be app credit, not money back",
            "the restaurant says the order was complete",
            "the driver is another forty minutes away",
            "only the delivery fee can be refunded",
        ),
        "good_news": (
            "a full refund is on its way",
            "a fresh order is coming at no charge",
            "a credit has been added to the account",
            "the driver is just around the corner",
        ),
    },
    "laptop_support": {
        "companies": ("Volt Computers", "Apex Laptops"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the support line of {company}, a laptop maker, and a customer who phones in",
        "scenarios": (
            "the person's laptop will not turn on",
            "the person's laptop is very slow after an update",
            "the person's screen flickers",
            "the person spilled coffee on the keyboard",
            "the person's battery dies in an hour",
            "the person's laptop will not connect to wifi",
            "the person forgot their login password",
            "the person's laptop fan is loud and it gets hot",
        ),
        "setbacks": (
            "the repair takes three weeks",
            "liquid damage is not covered by the warranty",
            "the warranty ran out last month",
            "the fix wipes the laptop, so everything must be backed up first",
            "the advice is to restart the laptop, which the person already did",
        ),
        "good_news": (
            "the repair is covered by the warranty",
            "a quick setting change fixes it",
            "a new battery ships free tomorrow",
            "the person can keep all their files",
        ),
    },
    "utility_bill": {
        "companies": ("Greenfield Energy", "Clearwater Utilities"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the billing line of {company}, an energy and water supplier, and a customer who phones in",
        "scenarios": (
            "the person's bill is three times higher than usual",
            "the person was billed for a flat they moved out of",
            "the person wants to set up a payment plan",
            "the person's meter reading looks wrong",
            "the person got a final warning for a bill they already paid",
            "the person wants a cheaper tariff",
            "the person's direct debit went up without warning",
            "the person's smart meter stopped working",
        ),
        "setbacks": (
            "the bill was based on an estimate and is correct",
            "a meter test costs a fee if the meter turns out fine",
            "the cheaper tariff has an exit fee",
            "a dispute takes up to eight weeks",
        ),
        "good_news": (
            "the bill was wrong and has been corrected",
            "the overpayment will be refunded this week",
            "the payment plan is set up with no extra charges",
            "the warning letter was a mistake and is cancelled",
        ),
    },
    "insurance_claim": {
        "companies": ("Safeharbor Insurance", "Keystone Insure"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the claims line of {company}, a car and home insurer, and a customer who phones in",
        "scenarios": (
            "someone reversed into the person's parked car",
            "a burst pipe flooded the person's kitchen",
            "the person's bike was stolen from the garden",
            "the person wants news on a claim made a month ago",
            "the person's windscreen cracked on the motorway",
            "the person dropped and broke their phone",
            "the person's claim was turned down and they do not understand why",
            "the person wants to add a new driver to the policy",
        ),
        "setbacks": (
            "the excess is more than the repair costs",
            "the claim needs photos and three quotes first",
            "the claim was turned down because the bike was not locked",
            "the premium will go up next year",
        ),
        "good_news": (
            "the claim is approved",
            "a repair van can come tomorrow",
            "the excess is waived",
            "the money will be paid by Friday",
        ),
    },
    "retail_return": {
        "companies": ("Northgate Home", "Bramble and Co"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the customer line of {company}, an online shop, and a customer who phones in",
        "scenarios": (
            "the person wants to return shoes that do not fit",
            "the person received a broken lamp",
            "the person's refund has not arrived after two weeks",
            "the person got a sofa in the wrong colour",
            "the person wants to return a gift without a receipt",
            "the person's parcel shows as delivered but never came",
            "the person wants to swap a jacket for a bigger size",
            "the shop says a returned item never arrived",
        ),
        "setbacks": (
            "the return window closed two days ago",
            "returns cost a fee",
            "without a receipt it can only be store credit",
            "the bigger size is out of stock until next month",
        ),
        "good_news": (
            "the refund went through today",
            "the person can keep the broken lamp and get a new one free",
            "a courier will collect it from home for free",
            "the swap is being sent out today",
        ),
    },
    "hotel_booking": {
        "companies": ("Bayview Hotel", "The Linden Inn"),
        "first": "agent",
        "medium": "phone call",
        "intro": "{agent}, the AI voice agent on the reservations line of {company}, a hotel, and a guest who phones in",
        "scenarios": (
            "the person wants two nights next weekend",
            "the person needs to change their dates",
            "the person wants an early check-in",
            "the person left a jacket in their room",
            "the person was charged for a booking they cancelled",
            "the person asks about breakfast and parking",
            "the person needs a room for a family of five",
            "the person is phoning from their room because the heating does not work",
        ),
        "setbacks": (
            "the hotel is fully booked those nights",
            "the cancellation came too late for a refund",
            "early check-in costs extra",
            "only a small room is left",
        ),
        "good_news": (
            "the jacket was found and will be posted",
            "the charge has been refunded",
            "the person gets a free upgrade to a sea-view room",
            "early check-in is free",
        ),
    },
    "companion_chat": {
        "companies": ("Haven", "Juniper"),
        "first": "either",
        "medium": "voice conversation",
        "intro": "{agent}, an AI companion in the {company} app that the person talks to out loud at home, and the person",
        "scenarios": (
            "the person tells {agent} about a stressful day at work",
            "the person wants help planning the weekend",
            "the person is nervous about a job interview tomorrow",
            "the person had a falling-out with a friend",
            "the person is bored and wants to chat",
            "the person just got good news and wants to share it",
            "the person cannot sleep",
            "the person wants a dinner idea from what is in the fridge",
        ),
        "setbacks": (
            "the forecast says rain all weekend",
            "the person has an early meeting tomorrow",
            "the concert tickets the person wanted have sold out",
            "the gym closes in ten minutes",
        ),
        "good_news": (
            "Monday is a holiday",
            "the parcel the person was waiting for arrives tomorrow",
            "there is a recipe that uses exactly what they have",
            "a new season of the person's favourite show is out",
        ),
        "agent_moves": (
            "At some point {agent} asks the person a question about how they feel.",
            "At some point {agent} admits it does not know something.",
            "At some point {agent} gently suggests something.",
            "At some point {agent} says it cannot do something, because it is only an app.",
        ),
    },
    "sales_call": {
        "companies": ("Brightpath Direct", "Clearline Offers"),
        "first": "user",
        "medium": "phone call",
        "intro": "{agent}, an AI voice agent phoning people for {company}, a telesales firm, and the person who picks up",
        "scenarios": (
            "{agent} offers solar panels with nothing to pay upfront",
            "{agent} offers a cheaper mobile phone plan",
            "{agent} offers a home security system",
            "{agent} offers a free home energy survey",
            "{agent} follows up on a free trial the person signed up for",
            "{agent} offers a faster broadband deal",
            "{agent} offers a discounted gym membership",
            "{agent} offers a car insurance quote",
        ),
        "setbacks": (
            "the deal needs a two-year contract",
            "this is the third call this week",
            "{agent} asks for their date of birth and address",
            "the price goes up after six months",
        ),
        "good_news": (
            "there is no contract and they can cancel any time",
            "it is cheaper than what they pay now",
            "{agent} will take them off the call list right away",
            "the first three months are free",
        ),
    },
}

AGENT_MOVES = (
    "At some point {agent} cannot share something or do something for security or policy reasons, and says so.",
    "At some point {agent} does not have some information and says so, offering to find out or pointing the person elsewhere.",
    "At some point {agent} gives an estimate it is not sure of.",
    "{agent} asks the person to confirm a detail or two.",
    "{agent} explains what happens next.",
    "{agent} apologises for the trouble once, briefly.",
)

PERSONAS = (
    "a retired man in his seventies, polite and not confident with technology",
    "a busy parent with children in the background, distracted",
    "a university student, casual, says like a lot",
    "a small-business owner, direct and short on time",
    "a nurse just off a night shift, tired",
    "someone who speaks English as a second language, with simple words and the odd grammar slip",
    "a chatty woman in her fifties who adds little stories",
    "a precise, formal man who spells things out",
    "a young professional who talks fast",
    "someone who has already phoned about this twice",
    "a shy person who apologises a lot",
    "a blunt, plain-spoken tradesman",
    "a grandmother sorting this out for her grandson",
    "a teenager, a bit mumbly",
)

# How the person's mood runs; `fits` lists the seeds it goes with (sarcasm comes from annoyance or jokes, sincere thanks
# from relief or calm).
ARCS: dict[str, dict] = {
    "calm": {"weight": 2.0, "text": "calm and matter-of-fact throughout", "fits": ("lookalike", "none")},
    "frustrated": {
        "weight": 2.0,
        "text": "annoyed from the start and more frustrated as it drags on, grudgingly satisfied at the end",
        "fits": ("sarcasm", "none"),
    },
    "angry": {
        "weight": 1.2,
        "text": "angry from the start because this has happened before; blunt, with the odd mild swear word",
        "fits": ("sarcasm", "none"),
    },
    "worried": {
        "weight": 2.0,
        "text": "worried and anxious at first, relieved and grateful once it is sorted",
        "fits": ("lookalike", "none"),
    },
    "confused": {
        "weight": 1.5,
        "text": "confused by the process and needs things explained, thankful at the end",
        "fits": ("lookalike", "none"),
    },
    "jokey": {
        "weight": 1.2,
        "text": "cheerful and jokey, makes a joke or two and laughs at themself",
        "fits": ("sarcasm", "lookalike", "none"),
    },
    "hurried": {
        "weight": 1.0,
        "text": "in a hurry, with short answers and little patience",
        "fits": ("sarcasm", "none"),
    },
    "resigned": {
        "weight": 1.0,
        "text": "tired and resigned, expecting the worst, then pleasantly surprised",
        "fits": ("sarcasm", "lookalike", "none"),
    },
    "disappointed": {
        "weight": 0.8,
        "text": "disappointed and a little sad about what went wrong, a bit cheered by the end",
        "fits": ("lookalike", "none"),
    },
    "guarded": {
        "weight": 0.8,
        "text": "suspicious of the company and guarded about personal details",
        "fits": ("sarcasm", "none"),
    },
    "delighted": {"weight": 0.6, "text": "in a great mood, warm and friendly", "fits": ("lookalike", "none")},
}

# The sarcasm seed: what the agent says before it, what the person's answer does, and example answers (three shown).
SARCASM_KINDS: dict[str, dict] = {
    "after_setback": {
        "weight": 3.0,
        "agent": "tells the person something they do not want to hear: {setback}",
        "does": "words that sound pleased or grateful but plainly mean the opposite",
        "examples": (
            "oh great, love that",
            "perfect, just what I needed today",
            "wonderful, thanks so much",
            "brilliant, can't wait",
            "fantastic, really",
            "oh good, my favourite thing",
            "lovely, that's just lovely",
            "amazing, thanks for that",
            "oh, that's just great",
            "super, love waiting",
            "well isn't that nice",
            "cool, love it",
        ),
    },
    "mock_agreement": {
        "weight": 1.5,
        "agent": "makes a promise or a claim the person plainly does not believe",
        "does": "agreement that plainly means the opposite",
        "examples": (
            "sure they will",
            "right, like last time",
            "oh absolutely, I believe you",
            "yeah, I'm sure it will",
            "of course it is",
            "uh huh, sure",
            "oh yeah, definitely",
            "right, because that always works",
            "can't wait to see that happen",
            "sure, and I'm the king of England",
        ),
    },
    "sarcastic_thanks": {
        "weight": 1.5,
        "agent": "gives useless or obvious advice, or repeats something that does not help",
        "does": "thanks that plainly mean the opposite",
        "examples": (
            "wow, thanks, never thought of that",
            "so helpful, really",
            "great tip, thanks",
            "oh, that's very useful",
            "thanks, that clears it all up",
            "what would I do without you",
            "oh thanks, that's a big help",
            "gee, thanks",
            "wow, groundbreaking advice",
            "thanks a bunch for that",
        ),
    },
    "mock_impressed": {
        "weight": 1.0,
        "agent": "states the obvious, or reads back the problem the person already described",
        "does": "pretending to be impressed or surprised",
        "examples": (
            "you don't say",
            "what gave it away",
            "wow, really, no kidding",
            "well spotted",
            "no way, really",
            "genius",
            "oh, is that so",
            "amazing detective work",
        ),
    },
    "situational": {
        "weight": 1.5,
        "agent": "mentions some bad luck that is nobody's fault, least of all {agent}'s (bad timing, the weather, a closure, a queue)",
        "does": "a remark that sounds pleased but plainly complains about the situation, not about {agent}",
        "examples": (
            "lovely, just my luck",
            "well, this day keeps getting better",
            "oh, perfect timing as always",
            "great, of course it is",
            "fantastic, what a week",
            "just brilliant",
            "oh good, exactly what this week needed",
            "brilliant, and it's raining too",
            "love that for me",
        ),
    },
    "self": {
        "weight": 0.7,
        "agent": "kindly points out a mistake the person made",
        "does": "mock praise of themself",
        "examples": (
            "nice one, me",
            "genius move on my part",
            "great, I'm a natural at this",
            "well done me",
            "smooth, really smooth",
            "top marks to me",
            "classic me, brilliant",
        ),
    },
}

# The look-alike seed: the same kinds of words, meant.
LOOKALIKE_KINDS: dict[str, dict] = {
    "after_good_news": {
        "weight": 3.0,
        "agent": "gives good news: {good_news}",
        "does": "with words like great, perfect, love that, brilliant, thanks a lot or just what I needed",
        "examples": (
            "oh great, love that",
            "perfect, that's just what I needed",
            "brilliant, thanks a lot",
            "wonderful, thank you so much",
            "oh fantastic",
            "amazing, honestly",
            "lovely, that's lovely",
            "great, can't wait",
            "oh, that's just great",
            "super, thanks",
            "well isn't that nice",
            "love that, thank you",
        ),
    },
    "sincere_agreement": {
        "weight": 1.5,
        "agent": "suggests a plan or makes a promise that sounds good",
        "does": "agreeing with words like sure, of course, right, absolutely or yeah",
        "examples": (
            "sure, that works",
            "of course, go ahead",
            "right, yes, let's do that",
            "absolutely",
            "yeah, I'm sure that'll be fine",
            "uh huh, sure",
            "oh yeah, definitely",
            "right, that makes sense",
        ),
    },
    "sincere_thanks": {
        "weight": 1.5,
        "agent": "gives a tip that really helps, or sorts something out",
        "does": "thanking {agent} with words like so helpful, really, thanks or what would I do without you",
        "examples": (
            "wow, thanks, I never thought of that",
            "so helpful, really",
            "great tip, thanks",
            "oh, that's very useful",
            "thanks, that clears it all up",
            "what would I do without you",
            "oh thanks, that's a big help",
            "thanks a bunch for that",
            "brilliant advice, thank you",
        ),
    },
    "sincere_impressed": {
        "weight": 1.0,
        "agent": "solves the problem quickly or explains something clever",
        "does": "being impressed, with words like wow, no way, really or well spotted",
        "examples": (
            "wow, that was quick",
            "no way, really",
            "well spotted",
            "you don't say, I had no idea",
            "genius, thank you",
            "no kidding, it worked",
            "oh, is that so, good to know",
            "amazing, how did you find that",
        ),
    },
}

# A move the person makes somewhere in the conversation; in conversations without a seed the marked turn is one of these.
USER_MOVES: dict[str, dict] = {
    "hedges": {
        "weight": 1.5,
        "agent": "asks for a detail the person is not sure of",
        "does": "gives a detail they are unsure of or only know second-hand",
        "examples": (
            "I think my husband set it up",
            "it might have been Tuesday, not sure",
            "my daughter said it was paid",
            "I'm pretty sure it was the blue one",
            "I want to say around three",
            "someone at the shop told me it was covered",
            "probably last month, maybe earlier",
            "I believe it's under my wife's name",
        ),
    },
    "doesnt_know": {
        "weight": 1.5,
        "agent": "asks for a detail the person does not have",
        "does": "does not know something {agent} asks and says so",
        "examples": (
            "I don't know",
            "no idea, sorry",
            "I'd have to check",
            "I honestly couldn't tell you",
            "I never got one",
            "not a clue",
            "I don't remember",
            "you'd have to ask my son",
        ),
    },
    "withholds": {
        "weight": 1.5,
        "agent": "asks for a personal detail",
        "does": "declines to give a detail or dodges a question",
        "examples": (
            "I'd rather not give that out",
            "can we skip that",
            "why do you need that",
            "I'm not comfortable sharing that",
            "I'll keep that to myself",
            "let's not get into that",
            "I'd prefer not to say",
            "do I have to answer that",
        ),
    },
    "asks": {
        "weight": 1.0,
        "agent": "explains something or gives an update",
        "does": "asks {agent} a question of their own",
        "examples": (
            "how long will that take",
            "is there a charge for that",
            "what happens next",
            "why did that happen",
            "can I do this online instead",
            "will I get an email",
        ),
    },
    "requests": {
        "weight": 1.0,
        "agent": "says what it can do",
        "does": "asks {agent} to do something",
        "examples": (
            "can you cancel it",
            "could you send it again",
            "put me through to a person, please",
            "can you note that down",
            "call me back later",
            "can you text me the details",
        ),
    },
    "answers": {
        "weight": 0.6,
        "agent": "asks a question",
        "does": "answers one of {agent}'s questions plainly",
        "examples": (
            "yes, that's right",
            "it's the fourteenth of March",
            "no, just me",
            "the one ending four two",
            "Saturday works",
            "yeah, same address",
        ),
    },
    "closes": {
        "weight": 1.0,
        "agent": "asks if there is anything else",
        "does": "winds the conversation up once it is sorted",
        "examples": (
            "that's all, thanks",
            "no, that's everything",
            "okay, bye",
            "great, cheers, bye",
            "that's it for now",
            "lovely, thank you, bye",
        ),
    },
}

STYLE_ASR = (
    "The person's turns are speech-to-text output: all lower case and no punctuation, but apostrophes stay in words like "
    "i'm, don't and that's; with fillers like um, uh, like and you know, false starts, repeated words and now and then a "
    "misheard word."
)
STYLE_CLEAN = (
    "The person's turns are a clean transcript with normal capitals and punctuation, but they still sound like someone "
    "talking: contractions, short replies, now and then an um or a false start."
)


def _weighted(rng: random.Random, table: dict[str, dict], allowed=None) -> str:
    names = sorted(n for n in table if allowed is None or n in allowed)
    return rng.choices(names, weights=[table[n]["weight"] for n in names])[0]


def _examples(rng: random.Random, pool: tuple[str, ...], k: int = 3) -> str:
    return "; ".join(rng.sample(pool, min(k, len(pool))))


def build_prompts(n_per_domain: int, seed: int = SEED, prefix: str = PROMPT_PREFIX) -> list[dict]:
    """`n_per_domain` conversation prompts for each domain, each fixing the agent's name and company, what happens, who
    the person is and how their mood runs, and the moment: what the agent's line does and what the person's answer
    straight after it does, sarcastic (sarcasm seed), sincere in the same kinds of words (look-alike seed) or a move the
    set needs more of (no seed). Seeds are interleaved in prompt order and steer the writing only: the committee decides
    every label. A third of each domain's conversations, drawn at random, have the person's side as speech-to-text
    output."""
    rng = random.Random(seed)
    out = []
    for domain in sorted(DOMAINS):
        spec = DOMAINS[domain]
        n_asr = round(n_per_domain * ASR_SHARE)
        asr_flags = [True] * n_asr + [False] * (n_per_domain - n_asr)
        rng.shuffle(asr_flags)
        scen = list(range(len(spec["scenarios"])))
        rng.shuffle(scen)
        for j in range(n_per_domain):
            seed_kind = SEEDS[j % len(SEEDS)]
            agent = rng.choice(AGENT_NAMES)
            company = rng.choice(spec["companies"])
            persona = rng.choice(PERSONAS)
            arc = _weighted(rng, ARCS, [a for a, v in ARCS.items() if seed_kind in v["fits"]])
            first = spec["first"] if spec["first"] != "either" else rng.choice(("agent", "user"))
            agent_move = rng.choice(spec.get("agent_moves", AGENT_MOVES))
            focus = _weighted(rng, USER_MOVES) if seed_kind == "none" else None
            extra = _weighted(rng, USER_MOVES, [m for m in USER_MOVES if m != focus])
            p = {
                "prompt_id": f"{prefix}-{domain}-{j:03d}",
                "domain": domain,
                "agent": agent,
                "company": company,
                "scenario": spec["scenarios"][scen[j % len(scen)]].format(agent=agent),
                "persona": persona,
                "arc": arc,
                "first": first,
                "agent_move": agent_move.format(agent=agent),
                "seed": seed_kind,
                "seed_kind": None,
                "move_focus": focus,
                "move_extra": extra,
                "asr": asr_flags[j],
                "format": "json",
            }
            fresh = "Write a fresh line that fits this moment rather than copying those."
            if seed_kind == "sarcasm":
                p["seed_kind"] = _weighted(rng, SARCASM_KINDS)
                k = SARCASM_KINDS[p["seed_kind"]]
                p["moment_agent"] = k["agent"].format(
                    agent=agent, setback=rng.choice(spec["setbacks"]).format(agent=agent)
                )
                p["moment_user"] = (
                    f"answers it with sarcasm: {k['does'].format(agent=agent)} (the kind of thing people say: "
                    f"{_examples(rng, k['examples'])}). {fresh} Nobody mentions the sarcasm."
                )
            elif seed_kind == "lookalike":
                p["seed_kind"] = _weighted(rng, LOOKALIKE_KINDS)
                k = LOOKALIKE_KINDS[p["seed_kind"]]
                p["moment_agent"] = k["agent"].format(
                    agent=agent, good_news=rng.choice(spec["good_news"]).format(agent=agent)
                )
                p["moment_user"] = (
                    f"answers it sincerely and means it, {k['does'].format(agent=agent)} (the kind of thing people "
                    f"say: {_examples(rng, k['examples'])}). {fresh}"
                )
            else:
                k = USER_MOVES[focus]
                p["moment_agent"] = k["agent"].format(agent=agent)
                p["moment_user"] = (
                    f"{k['does'].format(agent=agent)} (the kind of thing people say: {_examples(rng, k['examples'])}). {fresh}"
                )
            p["messages"] = messages(p)
            out.append(p)
    return out


def messages(p: dict) -> list[dict]:
    """The prompt: the conversation as JSON in rounds (the agent, then the person straight after), with the moment as its
    own round, so the person's turns always answer the agent's line before them and exactly one turn is the moment."""
    spec = DOMAINS[p["domain"]]
    agent = p["agent"]
    xm = USER_MOVES[p["move_extra"]]
    if p["first"] == "user":
        opening = (
            '"opening" is what the person says as they pick up, before '
            if spec["first"] == "user"
            else '"opening" is the person\'s first words, before '
        ) + f"{agent} speaks. "
    else:
        opening = f"The first round opens with {agent}'s first words. "
    task = (
        f"Write one complete {spec['medium']} between {spec['intro'].format(agent=agent, company=p['company'])}.\n\n"
        f"What happens: {p['scenario']}. {p['agent_move']}\n"
        f"The person: {p['persona']}. Their mood: {ARCS[p['arc']]['text']}. Let it show in what they say and how; people seldom "
        "name their feelings.\n"
        f"Somewhere in the conversation the person also {xm['does'].format(agent=agent)}.\n"
        f"One round in the middle is the moment: {agent} {p['moment_agent']}, and the person {p['moment_user']}\n\n"
        "How they speak:\n"
        f"- {agent} sounds like a good voice assistant: warm, clear and brief, with contractions and normal capitals and punctuation. "
        "One or two short sentences a turn, at most 25 words.\n"
        f"- {STYLE_ASR if p['asr'] else STYLE_CLEAN}\n"
        "- The person's turns run from one word to about 25 words: short replies, half sentences, questions, now and then a "
        "longer explanation.\n\n"
        f'Write it as JSON. {opening}Each round is {{"agent": what {agent} says, "user": what the person says straight '
        'after}. "before" holds the 3 to 5 rounds before the moment, "moment" is that one round, "after" holds the 2 to 4 '
        f'rounds after it, and "agent_last" is {agent}\'s last words, or "" if the person has the last word. Each string '
        "holds only spoken words: no stage directions, no descriptions of tone or feelings, no sound effects, no emoji."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


ROUND = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "user"],
    "properties": {"agent": {"type": "string"}, "user": {"type": "string"}},
}


def schema(first: str) -> dict:
    """The JSON the teacher fills (vLLM structured output): rounds before the moment, the moment, rounds after, the agent's
    last words; `opening` first when the person speaks first."""
    props = {
        "before": {"type": "array", "items": ROUND},
        "moment": ROUND,
        "after": {"type": "array", "items": ROUND},
        "agent_last": {"type": "string"},
    }
    if first == "user":
        props = {"opening": {"type": "string"}, **props}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def parse_json_conversation(text: str, first: str) -> tuple[list[dict], list[str]]:
    """The turns of a JSON conversation (schema()), [{"who": "agent"|"user", "text", "marked"}], the moment's answer
    marked, and the problems met. Empty strings are skipped and a speaker twice in a row joined into one turn; a mark
    joined after the person's own turn does not answer the agent's line before it, so it is dropped."""
    try:
        obj = json.loads(text)
    except ValueError:
        return [], ["not json"]
    seq: list[tuple[str, str, bool]] = []
    if first == "user":
        seq.append(("user", obj.get("opening") or "", False))
    for r in obj.get("before") or []:
        seq += [("agent", r.get("agent") or "", False), ("user", r.get("user") or "", False)]
    m = obj.get("moment") or {}
    seq += [("agent", m.get("agent") or "", False), ("user", m.get("user") or "", True)]
    for r in obj.get("after") or []:
        seq += [("agent", r.get("agent") or "", False), ("user", r.get("user") or "", False)]
    seq.append(("agent", obj.get("agent_last") or "", False))
    turns: list[dict] = []
    problems: list[str] = []
    for who, words, marked in seq:
        words = words.strip()
        if not words:
            if marked:
                problems.append("empty moment")
            continue
        if turns and turns[-1]["who"] == who:
            if marked and not turns[-1]["marked"]:
                problems.append("mark after a person's row")
            turns[-1]["text"] += " " + words
            problems.append("same speaker twice")
        else:
            turns.append({"who": who, "text": words, "marked": marked})
    return turns, problems


# --- reading the teacher's conversations ---------------------------------------------------------------------------

# Stage directions, tone notes and anything that names the seed are never kept: the line is dropped, not repaired.
LEAK = re.compile(r"\*|\(.*?\)|\[.*?\]|\bsarcas|\bironic", re.I)


def asr_text(t: str) -> str:
    """Speech-to-text output as most voice pipelines see it with punctuation off: lower case, no punctuation but
    apostrophes and hyphens inside words."""
    t = re.sub(r"[,.!?;:\"“”…]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def turn_text(turn: dict, asr: bool) -> str | None:
    """A turn's words as a corpus line's text, or None when they cannot be one (a stage direction, a tone note, the seed
    named, too long, markup)."""
    t = turn["text"].strip().strip('"“”').strip()
    if not t or LEAK.search(t):
        return None
    if asr and turn["who"] == "user":
        t = asr_text(t)
    return clean_text(t)


def candidates(conv: dict) -> list[dict]:
    """Every turn that can be a test line: its text, and the other party's turn before it (None for the first turn)."""
    out = []
    turns = conv["turns"]
    for i, t in enumerate(turns):
        text = turn_text(t, conv["asr"])
        if text is None:
            continue
        prev = None
        if i > 0:
            prev = turn_text(turns[i - 1], conv["asr"])
            if prev is None or turns[i - 1]["who"] == t["who"]:
                continue
        out.append({"i": i, "who": t["who"], "text": text, "previous": prev, "marked": t["marked"]})
    return out


def usable(conv: dict) -> bool:
    """At least 8 turns, alternating, with two person lines and one agent line that can be taken."""
    c = candidates(conv)
    return (
        len(conv["turns"]) >= 8
        and sum(x["who"] == "user" for x in c) >= 2
        and any(x["who"] == "agent" for x in c)
    )


def marked_ok(conv: dict) -> bool:
    """Exactly one marked turn, the person's, and it can be taken."""
    marks = [t for t in conv["turns"] if t["marked"]]
    return len(marks) == 1 and any(x["marked"] and x["who"] == "user" for x in candidates(conv))


def corpus_line(conv: dict, cand: dict, generator: str, extra_slices: tuple[str, ...] = ()) -> CorpusLine:
    """A test line in Tone's generated-line conventions: the person speaks as `player` to the agent, the agent as its
    name to `player`, each with the other's turn before it as `previous`."""
    agent = conv["agent"]
    if cand["who"] == "user":
        speaker, kind, to, targets, before = PLAYER, "player", agent, [agent], agent
    else:
        speaker, kind, to, targets, before = agent, "npc", PLAYER, [PLAYER], PLAYER
    previous = f"{before}: {cand['previous']}"[:MAX_PREVIOUS] if cand["previous"] else None
    inp = LineInput(
        speaker=speaker, speaker_kind=kind, to=to, targets=targets, text=cand["text"], previous=previous
    )
    slices = [
        "voice_agent",
        conv["domain"],
        kind,
        "user" if cand["who"] == "user" else "agent",
        *extra_slices,
    ]
    if cand["who"] == "user" and conv["asr"]:
        slices.append("asr_style")
    if previous is None:
        slices.append("opener")
    source_ref = f"{conv['prompt_id']}#{cand['i']}"
    return CorpusLine(
        line_id=f"{SOURCE[:2]}-{short_id(SOURCE, source_ref)}",
        source=SOURCE,
        source_ref=source_ref,
        licence=LICENCE,
        split="test",
        group_key=f"gen:{conv['prompt_id']}",
        input=inp,
        slices=sorted(set(slices)),
        generator=generator,
    )


def pick_lines(
    conv: dict, rng: random.Random, seen_pairs: set, per_text: Counter
) -> list[tuple[CorpusLine, dict]]:
    """Two of the person's lines (the marked one first when it can be taken) and one of the agent's, each new to the set:
    a (text, line before) pair once, a text at most MAX_PER_TEXT times. Returns (line, its metadata) pairs; fewer than
    three when the conversation runs out of new lines."""
    cands = candidates(conv)

    def fresh(c: dict) -> bool:
        k = norm_key(c["text"])
        return bool(k) and (k, norm_key(c["previous"] or "")) not in seen_pairs and per_text[k] < MAX_PER_TEXT

    def take(c: dict) -> None:
        k = norm_key(c["text"])
        seen_pairs.add((k, norm_key(c["previous"] or "")))
        per_text[k] += 1

    out: list[tuple[CorpusLine, dict]] = []
    users = [c for c in cands if c["who"] == "user"]
    mark = next((c for c in users if c["marked"]), None) if marked_ok(conv) else None
    chosen: list[dict] = []
    if mark is not None and fresh(mark):
        chosen.append(mark)
        take(mark)
    others = [c for c in users if c is not mark]
    rng.shuffle(others)
    for c in others:
        if len(chosen) >= 2:
            break
        if fresh(c):
            chosen.append(c)
            take(c)
    agents = [c for c in cands if c["who"] == "agent"]
    opener = next((c for c in agents if c["i"] == 0), None)
    after = next((c for c in agents if mark is not None and c["i"] == mark["i"] + 1), None)
    rest = [c for c in agents if c["i"] > 0]
    rng.shuffle(rest)
    r = rng.random()
    first = (
        opener
        if (opener is not None and r < OPENER_SHARE)
        else (after if (after is not None and OPENER_SHARE <= r < OPENER_SHARE + AFTER_MARK_SHARE) else None)
    )
    agent_pick = next((c for c in ([first] if first else []) + rest if fresh(c)), None)
    if agent_pick is not None:
        take(agent_pick)
    for c in chosen:
        if c is mark and conv["seed"] in ("sarcasm", "lookalike"):
            gen, extra = f"{conv['seed']}_seed:{conv['seed_kind']}", (f"{conv['seed']}_seed",)
        elif c is mark:
            gen, extra = f"move_focus:{conv['move_focus']}", ()
        else:
            gen, extra = "user_turn", ()
        out.append((corpus_line(conv, c, gen, extra), _meta(conv, c, gen)))
    if agent_pick is not None:
        gen = (
            "agent_turn:opener"
            if agent_pick["i"] == 0
            else ("agent_turn:after_mark" if agent_pick is after else "agent_turn")
        )
        out.append((corpus_line(conv, agent_pick, gen), _meta(conv, agent_pick, gen)))
    return out


def _meta(conv: dict, cand: dict, gen: str) -> dict:
    keys = (
        "prompt_id",
        "domain",
        "agent",
        "company",
        "scenario",
        "persona",
        "arc",
        "seed",
        "seed_kind",
        "move_focus",
        "move_extra",
        "asr",
    )
    return {
        **{k: conv[k] for k in keys},
        "turn": cand["i"],
        "turns": len(conv["turns"]),
        "who": cand["who"],
        "marked": cand["marked"],
        "generator": gen,
    }


def select(convs: list[dict], per_domain: int, seed: int = SEED) -> tuple[list[CorpusLine], list[dict], dict]:
    """Up to `per_domain` usable conversations per domain, three lines from each: first those with a usable mark, up to
    QUOTA per seed in prompt order, then any other with a usable mark, then the rest. Returns the lines, their
    metadata (by line_id) and counts."""
    seen_pairs: set = set()
    per_text: Counter = Counter()
    lines, meta = [], []
    counts: Counter = Counter()
    by_domain: dict[str, list[dict]] = {}
    for c in convs:
        by_domain.setdefault(c["domain"], []).append(c)
    for domain in sorted(by_domain):
        cs = by_domain[domain]
        ok = sorted((c for c in cs if usable(c)), key=lambda c: c["prompt_id"])
        counts["unusable"] += len(cs) - len(ok)
        marked = [c for c in ok if marked_ok(c)]
        first = [c for sd, q in QUOTA.items() for c in [x for x in marked if x["seed"] == sd][:q]]
        order = first + [c for c in marked if c not in first] + [c for c in ok if not marked_ok(c)]
        taken = 0
        for c in order:
            if taken >= per_domain:
                break
            picked = pick_lines(c, random.Random(f"{seed}:{c['prompt_id']}"), seen_pairs, per_text)
            if not picked:
                continue
            taken += 1
            counts["conversations"] += 1
            counts["short_of_three"] += len(picked) < 3
            for ln, m in picked:
                lines.append(ln)
                meta.append({"line_id": ln.line_id, **m})
    ids = [ln.line_id for ln in lines]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate line ids")
    return lines, meta, dict(counts)
