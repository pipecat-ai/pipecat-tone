"""r3's support-call set: calls between an AI voice agent and a customer that a teacher model writes from the prompts
built here. Each call has a moment, one line of the agent's answered twice, once as the call goes and once instead:
one answer sarcastic and one sincere, so the same line comes before both. It is written for rc1's misread, a turn that
is sarcastic on its own words read sincere after a plain line, with the answers to news that rc1 reads well beside it.
Its lines are split by call, with three domains held out whole for test, and each keeps the turns before it, for a
reader that reads more of the call. Pure functions, run on Modal by ``lab.reader.pipeline.support_generate``."""

import hashlib
import json
import random
import re
from collections import Counter

from lab.reader import voice
from lab.reader.schemas import MAX_PREVIOUS, PLAYER, LineInput
from lab.reader.voice import CorpusLine, norm_key, short_id, turn_text

SOURCE = voice.SOURCE
LICENCE = voice.LICENCE
PROMPT_PREFIX = "ts1"
SEED = 601
ASR_SHARE = 1 / 3  # calls whose person side reads as speech-to-text output
ORDINARY_SHARE = 0.5  # calls that also give one of the person's other turns
HELD_DOMAINS = ("rail_travel", "vet_clinic", "moving_company")  # test only, every call
TEST_SHARE, DEV_SHARE = 0.02, 0.05  # of the other domains' calls, by call
SHORT_WORDS = 4  # a text this short may come up to MAX_SHORT times, a longer one MAX_LONG times
MAX_SHORT, MAX_LONG = 6, 2
BAN_WORDS = 4  # an evaluation line's text this long or longer is kept out of training; its (text, line before), always
# What tells an agent's plain line it is not plain: it was meant to carry no news.
NEWS_WORDS = re.compile(r"\b(unfortunately|sorry|afraid|apologi[sz]e|bad news|good news|great news)\b", re.I)

# Each domain: `intro` names the two parties (after "between"), `scenarios` what the call is about. Companies are made
# up.
DOMAINS: dict[str, dict] = {
    "online_shop": {
        "companies": ("Tidewater Supply", "Kestrel and Finch"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, an online shop, and a customer who phones in",
        "scenarios": (
            "the person's order has not shipped a week after they paid",
            "the tracking on the person's parcel has not moved in four days",
            "the jacket the person ordered came in the wrong size",
            "the person was charged but never got an order confirmation",
            "the person wants to change the delivery address on an order",
            "a discount code did not apply at checkout",
            "the person's parcel shows as delivered but nothing came",
            "the person's order was split into three parcels and one is missing",
        ),
    },
    "parcel_courier": {
        "companies": ("Swiftline Parcels", "Copperpost Couriers"),
        "intro": "{agent}, the AI voice agent on the delivery line of {company}, a parcel courier, and a person who phones in",
        "scenarios": (
            "the person missed a delivery and found a card through the door",
            "the person wants a parcel delivered again on Saturday",
            "the person's parcel has been at a depot for a week",
            "the driver left the person's parcel out in the rain",
            "a parcel needs a signature but the person is at work all day",
            "the person's parcel was left with a neighbour they do not know",
            "the person has to pay a customs fee before a parcel is delivered",
            "the person wants to pick a parcel up from the depot themself",
        ),
    },
    "grocery_delivery": {
        "companies": ("Greenbasket", "Harvest Lane Grocery"),
        "intro": "{agent}, the AI voice agent on the help line of {company}, a grocery delivery service, and a customer who phones in",
        "scenarios": (
            "half the items in the person's delivery were missing",
            "the person got substitutions they did not want",
            "the person's delivery slot came and went with no van",
            "the frozen food in the person's order arrived warm",
            "the person was charged for items that never came",
            "the person wants to move tomorrow's delivery slot",
            "every egg in the person's order was broken",
            "the person cannot add items to an order already placed",
        ),
    },
    "meal_delivery": {
        "companies": ("Dashbite", "Forkful"),
        "intro": "{agent}, the AI voice agent on the support line of {company}, a food delivery app, and a customer who phones in",
        "scenarios": (
            "the person's dinner order is an hour late",
            "the person got someone else's order",
            "the person's food arrived cold",
            "the driver cannot find the person's building",
            "a drink was missing from the person's order",
            "the person was charged twice for one order",
            "the restaurant cancelled the person's order after they paid",
            "a promotion did not apply to the person's order",
        ),
    },
    "furniture_delivery": {
        "companies": ("Oakhaven Home", "Linden and Moss Furniture"),
        "intro": "{agent}, the AI voice agent on the delivery line of {company}, a furniture shop, and a customer who phones in",
        "scenarios": (
            "the person wants to know when their sofa will be delivered",
            "the person's new wardrobe arrived with parts missing",
            "the person's dining table arrived scratched",
            "the person wants to book someone to put their bed together",
            "the delivery date for the person's order was moved again",
            "the person got the right chair in the wrong colour",
            "the person wants to cancel an order that has not been delivered",
            "the person's sofa would not fit up their stairs",
        ),
    },
    "appliance_repair": {
        "companies": ("FixRight Appliance Care", "Brightline Home Services"),
        "intro": "{agent}, the AI voice agent on the booking line of {company}, an appliance repair service, and a customer who phones in",
        "scenarios": (
            "the person's washing machine leaks every time it runs",
            "the person's fridge has stopped getting cold",
            "the technician never turned up for the person's booking",
            "the part the person's dishwasher needs is on back order",
            "the person needs to move a repair visit",
            "the person wants to know if the repair is under warranty",
            "the door of the person's oven will not close",
            "the person's dryer broke again a week after it was repaired",
        ),
    },
    "home_internet": {
        "companies": ("Lumen Fibre", "Crestnet Broadband"),
        "intro": "{agent}, the AI voice agent on the support line of {company}, a home internet provider, and a customer who phones in",
        "scenarios": (
            "the person's internet drops every evening",
            "the person's speeds are far below what they pay for",
            "every light on the person's router has gone red",
            "the person is waiting for a new connection to be put in",
            "the person is moving house and wants to take the service with them",
            "the person's bill went up without warning",
            "the internet in the person's area has been down all day",
            "the person cannot log in to their account online",
        ),
    },
    "mobile_carrier": {
        "companies": ("Veltro Mobile", "Bluepeak Wireless"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, a mobile network, and a customer who phones in",
        "scenarios": (
            "the person's phone has no signal at home",
            "the person was charged for roaming on a trip abroad",
            "moving the person's number from another network is stuck",
            "the person's bill is higher than their plan",
            "the person keeps running out of data",
            "the person's new SIM card has not been switched on",
            "the credit for the person's traded-in phone never showed up",
            "the person's contract is ending and they want to know their options",
        ),
    },
    "streaming_tv": {
        "companies": ("Streamhaven", "Novaview TV"),
        "intro": "{agent}, the AI voice agent on the help line of {company}, a streaming service, and a subscriber who phones in",
        "scenarios": (
            "the person cannot log in to their account",
            "the person was charged after they cancelled",
            "every show the person plays keeps buffering",
            "the price of the person's plan went up",
            "the person wants to move to a cheaper plan",
            "someone else has been using the person's account",
            "the person wants to know how many screens their plan allows",
            "the person's free trial turned into a paid plan",
        ),
    },
    "bank_account": {
        "companies": ("Harbourline Bank", "Westmere Savings"),
        "intro": "{agent}, the AI voice agent on the phone banking line of {company}, and a customer who phones in",
        "scenarios": (
            "a transfer the person made has not arrived",
            "the person's account has been locked",
            "the person was charged an overdraft fee",
            "the person's pay has not shown up in their account",
            "the person wants to set a new card PIN",
            "the person cannot log in to the banking app",
            "the person wants to add their partner to their account",
            "the person wants to close an old savings account",
        ),
    },
    "credit_card": {
        "companies": ("Corvid Card Services", "Summit Card"),
        "intro": "{agent}, the AI voice agent on the card services line of {company}, and a cardholder who phones in",
        "scenarios": (
            "the person's card was declined at a shop",
            "the person sees a charge they do not recognise",
            "the person wants a higher credit limit",
            "the person was charged a late fee",
            "the person's replacement card has not arrived",
            "the person is going abroad and wants the card to work there",
            "the person wants to dispute a purchase that never arrived",
            "the person's payment does not show on their account",
        ),
    },
    "loan_servicing": {
        "companies": ("Keystone Home Loans", "Bridgewater Lending"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, a mortgage and loan servicer, and a borrower who phones in",
        "scenarios": (
            "the person wants to move the date their payment comes out",
            "the person's monthly payment went up",
            "the person wants to know how much it would take to pay the loan off",
            "the person missed a payment by a few days",
            "an online payment the person made failed",
            "the person is thinking about refinancing",
            "the person never got their yearly statement",
            "the person has questions about the insurance on their loan",
        ),
    },
    "car_insurance": {
        "companies": ("Ironbridge Auto Insurance", "Pinecrest Mutual"),
        "intro": "{agent}, the AI voice agent on the claims and policy line of {company}, a car insurer, and a policyholder who phones in",
        "scenarios": (
            "the person wants to start a claim after a small crash",
            "the person wants to know if a rental car is covered while theirs is fixed",
            "the person's premium went up at renewal",
            "the person wants to add their son as a driver",
            "the garage fixing the person's car keeps pushing back the date",
            "the person has a chip in their windscreen",
            "the person wants to know where their claim has got to",
            "the person wants to cancel their policy",
        ),
    },
    "home_insurance": {
        "companies": ("Hearthstone Insurance", "Calloway Home Cover"),
        "intro": "{agent}, the AI voice agent on the claims line of {company}, a home insurer, and a policyholder who phones in",
        "scenarios": (
            "a pipe burst in the person's kitchen",
            "the person is waiting for an assessor to come and look at the damage",
            "a storm took tiles off the person's roof",
            "the person's claim was turned down",
            "the person wants to know when their claim will be paid",
            "the person's renewal price has gone up a lot",
            "the person's bike was stolen from their garden",
            "the person needs somewhere to stay while their home is repaired",
        ),
    },
    "airline": {
        "companies": ("Aerolume", "Skylark Airways"),
        "intro": "{agent}, the AI voice agent on the bookings line of {company}, an airline, and a passenger who phones in",
        "scenarios": (
            "the person's flight was cancelled the night before",
            "the person missed a connecting flight",
            "the person's bag did not arrive with them",
            "the person wants to change their seat",
            "the person wants a refund for a cancelled flight",
            "the person's flight is delayed by five hours",
            "the person wants to bring their dog on the plane",
            "the person's name is spelled wrong on the ticket",
        ),
    },
    "hotel": {
        "companies": ("Marigold Hotels", "The Ashby Inn"),
        "intro": "{agent}, the AI voice agent on the reservations line of {company}, a hotel, and a guest who phones in",
        "scenarios": (
            "the person's room was not ready when they arrived",
            "the hotel cannot find the person's booking",
            "the person's room is next to the lifts and very noisy",
            "the person was charged a deposit twice",
            "the person wants to check in early",
            "the person wants to cancel a booking",
            "the person wants to know about parking",
            "the person left a jacket in their room",
        ),
    },
    "car_rental": {
        "companies": ("Roadway Rentals", "Meadow Car Hire"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, a car rental firm, and a customer who phones in",
        "scenarios": (
            "the car the person booked was not there when they went to pick it up",
            "the person was charged for damage they say they did not do",
            "the person was charged for fuel although they filled the tank",
            "the person wants to keep the car two more days",
            "the person wants to return the car to a different branch",
            "the person's rental car broke down on the motorway",
            "the person's deposit has not been returned",
            "the person got a bill for tolls weeks later",
        ),
    },
    "rail_travel": {
        "companies": ("Northbound Rail", "Silverline Trains"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, a train company, and a passenger who phones in",
        "scenarios": (
            "the person's train was cancelled",
            "the person wants money back for a late train",
            "the person wants to renew their season ticket",
            "the person lost their ticket",
            "the person wants to reserve a seat",
            "the person wants to bring a bike on the train",
            "engineering works ruined the person's journey",
            "the person booked a ticket for the wrong day",
        ),
    },
    "rideshare": {
        "companies": ("Hopride", "Glide Cars"),
        "intro": "{agent}, the AI voice agent on the rider support line of {company}, a ride app, and a rider who phones in",
        "scenarios": (
            "the person was charged for a ride they never took",
            "the person's driver took a long way round",
            "the person left their phone in a car",
            "the person was charged a fee for cancelling",
            "the person was charged far more than the quote",
            "the person's driver was rude",
            "the person's account was locked",
            "a promo code did not work on the person's ride",
        ),
    },
    "event_tickets": {
        "companies": ("Stagepass", "Ticketgrove"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, a ticket seller, and a customer who phones in",
        "scenarios": (
            "the concert the person has tickets for was moved to another date",
            "the person's tickets do not show in the app",
            "the person wants a refund",
            "the person wants to give their tickets to a friend",
            "the person's seats are not together",
            "the event the person booked was cancelled",
            "the person wants to know why the fees were so high",
            "the person bought tickets for the wrong night",
        ),
    },
    "electricity": {
        "companies": ("Brightvolt Energy", "Coastal Power"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, an electricity supplier, and a customer who phones in",
        "scenarios": (
            "the power is out on the person's street",
            "the person's bill was estimated and looks far too high",
            "the person wants to give a meter reading",
            "the person needs a plan to pay off what they owe",
            "the person wants to switch to a cheaper tariff",
            "the person's smart meter has stopped sending readings",
            "the person's monthly payment went up",
            "the person is moving out and wants to close the account",
        ),
    },
    "water_utility": {
        "companies": ("Clearwater Utilities", "Riverside Water"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, a water company, and a customer who phones in",
        "scenarios": (
            "the water pressure at the person's house is very low",
            "water is leaking up through the road outside the person's house",
            "the person's water bill doubled",
            "the person has questions about their water meter",
            "the person heard there is a notice to boil water",
            "the person is moving in and wants to set up an account",
            "the person wants to pay a bill in parts",
            "the water from the person's taps is brown",
        ),
    },
    "clinic_appointments": {
        "companies": ("Larkspur Medical Group", "Fairview Clinic"),
        "intro": "{agent}, the AI voice agent that answers the phone at {company}, a doctors' practice, and a patient who phones in",
        "scenarios": (
            "the person wants an appointment this week",
            "the person needs to move an appointment",
            "the person is waiting for test results",
            "the person needs a referral to a specialist",
            "the person needs a repeat prescription",
            "the person wants to cancel an appointment",
            "the person wants the earliest appointment they can get",
            "the person never got the link for a video appointment",
        ),
    },
    "pharmacy": {
        "companies": ("Wellspring Pharmacy", "CareRx"),
        "intro": "{agent}, the AI voice agent on the phone line of {company}, a pharmacy, and a customer who phones in",
        "scenarios": (
            "the person's prescription refill is not ready",
            "the person's insurance would not cover a prescription",
            "the medicine the person needs is out of stock",
            "the person wants to move a prescription from another pharmacy",
            "the person wants their medicine delivered",
            "the label on the person's medicine looks wrong",
            "the person wants to know what a prescription will cost",
            "the person wants refills to be sent on their own",
        ),
    },
    "vet_clinic": {
        "companies": ("Paws and Claws Veterinary", "Hillside Animal Hospital"),
        "intro": "{agent}, the AI voice agent that answers the phone at {company}, a vet's practice, and a pet owner who phones in",
        "scenarios": (
            "the person wants a check-up for their dog",
            "the person's dog ate something it should not have",
            "the person needs their cat's vaccination records",
            "the person needs more of their dog's prescription food",
            "the person has a question about a bill",
            "the person's dog had surgery and needs a follow-up",
            "the person's cat has stopped eating",
            "the person needs to move their rabbit's appointment",
        ),
    },
    "gym_membership": {
        "companies": ("Peak Form Fitness", "Ironworks Gym"),
        "intro": "{agent}, the AI voice agent on the membership line of {company}, a gym chain, and a member who phones in",
        "scenarios": (
            "the person wants to cancel their membership",
            "the person was charged while their membership was frozen",
            "every class the person wants is full",
            "the person's locker was broken into",
            "the person wants to bring a friend for free",
            "the person wants to change the day they pay",
            "the person wants to move to another branch",
            "the person wants to book a personal trainer",
        ),
    },
    "software_support": {
        "companies": ("Taskloom", "Cloudnest"),
        "intro": "{agent}, the AI voice agent on the support line of {company}, a software company, and a customer who phones in",
        "scenarios": (
            "the person cannot log in to their account",
            "the person's files have stopped syncing",
            "the person was charged for the wrong plan",
            "a feature the person relies on has disappeared",
            "the person wants to export all their data",
            "the person lost the phone they use for two-step sign-in",
            "the person wants to add seats for their team",
            "the app crashes every time the person opens it",
        ),
    },
    "laptop_repair": {
        "companies": ("Corelight Computers", "Arcwell Tech"),
        "intro": "{agent}, the AI voice agent on the repairs line of {company}, a computer maker, and a customer who phones in",
        "scenarios": (
            "the person's laptop has been away for repair for three weeks",
            "the battery in the person's laptop has swollen",
            "the person's warranty claim was turned down",
            "the person's screen flickers",
            "the person wants to know if their files will be kept during the repair",
            "the person never got the box to send their laptop in",
            "the person wants an update on their repair",
            "the person was told they would get a replacement laptop",
        ),
    },
    "retail_returns": {
        "companies": ("Thornfield Department Store", "Marlow and Co"),
        "intro": "{agent}, the AI voice agent on the customer line of {company}, a department store, and a customer who phones in",
        "scenarios": (
            "the person wants to know if they can still return something",
            "the person's refund has not come through",
            "the person wants to swap shoes for another size",
            "the person was given store credit instead of a refund",
            "the person needs a returns label",
            "the person got a damaged lamp",
            "the person has a gift receipt and wants to return the gift",
            "the person was charged a restocking fee",
        ),
    },
    "property_management": {
        "companies": ("Elmstead Property Management", "Cornerstone Lettings"),
        "intro": "{agent}, the AI voice agent on the tenant line of {company}, a property manager, and a tenant who phones in",
        "scenarios": (
            "the boiler in the person's flat has broken",
            "water is coming through the person's ceiling from upstairs",
            "the person cannot pay rent through the online portal",
            "the person's lease is up for renewal",
            "the person wants to complain about noise from next door",
            "the person has mice in their kitchen",
            "the person is waiting for their deposit back",
            "the person's key fob has stopped working",
        ),
    },
    "moving_company": {
        "companies": ("Swiftmove Removals", "Hollis Moving and Storage"),
        "intro": "{agent}, the AI voice agent on the bookings line of {company}, a moving company, and a customer who phones in",
        "scenarios": (
            "the person wants to change their moving day",
            "the person wants a quote for a move",
            "the movers broke the person's mirror",
            "the moving truck is late on moving day",
            "the person wants to keep some things in storage",
            "the person wants the movers to pack for them",
            "the person wants to know what the insurance covers",
            "the person wants to cancel their booking",
        ),
    },
    "government_services": {
        "companies": ("the City Permits Office", "the County Licensing Bureau"),
        "intro": "{agent}, the AI voice agent on the public phone line of {company}, and a resident who phones in",
        "scenarios": (
            "the person needs to renew a parking permit",
            "the person wants an appointment to renew their licence",
            "the person wants to know what a permit costs",
            "the person does not know which documents to bring",
            "the person's application has been pending for weeks",
            "the person lost their permit card",
            "the online portal will not take the person's payment",
            "the person changed their name and needs their records updated",
        ),
    },
    "student_services": {
        "companies": ("Ridgemont University", "Lakeshore College"),
        "intro": "{agent}, the AI voice agent on the student services line of {company}, and a student who phones in",
        "scenarios": (
            "the person's financial aid has not come through",
            "there is a hold on the person's enrolment",
            "the person needs a copy of their transcript",
            "the person's tuition payment did not go through",
            "the person has a problem with their student housing",
            "the person cannot register for a course they need",
            "the person is waiting for a refund from the college",
            "the person lost their student ID card",
        ),
    },
    "car_service": {
        "companies": ("Fenwick Motors Service", "Granite Auto Care"),
        "intro": "{agent}, the AI voice agent on the service line of {company}, a car dealer's workshop, and a customer who phones in",
        "scenarios": (
            "the person wants to book their car in for a service",
            "the person got a recall letter about their car",
            "the repair on the person's car is taking longer than promised",
            "the person wants a courtesy car while theirs is in",
            "the person wants a quote for new brakes",
            "a warning light has come on in the person's car",
            "the person needs new tyres",
            "the person's bill came to more than the quote",
        ),
    },
    "home_warranty": {
        "companies": ("Shieldhome Warranty", "Bluestone Home Protect"),
        "intro": "{agent}, the AI voice agent on the claims line of {company}, a home warranty company, and a member who phones in",
        "scenarios": (
            "the person's air conditioning stopped working in a heatwave",
            "the person wants a contractor sent for a leaking water heater",
            "the contractor the company sent never came back to finish",
            "the person's claim for a broken dishwasher was refused",
            "the person wants to know what their plan covers",
            "the person was charged a service fee they did not expect",
            "the person's garage door opener has broken",
            "the person wants to renew their plan",
        ),
    },
}

PERSONAS = voice.PERSONAS + (
    "a man calling from his car on a hands-free set",
    "someone calling on behalf of their elderly father",
    "a new parent who has hardly slept",
    "an IT manager who knows exactly what they want",
    "a woman whose first language is Spanish, with careful English",
    "a retired teacher who is patient but firm",
    "a delivery driver calling between drops",
    "a man who has been on hold for twenty minutes already",
)

# The agent's line at the moment: what it does, a few lines like it, the answer the call goes on from (`main`) and the
# kinds the answer given instead is drawn from. Every moment pairs one sarcastic answer with one sincere one.
MOMENTS: dict[str, dict] = {
    "status": {
        "weight": 3.0,
        "agent": "gives a plain update on where things stand, which is neither good news nor bad news in itself, and "
        "may ask what the person would like to do",
        "plain": True,
        "examples": (
            "It's still being processed, so it hasn't shipped yet. What would you like to do?",
            "The case is with the team that handles these now.",
            "I can see the payment is still pending on our side.",
            "The technician is booked for the window you picked.",
            "Your parcel is at the local depot at the moment.",
            "The request went in this morning and it's in the queue.",
        ),
        "main": ("sarcastic_own",),
        "instead": ("sincere_pleased", "sincere_negative", "sincere_accepting"),
    },
    "question": {
        "weight": 1.5,
        "agent": "asks a routine question: to confirm a detail, what the person would like to do next, or whether "
        "there is anything else",
        "plain": True,
        "examples": (
            "Can you confirm the address on the account for me?",
            "Would you like me to keep the order open, or cancel it?",
            "Is there anything else I can help with today?",
            "What's the best number to reach you on?",
            "Do you want me to book that in?",
        ),
        "main": ("sarcastic_own",),
        "instead": ("sincere_pleased", "sincere_negative", "sincere_accepting"),
    },
    "procedure": {
        "weight": 1.0,
        "agent": "explains, matter-of-factly, the next step or how something works: a form, an email with a link, a "
        "team that will call back, how long it usually takes",
        "plain": True,
        "examples": (
            "You'll get an email with a link to fill in the form.",
            "Once it's back with us, the team checks it and sends an update.",
            "These usually take three to five working days to go through.",
            "I'll pass this to billing and they'll call you back.",
            "The next step is a quick check of your identity.",
        ),
        "main": ("sarcastic_own",),
        "instead": ("sincere_pleased", "sincere_negative", "sincere_accepting"),
    },
    "kind": {
        "weight": 1.5,
        "agent": "says something warm or helpful: that it is happy to help, that it has noted something on the "
        "account, or that it has done a small thing for the person",
        "plain": True,
        "examples": (
            "Happy to help! I've added a note to your account.",
            "Thanks for bearing with me, I really appreciate it.",
            "I've flagged this so the next person sees it straight away.",
            "No problem at all, I'm glad to look into it for you.",
            "I've sent you a ten dollar voucher for the trouble.",
        ),
        "main": ("sarcastic_own",),
        "instead": ("sincere_pleased", "sincere_negative", "sincere_accepting"),
    },
    "bad_news": {
        "weight": 1.8,
        "agent": "tells the person something they do not want to hear: a delay, a fee, a refusal, a long wait, or no "
        "fix for now",
        "plain": False,
        "examples": (
            "It looks like it's delayed by a few days.",
            "I'm afraid the earliest visit is next Thursday.",
            "There's a forty dollar fee to change that.",
            "I'm sorry, I can't refund that one.",
            "The wait to speak to someone is about an hour.",
        ),
        "main": ("sarcastic_news", "sarcastic_own"),
        "instead": ("sincere_negative", "sincere_accepting"),
    },
    "good_news": {
        "weight": 1.2,
        "agent": "gives good news: the problem is fixed, a refund or credit is on its way, or something will happen "
        "sooner than expected",
        "plain": False,
        "examples": (
            "Good news, your refund went through today.",
            "I've reset it from here, so it should work now.",
            "I can get someone out to you tomorrow morning.",
            "I've waived the fee for you this time.",
            "It's arriving a day early, actually.",
        ),
        "main": ("sincere_pleased",),
        "instead": ("sarcastic_own",),
    },
}

# What the person's answer at the moment does, and a few answers like it (three are shown).
REPLIES: dict[str, dict] = {
    "sarcastic_own": {
        "sarcastic": True,
        "does": "is sarcastic in its own words: it sounds pleased, grateful or impressed and, in the same turn, says "
        "the bad consequence or the complaint itself, so the sarcasm is plain even to someone who never heard "
        "{agent}'s line",
        "examples": (
            "Oh, perfect. So it'll get here after the party, then.",
            "Great, love that. Another week with no internet.",
            "Wonderful, so I get to take another day off work for this.",
            "Fantastic. Three calls and nothing's fixed. Love it.",
            "Oh good, I'll just sit here and wait then, shall I?",
            "Brilliant, so the refund's basically never.",
            "Awesome, so I'm paying for a phone I can't even use.",
            "Lovely. The one week I actually needed it.",
            "Wow, thanks so much. That only took an hour.",
            "Oh, how convenient, right after the trial ends.",
            "Well, isn't that wonderful. The flight's gone and so's my hotel.",
            "Cool, cool. So nobody knows anything. Great.",
            "Super helpful, really. I'm still locked out.",
            "Oh, that's just great, isn't it? Now I'll miss my appointment.",
        ),
    },
    "sarcastic_news": {
        "sarcastic": True,
        "does": "is a short sarcastic answer to that news: pleased or grateful words that, after it, plainly mean the "
        "opposite, saying nothing of its own about what is wrong",
        "examples": (
            "Oh, wonderful.",
            "Perfect. Just perfect.",
            "Great. Love that.",
            "Fantastic.",
            "Thanks a lot.",
            "Oh, good.",
            "Brilliant.",
            "Can't wait.",
            "Lovely.",
            "Sounds great.",
            "Oh, awesome.",
            "Well, isn't that nice.",
        ),
    },
    "sincere_pleased": {
        "sarcastic": False,
        "does": "is pleased and means it, in words like great, perfect, wonderful, love that, thanks so much or that's "
        "really helpful",
        "examples": (
            "Oh, perfect. That's all I needed to know.",
            "Great, thanks so much.",
            "Wonderful, thank you, that's a weight off.",
            "Brilliant, that's really helpful.",
            "Oh, lovely. Thanks for sorting that.",
            "Fantastic, I can work with that.",
            "Awesome, thank you.",
            "That's great news, honestly.",
            "Perfect, yeah, let's do that.",
            "Oh, that's really kind, thank you.",
        ),
    },
    "sincere_negative": {
        "sarcastic": False,
        "does": "is plainly unhappy and says so in straight words, with no sarcasm and no mock praise: annoyed, let down "
        "or worried",
        "examples": (
            "Ugh, that's really annoying. It'll be late, then.",
            "That's so frustrating, I needed it this week.",
            "Oh no. So it won't be here before Friday?",
            "Honestly, that's not good enough.",
            "I'm really disappointed, I paid for express.",
            "That's a pain. I can't keep taking time off.",
            "Okay, I'm not happy about that.",
            "That worries me, to be honest.",
            "Seriously? This is the third time.",
        ),
    },
    "sincere_accepting": {
        "sarcastic": False,
        "does": "takes a downside in good spirit and means it: says the consequence out loud and is fine with it",
        "examples": (
            "That's fine, it'll be a couple of days late but no big deal.",
            "Okay, great, I can wait till next week.",
            "Perfect, I'm not in a rush anyway.",
            "Oh well, these things happen. Thanks for checking.",
            "No worries, Monday's fine.",
            "Alright, that's okay, I'll manage till then.",
            "Fair enough. It is what it is.",
            "Great, so it's just a bit slower. That's fine.",
        ),
    },
}

FRESH = "Write a fresh line that fits this call rather than copying those."


def _weighted(rng: random.Random, table: dict[str, dict]) -> str:
    names = sorted(table)
    return rng.choices(names, weights=[table[n]["weight"] for n in names])[0]


def _examples(rng: random.Random, pool: tuple[str, ...], k: int = 3) -> str:
    return "; ".join(rng.sample(pool, min(k, len(pool))))


def build_prompts(n_per_domain: int, seed: int = SEED, prefix: str = PROMPT_PREFIX) -> list[dict]:
    """`n_per_domain` call prompts for each domain, each fixing the agent's name and company, what the call is about, who
    the person is and how their mood runs, and the moment: what the agent's line does, what the answer the call goes on
    from does, and what the answer given instead does. Kinds steer the writing only: the committee decides every label.
    A third of each domain's calls, drawn at random, have the person's side as speech-to-text output."""
    rng = random.Random(seed)
    out = []
    for domain in sorted(DOMAINS):
        spec = DOMAINS[domain]
        n_asr = round(n_per_domain * ASR_SHARE)
        asr_flags = [True] * n_asr + [False] * (n_per_domain - n_asr)
        rng.shuffle(asr_flags)
        for j in range(n_per_domain):
            agent = rng.choice(voice.AGENT_NAMES)
            moment = _weighted(rng, MOMENTS)
            m = MOMENTS[moment]
            main, instead = rng.choice(m["main"]), rng.choice(m["instead"])
            sarcastic = main if REPLIES[main]["sarcastic"] else instead
            fits = [a for a, v in voice.ARCS.items() if "sarcasm" in v["fits"]]
            arc = voice._weighted(rng, voice.ARCS, fits if sarcastic == main else None)
            p = {
                "prompt_id": f"{prefix}-{domain}-{j:04d}",
                "domain": domain,
                "agent": agent,
                "company": rng.choice(spec["companies"]),
                "scenario": rng.choice(spec["scenarios"]),
                "persona": rng.choice(PERSONAS),
                "arc": arc,
                "moment": moment,
                "main": main,
                "instead": instead,
                "agent_examples": _examples(rng, m["examples"], 2),
                "main_examples": _examples(rng, REPLIES[main]["examples"]),
                "instead_examples": _examples(rng, REPLIES[instead]["examples"]),
                "move_extra": voice._weighted(rng, voice.USER_MOVES),
                "asr": asr_flags[j],
            }
            p["messages"] = messages(p)
            out.append(p)
    return out


def messages(p: dict) -> list[dict]:
    """The prompt: the call as JSON in rounds (the agent, then the person straight after), with the moment as its own
    round holding both answers to the agent's line."""
    spec = DOMAINS[p["domain"]]
    agent = p["agent"]
    m, main, instead = MOMENTS[p["moment"]], REPLIES[p["main"]], REPLIES[p["instead"]]
    xm = voice.USER_MOVES[p["move_extra"]]
    plain = (
        f" {agent}'s line there is not bad news itself: no apology, no 'unfortunately', nothing the person would mind "
        "on its own."
        if m["plain"]
        else ""
    )
    task = (
        f"Write one complete phone call between {spec['intro'].format(agent=agent, company=p['company'])}.\n\n"
        f"What happens: {p['scenario']}.\n"
        f"The person: {p['persona']}. Their mood: {voice.ARCS[p['arc']]['text']}. Let it show in what they say and how; "
        "people seldom name their feelings.\n"
        f"Somewhere in the call the person also {xm['does'].format(agent=agent)}.\n\n"
        f"One round in the middle is the moment. {agent} {m['agent']} (lines like: {p['agent_examples']}; write one of "
        f"{agent}'s own that fits this call).{plain}\n"
        f'- "user" is the person\'s answer, and the call goes on from it: it {main["does"].format(agent=agent)} (the '
        f"kind of thing people say: {p['main_examples']}).\n"
        f'- "user_instead" is another answer the person could have given to the same line from {agent} instead: it '
        f"{instead['does'].format(agent=agent)} (the kind of thing people say: {p['instead_examples']}). It must fit "
        f"the same moment, but the call does not go on from it.\n"
        f"{FRESH} Nobody mentions sarcasm or jokes about it.\n\n"
        "How they speak:\n"
        f"- {agent} sounds like a good voice assistant: warm, clear and brief, with contractions and normal capitals and "
        "punctuation. One or two short sentences a turn, at most 25 words.\n"
        f"- {voice.STYLE_ASR if p['asr'] else voice.STYLE_CLEAN}\n"
        "- The person's turns run from one word to about 25 words: short replies, half sentences, questions, now and "
        "then a longer explanation.\n\n"
        f"Write it as JSON. The first round opens with {agent}'s first words. Each round is "
        f'{{"agent": what {agent} says, "user": what the person says straight after}}. "before" holds the 3 to 5 rounds '
        'before the moment, "moment" is that one round with "user_instead" beside "user", "after" holds the 2 to 4 '
        f'rounds after it, and "agent_last" is {agent}\'s last words. Each string holds only spoken words: no stage '
        "directions, no descriptions of tone or feelings, no sound effects, no emoji."
    )
    return [{"role": "system", "content": voice.SYSTEM}, {"role": "user", "content": task}]


MOMENT_ROUND = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agent", "user", "user_instead"],
    "properties": {
        "agent": {"type": "string"},
        "user": {"type": "string"},
        "user_instead": {"type": "string"},
    },
}


def schema() -> dict:
    """The JSON the teacher fills (vLLM structured output): rounds before the moment, the moment with both answers,
    rounds after, the agent's last words."""
    props = {
        "before": {"type": "array", "items": voice.ROUND},
        "moment": MOMENT_ROUND,
        "after": {"type": "array", "items": voice.ROUND},
        "agent_last": {"type": "string"},
    }
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def parse_call(text: str) -> tuple[list[dict], str, list[str]]:
    """A call's turns as ``voice.parse_json_conversation`` reads them, the moment's answer marked, the answer given
    instead, and the problems met."""
    try:
        obj = json.loads(text)
    except ValueError:
        return [], "", ["not json"]
    moment = dict(obj.get("moment") or {})
    instead = str(moment.pop("user_instead", "") or "")
    turns, problems = voice.parse_json_conversation(json.dumps({**obj, "moment": moment}), "agent")
    return turns, instead, problems


# --- the lines ---------------------------------------------------------------------------------------------------------


def split_of(call: dict) -> str:
    """test for the held domains' calls and a fixed share of the others', dev for another share, by call."""
    if call["domain"] in HELD_DOMAINS:
        return "test"
    u = int(hashlib.sha256(f"{SEED}:split:{call['prompt_id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < TEST_SHARE else "dev" if u < TEST_SHARE + DEV_SHARE else "train"


def moment_index(call: dict) -> int | None:
    """The marked answer's turn, when there is exactly one, the person's, after a line of the agent's."""
    marks = [i for i, t in enumerate(call["turns"]) if t["marked"]]
    if len(marks) != 1 or marks[0] == 0:
        return None
    i = marks[0]
    if call["turns"][i]["who"] != "user" or call["turns"][i - 1]["who"] != "agent":
        return None
    return i


def moment_lines(call: dict) -> tuple[dict, dict] | None:
    """The moment's two answers as candidates ({"i", "text", "previous", "reply"}), or None when either can't be a line:
    a stage direction, too long, both the same, a moment the call says twice, or a plain line of the agent's that
    carries news after all."""
    i = moment_index(call)
    if i is None:
        return None
    agent_line = turn_text(call["turns"][i - 1], call["asr"])
    main = turn_text(call["turns"][i], call["asr"])
    instead = turn_text({"who": "user", "text": call["instead_text"]}, call["asr"])
    if not (agent_line and main and instead) or norm_key(main) == norm_key(instead):
        return None
    elsewhere = {norm_key(t["text"]) for k, t in enumerate(call["turns"]) if k not in (i - 1, i)}
    if {norm_key(call["turns"][i - 1]["text"]), norm_key(call["turns"][i]["text"])} & elsewhere:
        return None
    if MOMENTS[call["moment"]]["plain"] and NEWS_WORDS.search(agent_line):
        return None
    return (
        {"i": i, "text": main, "previous": agent_line, "reply": call["main"], "instead": False},
        {"i": i, "text": instead, "previous": agent_line, "reply": call["instead"], "instead": True},
    )


def ordinary_lines(call: dict) -> list[dict]:
    """The person's other turns that can be lines, each with the agent's line before it."""
    out = []
    for c in voice.candidates(call):
        if c["who"] == "user" and not c["marked"] and c["previous"]:
            out.append({**c, "reply": "ordinary", "instead": False})
    return out


def corpus_line(call: dict, cand: dict, split: str) -> CorpusLine:
    """A line in Tone's generated-line conventions: the person speaks as `player` to the agent, with the agent's line
    before it as `previous`."""
    agent = call["agent"]
    inp = LineInput(
        speaker=PLAYER,
        speaker_kind="player",
        to=agent,
        targets=[agent],
        text=cand["text"],
        previous=f"{agent}: {cand['previous']}"[:MAX_PREVIOUS],
    )
    reply = cand["reply"]
    slices = ["voice_agent", "support", call["domain"], "player", "user", f"reply_{reply}"]
    if reply != "ordinary":
        slices += [
            f"moment_{call['moment']}",
            "sarcasm_seed" if REPLIES[reply]["sarcastic"] else "lookalike_seed",
        ]
    if call["asr"]:
        slices.append("asr_style")
    if call["domain"] in HELD_DOMAINS:
        slices.append("held_domain")
    source_ref = f"{call['prompt_id']}#{cand['i']}{'/instead' if cand['instead'] else ''}"
    return CorpusLine(
        line_id=f"{SOURCE[:2]}-{short_id(SOURCE, source_ref)}",
        source=SOURCE,
        source_ref=source_ref,
        licence=LICENCE,
        split=split,
        group_key=f"gen:{call['prompt_id']}",
        input=inp,
        slices=sorted(set(slices)),
        generator=f"support:{call['moment']}:{reply}" if reply != "ordinary" else "support:ordinary",
    )


def by_construction(reply: str) -> bool | None:
    """What a line was written to be: sarcastic, sincere, or None for the person's other turns."""
    return None if reply == "ordinary" else REPLIES[reply]["sarcastic"]


def meta(call: dict, cand: dict, line: CorpusLine) -> dict:
    """A line's call, its kinds, and the call's turns before it in the words a reader sees (no names): kept beside the
    set for a reader that reads more of the call."""
    before = []
    for t in call["turns"][: cand["i"]]:
        text = turn_text(t, call["asr"])
        if text:
            before.append({"who": t["who"], "text": text})
    keys = (
        "prompt_id",
        "domain",
        "agent",
        "company",
        "scenario",
        "persona",
        "arc",
        "moment",
        "main",
        "instead",
        "asr",
    )
    return {
        "line_id": line.line_id,
        **{k: call[k] for k in keys},
        "split": line.split,
        "reply": cand["reply"],
        "answered_instead": cand["instead"],
        "sarcastic_by_construction": by_construction(cand["reply"]),
        "turn": cand["i"],
        "turns": len(call["turns"]),
        "before": before,
    }


class Keeper:
    """The set's texts as they are taken: a (text, line before) pair once, a short text at most MAX_SHORT times and a
    longer one MAX_LONG times, and no training line the bans hold (an evaluation line's text or pair, as ``bans``
    reads them, the set's own dev and test lines included)."""

    def __init__(
        self, banned_texts: set[str] = frozenset(), banned_pairs: set[tuple[str, str]] = frozenset()
    ):
        self.pairs: set[tuple[str, str]] = set()
        self.texts: Counter = Counter()
        self.banned_texts, self.banned_pairs = set(banned_texts), set(banned_pairs)
        self.dropped: Counter = Counter()

    def fresh(self, cand: dict, split: str) -> bool:
        k, prev = norm_key(cand["text"]), norm_key(cand["previous"] or "")
        if not k:
            return False
        if split == "train" and ((k, prev) in self.banned_pairs or k in self.banned_texts):
            self.dropped["in_an_evaluation_set"] += 1
            return False
        cap = MAX_SHORT if len(k.split()) <= SHORT_WORDS else MAX_LONG
        if (k, prev) in self.pairs or self.texts[k] >= cap:
            self.dropped["repeat"] += 1
            return False
        return True

    def take(self, cand: dict, split: str) -> None:
        k, prev = norm_key(cand["text"]), norm_key(cand["previous"] or "")
        self.pairs.add((k, prev))
        self.texts[k] += 1
        if split != "train":
            texts, pairs = bans([(cand["text"], cand["previous"])])
            self.banned_texts |= texts
            self.banned_pairs |= pairs


def select(calls: list[dict], keeper: Keeper, max_lines: int, seed: int = SEED) -> tuple[list, list, dict]:
    """Lines from the usable calls in a fixed order, the dev and test calls first, until `max_lines`: both of the
    moment's answers (a call whose answers are not both new gives none), and from ORDINARY_SHARE of the calls one of
    the person's other turns. Returns the lines, their metadata and counts."""
    lines, metas, counts = [], [], Counter()
    order = sorted(
        calls,
        key=lambda c: (
            split_of(c) == "train",
            hashlib.sha256(f"{seed}:{c['prompt_id']}".encode()).hexdigest(),
        ),
    )
    for call in order:
        if len(lines) >= max_lines:
            break
        if len(call["turns"]) < 8:
            counts["short_call"] += 1
            continue
        pair = moment_lines(call)
        if pair is None:
            counts["no_moment"] += 1
            continue
        split = split_of(call)
        if not all(keeper.fresh(c, split) for c in pair):
            counts["moment_not_new"] += 1
            continue
        taken = list(pair)
        for c in pair:
            keeper.take(c, split)
        rng = random.Random(f"{seed}:{call['prompt_id']}")
        if rng.random() < ORDINARY_SHARE:
            others = ordinary_lines(call)
            rng.shuffle(others)
            pick = next((c for c in others if keeper.fresh(c, split)), None)
            if pick is not None:
                keeper.take(pick, split)
                taken.append(pick)
        counts["calls"] += 1
        for c in taken[: max(0, max_lines - len(lines))]:
            ln = corpus_line(call, c, split)
            lines.append(ln)
            metas.append(meta(call, c, ln))
    ids = [ln.line_id for ln in lines]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate line ids")
    counts["dropped"] = dict(keeper.dropped)
    return lines, metas, dict(counts)


# --- the cases that showed rc1's misread ----------------------------------------------------------------------------
# Read by every candidate as they stand, never trained on: (id, line, the agent's line before or None, what the line
# should read).

PROBES = (
    ("great_late_own", "Well, that's just great, isn't it? It's gonna be late.", None, "sarcastic"),
    (
        "great_late_after_status",
        "Well, that's just great, isn't it? It's gonna be late.",
        "It's still processing, so it hasn't shipped yet. What would you like to do?",
        "sarcastic",
    ),
    ("wonderful_after_delay", "Oh, wonderful.", "Your delivery is delayed again.", "sarcastic"),
    ("wonderful_own", "Oh, wonderful.", None, "sincere"),
    (
        "helpful_after_refusal",
        "Thanks, that's really helpful.",
        "I'm sorry, I can't help with that.",
        "sarcastic",
    ),
    ("helpful_own", "Thanks, that's really helpful.", None, "sincere"),
    ("wonderful_after_refund", "Oh, wonderful.", "Your refund went through today.", "sincere"),
    (
        "helpful_after_reset",
        "Thanks, that's really helpful.",
        "I've reset your password and sent the link to your email.",
        "sincere",
    ),
    (
        "few_days_after_delay",
        "Oh, wonderful. Another few days, then.",
        "It looks like it's delayed by a few days.",
        "sarcastic",
    ),
    ("few_days_own", "Oh, wonderful. Another few days, then.", None, "sarcastic"),
)


def bans(pairs) -> tuple[set[str], set[tuple[str, str]]]:
    """What a Keeper keeps out of training from (text, line before or None) pairs: texts of BAN_WORDS words or more,
    and every pair."""
    texts, keyed = set(), set()
    for text, previous in pairs:
        k = norm_key(text)
        if len(k.split()) >= BAN_WORDS:
            texts.add(k)
        keyed.add((k, norm_key(previous or "")))
    return texts, keyed


def probe_bans() -> tuple[set[str], set[tuple[str, str]]]:
    return bans((t, p) for _, t, p, _ in PROBES)
