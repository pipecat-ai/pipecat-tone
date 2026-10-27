# Line tags 0.1: the labelling guide

For the line classifier (`plans/tone/PLAN.md`). People, the labelling committee and anyone reading a label use this page. Schema: `contracts/schemas/line_tags.py`. Worked examples: `contracts/schemas/examples/line-tags-0.1.examples.yaml`.

## What you label

One line, spoken in a scene by the player or by a character. You get the speaker, who it was said to (`to`, or nobody: said to the room), the people in the scene (`targets`), and sometimes the line before it (`previous`). The reserved name `player` is the player.

You give:

1. **The speaker's emotion**, as the line shows it, with an intensity.
2. **The act toward each target**: what this line does to that person, with an intensity. Every target gets exactly one act. Most get `none`.
3. **Backchannel**: true only for a short sound of attention or agreement that does not take the turn.

**Label the words, not the friendship.** You do not know how these people feel about each other, and you must not guess. A friend's "you daft sod" is an insult at low intensity unless the words themselves show play (see tease). The game's feeling meters soften it for friends. Your job is only what the line says.

Use `previous` to understand replies and sarcasm. The act is always in the current line.

## Emotion (the speaker's, as the line shows it)

| Class | Means | Examples |
|---|---|---|
| neutral | No clear feeling. Plain information, routine questions and requests | "The smithy's past the well." · "Two loaves, please." |
| happy | Pleasure, warmth, relief, gratitude, excitement | "You're back! Come in, come in." · "That's the best news I've had all week." |
| amused | Finds something funny: joking, teasing, wry | "Ha, you call that a fish?" · "Oh, I'd pay to see that." |
| angry | Irritation up to rage, at someone or something | "Keep your hands off my stall." · "Enough! Get out!" |
| sad | Sorrow, disappointment, regret, loneliness | "She never came home." · "I thought you'd stay a while." |
| afraid | Fear, worry, nervousness, alarm | "Please, I don't want any trouble." · "Did you hear that? Something's out there." |
| surprised | Astonishment, disbelief, a start, good or bad | "You came back?" · "Wait, the bridge is gone?" |
| disgusted | Revulsion at something physical or moral | "Ugh, the smell in here." · "He sold his own brother. Vile." |
| contemptuous | Looking down on someone: scorn, disdain, sneering | "Of course you'd think that." · "Run along, little man." |

Sympathy is not sadness. A comforting line is `neutral` unless it shows the speaker's own sorrow. Excitement about danger is `afraid`, not `happy`.

**Emotion intensity.** `low`: a trace of the feeling. `medium`: clearly felt. `high`: overwhelming, such as shouting, sobbing, terror or rapture. `neutral` is always `low`.

## Act toward a target

| Class | What the line does to the target | Examples |
|---|---|---|
| insult | Demeans them: name-calling, mockery of their worth, looks, wits or work | "You layabout." · "Is that what passes for a sword where you're from?" |
| threat | Says harm will come to them, open or veiled, now or if they do something | "Touch that again and you'll lose the hand." · "Shame if something happened to that cart." |
| accusation | Says they did something wrong, or asks in a way that presumes it | "You took my purse." · "Did you think I wouldn't notice the coins?" |
| tease | A playful jab the words mark as play: laughter, exaggeration, a joking tag | "Ha, still can't hit a barn from ten paces, eh?" · "Careful, you'll strain something thinking that hard. Ha!" |
| demand | Orders them to do or stop something, without asking | "Get out." · "Keep your voice down." |
| dismiss | Brushes them off: ends or refuses the exchange, says they do not matter | "Whatever." · "Not now. Go bother someone else." |
| refusal | Declines something they asked for or offered | "No, I won't sell it." · "I can't take your money." |
| request | Asks them for a favour, an item, help or an action, with a softener | "Could you hold the lantern?" · "Please fetch the healer." |
| thanks | Thanks them | "Thank you, truly." · "Cheers for that." |
| praise | Says something good about them or their work | "Finest blade I've seen in years." · "You've a gift for this." |
| apology | Says sorry to them, or owns a fault toward them | "I'm sorry about your brother." · "That was my doing. I was wrong." |
| comfort | Reassures, consoles or encourages them | "It's all right. You did all you could." · "You'll get there." |
| warning | Tells them of a danger to them, for their own good | "Look out!" · "Don't drink that, it's gone off." |
| none | Nothing social toward this target: greetings, small talk, plain information, questions asking for information, lines about other people | "Morning." · "Where's the blacksmith?" |

**Act intensity.** `low`: mild, softened or implied. `medium`: clear and direct. `high`: strong words, swearing, violence, or shouted. `none` is always `low`.

## The hard edges

1. **Who a line touches.** The person it was said to, anyone it names or clearly points at ("your friend here", "him", "the lot of you"), and anyone it is about who is present. A line about a present third person acts on that person: "Tomas here is a thief", said to Mara, is an accusation toward Tomas and `none` toward Mara. Everyone else in the scene is a bystander and gets `none`.
2. **Groups.** "You're all cowards" and "Everyone out!" act on every target in the group addressed. A warning shouted to the room ("Fire!") is a warning toward every target.
3. **One line can hurt one and please another.** "At least Wren knows how to pour a drink", said to Tomas: insult (low) toward Tomas, praise (low) toward Wren.
4. **Tease or insult.** Tease only when the words show play: a laugh ("ha"), absurd exaggeration, a joking tag ("eh?", "you old goat" said with a grin the words make clear). Without that, a jab is an insult, usually `low`. Friendship never turns an insult into a tease; the meters handle friendship.
5. **Threat in jest.** "Spill my ale again and I'll throw you in the river, ha!" is a tease (play is marked). The same line without the laugh is a threat at `low` or `medium`.
6. **Sarcasm and praise that is really mockery.** Label the meaning, not the surface. "Oh, marvellous work. It only fell off twice." after someone fixed a wheel is an insult. When you cannot tell whether it is sincere, label the surface words.
7. **Request, demand, warning.** "Please", "could you", "would you mind" make a request. A bare order is a demand; its force sets the intensity ("Keep your voice down" low, "Get out. Now." high). An order about a danger to them is a warning ("Get down!").
8. **Refusal or dismiss.** Refusal declines a specific ask. Dismiss rejects the person or the conversation. "No. Now leave me be." is a dismiss (the stronger act).
9. **More than one act toward the same person.** Pick the one with the most social weight, in this order: threat, insult, accusation, demand, dismiss, refusal, apology, thanks, comfort, praise, warning, tease, request, none. Intensity shows any softening: "Sorry, but get out" is a demand at `low`.
10. **Questions.** A question asking for information is `none`. A question that presumes guilt is an accusation. A rhetorical put-down ("Are you stupid?") is an insult.
11. **About themselves or about someone absent.** "I'm such a fool" and "The mayor's a crook" (the mayor is not in the scene) are `none` toward every target. The emotion still counts.
12. **Speech-to-text noise.** Player lines may have no punctuation, misheard words or dropped words ("wheres thomas heading"). Label what the speaker most likely meant.
13. **Cut-off lines.** Label what is there. "Look ou" is still a warning. "Wait, I" is `none` unless the words carry more.
14. **Backchannel.** Short sounds of attention or agreement that do not take the turn: "mm", "uh-huh", "yeah", "right", "go on", "I see". Every act is `none`. Not backchannel: "Yeah, whatever" (dismiss), "No!" (a refusal or surprise), "Wait" (taking the turn), "Get on with it" (a demand).

## For the committee

Answer with the closed sets above and nothing else. Give one act per target, by the exact target name. Do not explain. If a line is ambiguous, label the most likely reading at a lower intensity.
